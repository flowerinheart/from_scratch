"""DiT (Diffusion Transformer) 实现 —— 传统单流 & 双流 decoder block + Diffusion 框架。

本文件实现:
    1. DiTBlock          — 传统 DiT decoder block (adaLN-Zero)
    2. DualStreamDiTBlock — 双流 DiT decoder block (MMDiT / SD3 风格, joint attention)
    3. DiffusionTransformer — 可配置 decoder block 类型的 Diffusion 模型框架

DiT 是由多个 block 堆叠而成, 每个 block 用 adaLN-Zero 注入时间步条件:
    - adaLN: 从 t_emb 回归出 (scale, shift, gate) 调制 LayerNorm 的输出
    - Zero-init: gate 参数初始化为 0, 训练初期每个 block ≈ 恒等映射

门控单元常用算子: sigmoid, tanh, SiLU(Swish), GELU, GLU, SwiGLU
    SiLU 公式: f(x) = x · σ(x) = x / (1 + e^{-x})
    优点:
        - 平滑可微 (不像 ReLU 在 0 处不可微)
        - 非单调: x ≈ -1.278 附近有最小值 ≈ -0.278, 允许小负信号通过
        - 自门控: 输入自己作为门控信号 (x 控制 σ(x) 的开闭)
        - 上无界 (像 ReLU), 梯度不会饱和消失
        - 在 DiT/LLM 中广泛使用, 实验效果优于 ReLU/GELU
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
#  基础组件
# ===========================================================================


class SiLU(nn.Module):
    """SiLU (Sigmoid Linear Unit) / Swish 激活函数。

    f(x) = x · σ(x) = x / (1 + e^{-x})

    性质:
        f(0)  = 0,  f'(0) = 0.5
        lim x→+∞  f(x) = x    (趋近恒等)
        lim x→-∞  f(x) = 0    (趋近零)
        最小值 ≈ -0.278 (在 x ≈ -1.278)

    与其他激活函数对比:
        ReLU:     max(0, x)              — 不光滑, 负半轴梯度为 0 (dying ReLU)
        GELU:     x · Φ(x)              — 用正态 CDF, 计算略贵
        Swish-β:  x · σ(βx)             — SiLU 是 β=1 的特例
        Mish:     x · tanh(softplus(x)) — 更光滑但计算更贵
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def timestep_embedding(
    t: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    """正弦位置编码 — 将标量时间步 t 编码为 dim 维向量。

    PE(t, 2i)   = sin(t / max_period^{2i/dim})
    PE(t, 2i+1) = cos(t / max_period^{2i/dim})
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


class PatchEmbed(nn.Module):
    """Patch Embedding: 将 2D 图像切分为 patch 并线性投影。

    (B, C, H, W) → (B, num_patches, hidden_dim)
    等价于 stride=patch_size 的卷积。
    """

    def __init__(
        self, patch_size: int = 2, in_channels: int = 4, hidden_dim: int = 768
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                       # (B, D, H/p, W/p)
        return x.flatten(2).transpose(1, 2)     # (B, N, D)


# ===========================================================================
#  注意力
# ===========================================================================


class Attention(nn.Module):
    """多头自注意力 (Multi-Head Self-Attention)。

    Q, K, V 均来自同一输入, 经 softmax 加权求和。
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class JointAttention(nn.Module):
    """联合注意力 (Joint Attention) — 双流共享 KV 空间。

    两个流 (image x + text c) 各自生成 Q/K/V,
    将 Q/K/V 拼接后统一做注意力, 再拆分回各自的输出。

    这使得 image token 可以 attend to text token (反之亦然),
    而每个流保持独立的投影参数。
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_c = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_x = nn.Linear(dim, dim)
        self.proj_c = nn.Linear(dim, dim)

    def _to_qkv(
        self, linear: nn.Linear, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = tokens.shape
        qkv = (
            linear(tokens)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        return qkv.unbind(0)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Nx, C = x.shape
        Nc = c.shape[1]
        q_x, k_x, v_x = self._to_qkv(self.qkv_x, x)
        q_c, k_c, v_c = self._to_qkv(self.qkv_c, c)
        q = torch.cat([q_x, q_c], dim=2)
        k = torch.cat([k_x, k_c], dim=2)
        v = torch.cat([v_x, v_c], dim=2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Nx + Nc, C)
        out_x, out_c = out[:, :Nx], out[:, Nx:]
        return self.proj_x(out_x), self.proj_c(out_c)


# ===========================================================================
#  前馈网络 (FFN) — 标准 GELU 与 SwiGLU 门控
# ===========================================================================


class FeedForward(nn.Module):
    """前馈网络, 可选 SiLU 门控 (SwiGLU) 或标准 GELU。

    标准 FFN:
        x → Linear → GELU → Linear

    SwiGLU FFN (LLaMA/PaLM 风格):
        x → [W_gate(x), W_up(x)] → SiLU(gate) ⊙ up → W_down → out

    SwiGLU 的门控机制:
        - gate 分支通过 SiLU 产生 [0, +∞) 的门控信号
        - up 分支提供要被门控的特征
        - 乘积 = 门控后的特征, 实现了可学习的特征筛选
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        use_gate: bool = True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Linear(dim, hidden_dim, bias=False)
            self.up = nn.Linear(dim, hidden_dim, bias=False)
            self.down = nn.Linear(hidden_dim, dim, bias=False)
            self.act = SiLU()
        else:
            self.fc1 = nn.Linear(dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, dim)
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gate:
            return self.down(self.act(self.gate(x)) * self.up(x))
        return self.fc2(self.act(self.fc1(x)))


# ===========================================================================
#  adaLN 调制 (Adaptive Layer Normalization Modulation)
# ===========================================================================


class AdaLNModulation(nn.Module):
    """从条件嵌入 (如 t_emb) 回归 adaLN 调制参数。

    流程: t_emb → SiLU → Linear → chunk → (param_1, param_2, ..., param_n)

    在 DiT 中 n_modulations=6, 输出:
        (shift_1, scale_1, gate_1, shift_2, scale_2, gate_2)
        分别用于注意力子层和 FFN 子层的 LayerNorm 调制 + 残差门控。

    为什么用 SiLU?
        - 自门控特性让 t_emb 中的强信号被放大, 弱信号被抑制
        - 不像 sigmoid 会饱和, SiLU 上无界, 梯度保持畅通
        - 平滑性保证了调制参数随 t 变化时的连续性

    Zero-init: weight 和 bias 初始化为 0
        → 训练初始时所有调制参数为 0
        → scale=0, shift=0, gate=0
        → block 退化为恒等映射 (类似 ResNet 的 zero-init 技巧)
    """

    def __init__(self, dim: int, n_modulations: int = 6):
        super().__init__()
        self.n_modulations = n_modulations
        self.act = SiLU()
        self.linear = nn.Linear(dim, dim * n_modulations)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, t_emb: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.linear(self.act(t_emb)).chunk(self.n_modulations, dim=-1)


def modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """adaLN 调制: norm(x) * (1 + scale) + shift。

    scale/shift 的 shape 为 (B, D), 需要 unsqueeze(1) 广播到 (B, N, D)。
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ===========================================================================
#  传统 DiT Block (adaLN-Zero) — 单流
# ===========================================================================


class DiTBlock(nn.Module):
    """传统 DiT Decoder Block — adaLN-Zero 架构。

    数据流:
        ┌─────────────────────────────────────────┐
        │ t_emb → SiLU → Linear                  │
        │ → (shift1, scale1, gate1,               │
        │    shift2, scale2, gate2)               │
        │                                         │
        │ h = LN(x) * (1 + scale1) + shift1      │ ← adaLN 调制
        │ h = MultiHeadSelfAttention(h)           │
        │ x = x + gate1 · h                      │ ← 门控残差
        │                                         │
        │ h = LN(x) * (1 + scale2) + shift2      │
        │ h = FFN(h)                              │
        │ x = x + gate2 · h                      │ ← 门控残差
        └─────────────────────────────────────────┘

    gate 初始化为 0 → 训练初期 block ≈ 恒等映射 → 稳定训练。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        use_gate_ffn: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = FeedForward(dim, int(dim * mlp_ratio), use_gate=use_gate_ffn)
        self.adaLN = AdaLNModulation(dim, n_modulations=6)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(t_emb)
        h = modulate(self.norm1(x), shift1, scale1)
        h = self.attn(h)
        x = x + gate1.unsqueeze(1) * h
        h = modulate(self.norm2(x), shift2, scale2)
        h = self.ffn(h)
        x = x + gate2.unsqueeze(1) * h
        return x


# ===========================================================================
#  双流 DiT Block (MMDiT / SD3 风格)
# ===========================================================================


class DualStreamDiTBlock(nn.Module):
    """双流 DiT Decoder Block — MMDiT / SD3 风格。

    两个独立的流 (x = 图像, c = 文本/条件), 仅在注意力层共享 KV 空间。

    数据流:
        ┌────────────────────────────────────────────────────┐
        │  mod_x = adaLN_x(t_emb) → (s1,sc1,g1,s2,sc2,g2)  │
        │  mod_c = adaLN_c(t_emb) → (s1,sc1,g1,s2,sc2,g2)  │
        │                                                    │
        │  h_x = adaLN(LN(x), s1_x, sc1_x)                 │
        │  h_c = adaLN(LN(c), s1_c, sc1_c)                 │
        │  h_x, h_c = JointAttention(h_x, h_c)    ← 联合   │
        │  x = x + g1_x · h_x                               │
        │  c = c + g1_c · h_c                               │
        │                                                    │
        │  h_x = adaLN(LN(x), s2_x, sc2_x)                 │
        │  x = x + g2_x · FFN_x(h_x)             ← 独立    │
        │  h_c = adaLN(LN(c), s2_c, sc2_c)                 │
        │  c = c + g2_c · FFN_c(h_c)              ← 独立    │
        └────────────────────────────────────────────────────┘

    双流 vs 单流:
        - 单流: 把图像和文本 token 拼接后统一过同一个 block
        - 双流: 各有独立的 LN / FFN / 调制参数, 仅在注意力层交互
        - 优势: 不同模态有不同的特征分布, 独立参数更灵活
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        use_gate_ffn: bool = True,
    ):
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)
        self.norm1_x = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm1_c = nn.LayerNorm(dim, elementwise_affine=False)
        self.joint_attn = JointAttention(dim, num_heads)
        self.norm2_x = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2_c = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn_x = FeedForward(dim, mlp_dim, use_gate=use_gate_ffn)
        self.ffn_c = FeedForward(dim, mlp_dim, use_gate=use_gate_ffn)
        self.adaLN_x = AdaLNModulation(dim, n_modulations=6)
        self.adaLN_c = AdaLNModulation(dim, n_modulations=6)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, t_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s1x, sc1x, g1x, s2x, sc2x, g2x = self.adaLN_x(t_emb)
        s1c, sc1c, g1c, s2c, sc2c, g2c = self.adaLN_c(t_emb)
        h_x = modulate(self.norm1_x(x), s1x, sc1x)
        h_c = modulate(self.norm1_c(c), s1c, sc1c)
        h_x, h_c = self.joint_attn(h_x, h_c)
        x = x + g1x.unsqueeze(1) * h_x
        c = c + g1c.unsqueeze(1) * h_c
        h_x = modulate(self.norm2_x(x), s2x, sc2x)
        h_c = modulate(self.norm2_c(c), s2c, sc2c)
        x = x + g2x.unsqueeze(1) * self.ffn_x(h_x)
        c = c + g2c.unsqueeze(1) * self.ffn_c(h_c)
        return x, c


# ===========================================================================
#  最终输出层
# ===========================================================================


class FinalLayer(nn.Module):
    """最终输出层: adaLN → Linear, 将隐层 token 映射回 patch 像素空间。"""

    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN = AdaLNModulation(dim, n_modulations=2)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(t_emb)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# ===========================================================================
#  Diffusion Transformer 框架 — decoder block 类型作为配置项
# ===========================================================================


@dataclass
class DiffusionConfig:
    """DiffusionTransformer 配置。

    Attributes:
        block_type: "dit" — 传统单流 DiT block
                    "dual_stream" — 双流 MMDiT block
    """
    in_channels: int = 4
    patch_size: int = 2
    hidden_dim: int = 768
    num_heads: int = 12
    depth: int = 12
    mlp_ratio: float = 4.0
    block_type: str = "dit"
    context_dim: Optional[int] = None
    use_gate_ffn: bool = True
    learn_sigma: bool = False
    out_channels: Optional[int] = None
    input_size: int = 32


class DiffusionTransformer(nn.Module):
    """Diffusion Transformer — 可配置 decoder block 类型的扩散模型框架。

    整体流程:
        Input (B, C, H, W)
            │
            ▼
        PatchEmbed → (B, N, D)       将图像 patch 化
            │
            ▼
        + PositionEmbed              可学习位置编码
            │
            ▼
        TimestepMLP(t) → t_emb      时间步编码: sinusoidal → Linear → SiLU → Linear
            │
            ▼
        ┌──────────────────────┐
        │  DecoderBlock × depth │ ← t_emb 条件注入
        │  (DiT 或 DualStream) │ ← 由 config.block_type 决定
        └──────────────────────┘
            │
            ▼
        FinalLayer → (B, N, p²·C)   adaLN + Linear 映射到像素空间
            │
            ▼
        Unpatchify → (B, C, H, W)   还原为图像

    配置示例:
        # 传统 DiT
        DiffusionConfig(block_type="dit", depth=12, hidden_dim=768)

        # 双流 MMDiT (需提供 context)
        DiffusionConfig(block_type="dual_stream", context_dim=1024, depth=24)
    """

    BLOCK_REGISTRY = {
        "dit": DiTBlock,
        "dual_stream": DualStreamDiTBlock,
    }

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        out_ch = config.out_channels or config.in_channels
        if config.learn_sigma:
            out_ch *= 2
        self.out_channels = out_ch
        self.is_dual = config.block_type == "dual_stream"

        # ---------- Input ----------
        self.patch_embed = PatchEmbed(
            config.patch_size, config.in_channels, config.hidden_dim
        )
        num_patches = (config.input_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, config.hidden_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        # ---------- Timestep MLP ----------
        dim = config.hidden_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(dim, dim), SiLU(), nn.Linear(dim, dim)
        )

        # ---------- Context projection (双流模式) ----------
        if self.is_dual:
            ctx_dim = config.context_dim or dim
            self.ctx_proj = nn.Linear(ctx_dim, dim)

        # ---------- Decoder blocks (按配置选择类型) ----------
        block_cls = self.BLOCK_REGISTRY[config.block_type]
        self.blocks = nn.ModuleList()
        for _ in range(config.depth):
            self.blocks.append(
                block_cls(
                    dim=dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    use_gate_ffn=config.use_gate_ffn,
                )
            )

        # ---------- Output ----------
        self.final = FinalLayer(dim, config.patch_size, out_ch)

    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """(B, N, p²·C_out) → (B, C_out, H, W)"""
        p = self.config.patch_size
        hp, wp = h // p, w // p
        x = x.reshape(x.shape[0], hp, wp, p, p, self.out_channels)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(
            x.shape[0], self.out_channels, h, w
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 噪声图像 (B, C, H, W)
            t: 时间步 (B,)
            context: 文本条件 (B, L, context_dim), 双流模式必须提供

        Returns:
            预测噪声 ε_θ (B, C, H, W) — 或 (B, 2C, H, W) 若 learn_sigma=True
        """
        B, _, H, W = x.shape
        x = self.patch_embed(x) + self.pos_embed
        t_emb = self.t_mlp(
            timestep_embedding(t, self.config.hidden_dim)
        )
        if self.is_dual:
            assert context is not None, "dual_stream 模式需要提供 context"
            c = self.ctx_proj(context)
            for blk in self.blocks:
                x, c = blk(x, c, t_emb)
        else:
            for blk in self.blocks:
                x = blk(x, t_emb)
        x = self.final(x, t_emb)
        return self.unpatchify(x, H, W)
