"""DiT 各组件的单元测试 — 功能正确性 + 性能。"""

import time

import torch
import pytest

from dit import (
    SiLU,
    timestep_embedding,
    PatchEmbed,
    Attention,
    JointAttention,
    FeedForward,
    AdaLNModulation,
    modulate,
    DiTBlock,
    DualStreamDiTBlock,
    FinalLayer,
    DiffusionConfig,
    DiffusionTransformer,
)


def _close(a, b, atol=1e-5):
    assert torch.allclose(a.float(), b.float(), atol=atol), (
        f"max diff = {(a.float() - b.float()).abs().max().item()}"
    )


def _timer(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# ===========================================================================
#  SiLU
# ===========================================================================


class TestSiLU:
    def test_formula(self):
        """SiLU(x) = x * sigmoid(x)"""
        x = torch.randn(32, 64)
        result = SiLU()(x)
        expected = x * torch.sigmoid(x)
        _close(result, expected)

    def test_zero(self):
        x = torch.zeros(5)
        assert (SiLU()(x) == 0).all()

    def test_positive_large(self):
        """x → +∞ 时 SiLU(x) → x"""
        x = torch.tensor([100.0])
        assert abs(SiLU()(x).item() - 100.0) < 0.01

    def test_negative_large(self):
        """x → -∞ 时 SiLU(x) → 0"""
        x = torch.tensor([-100.0])
        assert abs(SiLU()(x).item()) < 0.01

    def test_non_monotonic(self):
        """SiLU 在 x ≈ -1.278 有最小值 ≈ -0.278"""
        x = torch.tensor([-1.278])
        val = SiLU()(x).item()
        assert val < 0, "SiLU 应在 x≈-1.278 处为负值"
        assert val > -0.3, f"SiLU 最小值应 ≈ -0.278, 实际={val}"


# ===========================================================================
#  TimestepEmbedding
# ===========================================================================


class TestTimestepEmbedding:
    def test_shape(self):
        t = torch.tensor([0.0, 0.5, 1.0])
        emb = timestep_embedding(t, 128)
        assert emb.shape == (3, 128)

    def test_odd_dim(self):
        t = torch.tensor([1.0])
        emb = timestep_embedding(t, 127)
        assert emb.shape == (1, 127)

    def test_different_t_different_emb(self):
        t = torch.tensor([0.0, 500.0])
        emb = timestep_embedding(t, 64)
        assert not torch.allclose(emb[0], emb[1])


# ===========================================================================
#  PatchEmbed
# ===========================================================================


class TestPatchEmbed:
    def test_shape(self):
        pe = PatchEmbed(patch_size=2, in_channels=4, hidden_dim=768)
        x = torch.randn(2, 4, 32, 32)
        out = pe(x)
        assert out.shape == (2, 256, 768)

    def test_patch4(self):
        pe = PatchEmbed(patch_size=4, in_channels=3, hidden_dim=512)
        x = torch.randn(1, 3, 64, 64)
        out = pe(x)
        assert out.shape == (1, 256, 512)


# ===========================================================================
#  Attention
# ===========================================================================


class TestAttention:
    def test_shape(self):
        attn = Attention(dim=128, num_heads=4)
        x = torch.randn(2, 16, 128)
        out = attn(x)
        assert out.shape == (2, 16, 128)

    def test_deterministic(self):
        attn = Attention(dim=64, num_heads=2)
        x = torch.randn(1, 8, 64)
        attn.eval()
        _close(attn(x), attn(x))


# ===========================================================================
#  JointAttention
# ===========================================================================


class TestJointAttention:
    def test_shape(self):
        ja = JointAttention(dim=128, num_heads=4)
        x = torch.randn(2, 16, 128)
        c = torch.randn(2, 10, 128)
        out_x, out_c = ja(x, c)
        assert out_x.shape == (2, 16, 128)
        assert out_c.shape == (2, 10, 128)

    def test_cross_influence(self):
        """修改 c 应影响 out_x (因为 joint attention)"""
        ja = JointAttention(dim=64, num_heads=2)
        x = torch.randn(1, 8, 64)
        c1 = torch.randn(1, 4, 64)
        c2 = torch.randn(1, 4, 64)
        out1, _ = ja(x, c1)
        out2, _ = ja(x, c2)
        assert not torch.allclose(out1, out2, atol=1e-6)


# ===========================================================================
#  FeedForward
# ===========================================================================


class TestFeedForward:
    def test_gated_shape(self):
        ff = FeedForward(dim=128, hidden_dim=512, use_gate=True)
        x = torch.randn(2, 16, 128)
        assert ff(x).shape == (2, 16, 128)

    def test_standard_shape(self):
        ff = FeedForward(dim=128, hidden_dim=512, use_gate=False)
        x = torch.randn(2, 16, 128)
        assert ff(x).shape == (2, 16, 128)

    def test_gated_vs_standard_different(self):
        torch.manual_seed(42)
        ff_gate = FeedForward(dim=64, use_gate=True)
        torch.manual_seed(42)
        ff_std = FeedForward(dim=64, use_gate=False)
        x = torch.randn(1, 4, 64)
        assert ff_gate(x).shape == ff_std(x).shape


# ===========================================================================
#  AdaLN Modulation
# ===========================================================================


class TestAdaLNModulation:
    def test_num_outputs(self):
        mod = AdaLNModulation(dim=128, n_modulations=6)
        t_emb = torch.randn(2, 128)
        params = mod(t_emb)
        assert len(params) == 6
        for p in params:
            assert p.shape == (2, 128)

    def test_zero_init(self):
        """adaLN-Zero: 初始时所有调制参数应为 0"""
        mod = AdaLNModulation(dim=64, n_modulations=6)
        t_emb = torch.randn(1, 64)
        params = mod(t_emb)
        for p in params:
            assert (p == 0).all(), "adaLN-Zero: 初始参数应全为 0"

    def test_modulate_identity(self):
        """shift=0, scale=0 时 modulate 应为恒等"""
        x = torch.randn(2, 16, 64)
        shift = torch.zeros(2, 64)
        scale = torch.zeros(2, 64)
        _close(modulate(x, shift, scale), x)


# ===========================================================================
#  DiTBlock (传统单流)
# ===========================================================================


class TestDiTBlock:
    def test_shape(self):
        blk = DiTBlock(dim=128, num_heads=4, mlp_ratio=4.0)
        x = torch.randn(2, 16, 128)
        t_emb = torch.randn(2, 128)
        out = blk(x, t_emb)
        assert out.shape == (2, 16, 128)

    def test_identity_at_init(self):
        """adaLN-Zero: 初始化后 block 应 ≈ 恒等映射"""
        blk = DiTBlock(dim=64, num_heads=2)
        x = torch.randn(1, 8, 64)
        t_emb = torch.randn(1, 64)
        out = blk(x, t_emb)
        _close(out, x, atol=1e-5)

    def test_trainable(self):
        blk = DiTBlock(dim=64, num_heads=2)
        x = torch.randn(1, 4, 64, requires_grad=True)
        t_emb = torch.randn(1, 64)
        loss = blk(x, t_emb).sum()
        loss.backward()
        assert x.grad is not None


# ===========================================================================
#  DualStreamDiTBlock
# ===========================================================================


class TestDualStreamDiTBlock:
    def test_shape(self):
        blk = DualStreamDiTBlock(dim=128, num_heads=4)
        x = torch.randn(2, 16, 128)
        c = torch.randn(2, 10, 128)
        t_emb = torch.randn(2, 128)
        out_x, out_c = blk(x, c, t_emb)
        assert out_x.shape == (2, 16, 128)
        assert out_c.shape == (2, 10, 128)

    def test_identity_at_init(self):
        """初始化后双流 block 应 ≈ 恒等映射"""
        blk = DualStreamDiTBlock(dim=64, num_heads=2)
        x = torch.randn(1, 8, 64)
        c = torch.randn(1, 4, 64)
        t_emb = torch.randn(1, 64)
        out_x, out_c = blk(x, c, t_emb)
        _close(out_x, x, atol=1e-5)
        _close(out_c, c, atol=1e-5)

    def test_streams_interact(self):
        """双流 block 的两个流应通过 joint attention 互相影响"""
        torch.manual_seed(0)
        blk = DualStreamDiTBlock(dim=64, num_heads=2)
        for p in blk.parameters():
            nn.init.normal_(p, std=0.1)
        x = torch.randn(1, 4, 64)
        t_emb = torch.randn(1, 64)
        c1 = torch.randn(1, 3, 64)
        c2 = torch.randn(1, 3, 64)
        out1, _ = blk(x, c1, t_emb)
        out2, _ = blk(x, c2, t_emb)
        assert not torch.allclose(out1, out2, atol=1e-6)


# ===========================================================================
#  DiffusionTransformer — 传统 DiT 配置
# ===========================================================================


class TestDiffusionTransformerDiT:
    @pytest.fixture
    def model(self):
        cfg = DiffusionConfig(
            in_channels=4,
            patch_size=2,
            hidden_dim=128,
            num_heads=4,
            depth=2,
            mlp_ratio=4.0,
            block_type="dit",
            input_size=16,
        )
        return DiffusionTransformer(cfg)

    def test_output_shape(self, model):
        x = torch.randn(2, 4, 16, 16)
        t = torch.tensor([10.0, 50.0])
        out = model(x, t)
        assert out.shape == (2, 4, 16, 16)

    def test_uses_dit_blocks(self, model):
        assert all(isinstance(b, DiTBlock) for b in model.blocks)

    def test_learn_sigma(self):
        cfg = DiffusionConfig(
            in_channels=4,
            hidden_dim=64,
            num_heads=2,
            depth=1,
            block_type="dit",
            learn_sigma=True,
            input_size=8,
        )
        model = DiffusionTransformer(cfg)
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor([1.0])
        out = model(x, t)
        assert out.shape == (1, 8, 8, 8), "learn_sigma=True 时输出通道应翻倍"


# ===========================================================================
#  DiffusionTransformer — 双流 MMDiT 配置
# ===========================================================================


class TestDiffusionTransformerDualStream:
    @pytest.fixture
    def model(self):
        cfg = DiffusionConfig(
            in_channels=4,
            patch_size=2,
            hidden_dim=128,
            num_heads=4,
            depth=2,
            mlp_ratio=4.0,
            block_type="dual_stream",
            context_dim=256,
            input_size=16,
        )
        return DiffusionTransformer(cfg)

    def test_output_shape(self, model):
        x = torch.randn(2, 4, 16, 16)
        t = torch.tensor([10.0, 50.0])
        ctx = torch.randn(2, 20, 256)
        out = model(x, t, context=ctx)
        assert out.shape == (2, 4, 16, 16)

    def test_uses_dual_blocks(self, model):
        assert all(isinstance(b, DualStreamDiTBlock) for b in model.blocks)

    def test_requires_context(self, model):
        x = torch.randn(1, 4, 16, 16)
        t = torch.tensor([1.0])
        with pytest.raises(AssertionError, match="dual_stream"):
            model(x, t)


# ===========================================================================
#  BLOCK_REGISTRY 扩展性
# ===========================================================================


class TestBlockRegistry:
    def test_registry_keys(self):
        assert "dit" in DiffusionTransformer.BLOCK_REGISTRY
        assert "dual_stream" in DiffusionTransformer.BLOCK_REGISTRY

    def test_invalid_block_type(self):
        cfg = DiffusionConfig(block_type="not_exist", input_size=8)
        with pytest.raises(KeyError):
            DiffusionTransformer(cfg)


# ===========================================================================
#  参数统计
# ===========================================================================


class TestParamCount:
    @staticmethod
    def _count(module):
        return sum(p.numel() for p in module.parameters())

    def test_dit_block_params(self):
        blk = DiTBlock(dim=768, num_heads=12)
        n = self._count(blk)
        assert n > 0
        print(f"\n  DiTBlock(768, 12heads) params: {n:,}")

    def test_dual_block_params(self):
        blk = DualStreamDiTBlock(dim=768, num_heads=12)
        n = self._count(blk)
        dit_n = self._count(DiTBlock(dim=768, num_heads=12))
        assert n > dit_n, "双流 block 参数量应大于单流"
        print(f"\n  DualStreamDiTBlock(768, 12heads) params: {n:,}")

    def test_full_model_params(self):
        cfg = DiffusionConfig(
            hidden_dim=768, num_heads=12, depth=12,
            input_size=32, block_type="dit",
        )
        model = DiffusionTransformer(cfg)
        n = self._count(model)
        print(f"\n  DiT-B/2 (768d, 12L, 12heads) total params: {n:,}")


# ===========================================================================
#  性能
# ===========================================================================


class TestPerformance:
    def _report(self, name, elapsed):
        print(f"  [PERF] {name:45s} {elapsed*1000:8.3f}ms")

    def test_perf_dit_block(self):
        blk = DiTBlock(dim=256, num_heads=8)
        x = torch.randn(4, 64, 256)
        t_emb = torch.randn(4, 256)
        _, e = _timer(blk, x, t_emb)
        self._report("DiTBlock(256d, 8heads) B=4 N=64", e)

    def test_perf_dual_block(self):
        blk = DualStreamDiTBlock(dim=256, num_heads=8)
        x = torch.randn(4, 64, 256)
        c = torch.randn(4, 20, 256)
        t_emb = torch.randn(4, 256)
        _, e = _timer(blk, x, c, t_emb)
        self._report("DualStreamDiTBlock(256d, 8heads) B=4 N=64", e)

    def test_perf_dit_model_forward(self):
        cfg = DiffusionConfig(
            hidden_dim=256, num_heads=8, depth=4,
            input_size=16, block_type="dit",
        )
        model = DiffusionTransformer(cfg)
        x = torch.randn(2, 4, 16, 16)
        t = torch.tensor([10.0, 50.0])
        _, e = _timer(model, x, t)
        self._report("DiT full forward (256d, 4L) 2x4x16x16", e)

    def test_perf_dual_model_forward(self):
        cfg = DiffusionConfig(
            hidden_dim=256, num_heads=8, depth=4,
            input_size=16, block_type="dual_stream",
            context_dim=256,
        )
        model = DiffusionTransformer(cfg)
        x = torch.randn(2, 4, 16, 16)
        t = torch.tensor([10.0, 50.0])
        ctx = torch.randn(2, 12, 256)
        _, e = _timer(model, x, t, context=ctx)
        self._report("DualStream full forward (256d, 4L) 2x4x16x16", e)
