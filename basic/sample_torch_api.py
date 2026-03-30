"""PyTorch 最常用算子的手动复现 —— 教育目的，展示每个算子的核心原理。

每个 custom_xxx 函数不调用对应的 torch.xxx / F.xxx，
而是用更底层的操作 (as_strided, 索引赋值, 基础运算) 来复现。

算子分类：
    形状操作 (1-5):    reshape, transpose, permute, squeeze, unsqueeze
    拼接拆分 (6-9):    cat, stack, split, chunk
    索引操作 (10-11):  gather, scatter
    矩阵运算 (12-14):  matmul, bmm, linear
    卷积池化 (15):      conv2d (img2col)
    激活归一 (16-19):   relu, softmax, layer_norm, embedding
    损失函数 (20):      cross_entropy
    索引读写 (21-22):   getitem, setitem
"""

import math

import torch


# ===========================================================================
#  辅助函数
# ===========================================================================

def _compute_strides(shape):
    """计算 row-major (C-order) 步长。"""
    if not shape:
        return []
    strides = [0] * len(shape)
    strides[-1] = 1
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


# ===========================================================================
#  1. reshape — 形状变换
# ===========================================================================

def custom_reshape(x, shape):
    """reshape = 重新计算 stride，不改变底层数据排列。

    核心原理：
    - 连续 tensor：只修改 shape + strides 元数据，零拷贝
    - 非连续 tensor：先 contiguous() 拷贝为连续，再修改元数据
    - 支持 -1 自动推断一个维度

    内存视角（同一块内存，不同的"解读方式"）：
        [1,2,3,4,5,6]  shape=(2,3) stride=(3,1)
                        shape=(3,2) stride=(2,1)
    """
    shape = list(shape)
    if -1 in shape:
        neg_idx = shape.index(-1)
        known = 1
        for i, s in enumerate(shape):
            if s != -1:
                known *= s
        shape[neg_idx] = x.numel() // known
    if x.is_contiguous():
        return torch.as_strided(x, shape, _compute_strides(shape),
                                x.storage_offset())
    x_c = x.contiguous()
    return torch.as_strided(x_c, shape, _compute_strides(shape),
                            x_c.storage_offset())


# ===========================================================================
#  2. transpose — 维度转置
# ===========================================================================

def custom_transpose(x, dim0, dim1):
    """transpose = 交换两个维度的 size 和 stride。

    核心原理：
    - 不移动任何数据！只交换 shape 和 stride 中的两个条目
    - 结果通常是非连续的（stride 不再递减）

    示例：
        shape=(2,3), stride=(3,1)
     →  shape=(3,2), stride=(1,3)   ← 同一块内存，按列优先"解读"
    """
    dim0 = dim0 if dim0 >= 0 else x.dim() + dim0
    dim1 = dim1 if dim1 >= 0 else x.dim() + dim1
    new_shape = list(x.shape)
    new_strides = list(x.stride())
    new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
    new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]
    return torch.as_strided(x, new_shape, new_strides, x.storage_offset())


# ===========================================================================
#  3. permute — 维度重排
# ===========================================================================

def custom_permute(x, dims):
    """permute = 按任意顺序重排所有维度。

    核心原理：
    - transpose 只能交换 2 个维度，permute 可以一次性重排所有维度
    - 同样只修改 shape 和 stride，不移动数据

    示例 (NCHW → NHWC)：
        permute([0, 2, 3, 1]) → shape/stride 按 [0,2,3,1] 顺序重排
    """
    new_shape = [x.shape[d] for d in dims]
    new_strides = [x.stride()[d] for d in dims]
    return torch.as_strided(x, new_shape, new_strides, x.storage_offset())


# ===========================================================================
#  4. squeeze — 去掉 size=1 维度
# ===========================================================================

def custom_squeeze(x, dim=None):
    """squeeze = 从 shape/stride 中去掉 size=1 的条目。

    核心原理：
    - size=1 的维度"不占额外空间"（stride 不会导致跨越更多元素）
    - 只需删除该条目，底层数据完全不动
    """
    old_shape = list(x.shape)
    old_strides = list(x.stride())
    if dim is not None:
        dim = dim if dim >= 0 else x.dim() + dim
        if old_shape[dim] != 1:
            return x
        new_shape = old_shape[:dim] + old_shape[dim + 1:]
        new_strides = old_strides[:dim] + old_strides[dim + 1:]
    else:
        new_shape, new_strides = [], []
        for s, st in zip(old_shape, old_strides):
            if s != 1:
                new_shape.append(s)
                new_strides.append(st)
    if not new_shape:
        return torch.as_strided(x, [], [], x.storage_offset())
    return torch.as_strided(x, new_shape, new_strides, x.storage_offset())


# ===========================================================================
#  5. unsqueeze — 插入 size=1 维度
# ===========================================================================

def custom_unsqueeze(x, dim):
    """unsqueeze = 在指定位置插入一个 size=1 的维度。

    核心原理：
    - 新维度的 stride 值不影响实际寻址（size=1，永远不会被用来跳转）
    - 约定设为"连续时应有的 stride"，保证 is_contiguous() 判断正确
    """
    dim = dim if dim >= 0 else x.dim() + 1 + dim
    new_shape = list(x.shape)
    new_strides = list(x.stride())
    if dim < x.dim():
        new_stride = x.stride()[dim] * x.shape[dim]
    else:
        new_stride = 1
    new_shape.insert(dim, 1)
    new_strides.insert(dim, new_stride)
    return torch.as_strided(x, new_shape, new_strides, x.storage_offset())


# ===========================================================================
#  6. cat — 拼接
# ===========================================================================

def custom_cat(tensors, dim=0):
    """cat = 沿已有维度拼接多个 tensor。

    核心原理：
    1. 计算输出 shape（拼接维 = 各 tensor 该维之和，其他维度不变）
    2. 预分配输出内存 (torch.empty)
    3. 用切片赋值逐个拷贝数据
    """
    n_dims = tensors[0].dim()
    dim = dim if dim >= 0 else n_dims + dim
    target_shape = list(tensors[0].shape)
    target_shape[dim] = sum(t.shape[dim] for t in tensors)
    out = torch.empty(target_shape, dtype=tensors[0].dtype,
                      device=tensors[0].device)
    offset = 0
    for t in tensors:
        size = t.shape[dim]
        indices = [slice(None)] * n_dims
        indices[dim] = slice(offset, offset + size)
        out[tuple(indices)] = t
        offset += size
    return out


# ===========================================================================
#  7. stack — 新维度堆叠
# ===========================================================================

def custom_stack(tensors, dim=0):
    """stack = unsqueeze + cat。

    核心原理：
    - 先给每个 tensor 在 dim 位置插入一个 size=1 的新维度
    - 再沿该维度 cat

    示例: stack([a(3,), b(3,)], dim=0) → (2, 3)
        每个 (3,) → unsqueeze(0) → (1,3) → cat(dim=0) → (2,3)
    """
    return custom_cat([t.unsqueeze(dim) for t in tensors], dim)


# ===========================================================================
#  8. split — 拆分
# ===========================================================================

def custom_split(x, split_size_or_sections, dim=0):
    """split = 沿维度切片，返回视图列表。

    核心原理：
    - 每个返回的子 tensor 是原 tensor 的视图（共享内存）
    - 底层通过 slice 调整 offset + shape 实现
    - split_size 为 int → 均匀切分；为 list → 按指定大小切分
    """
    n_dims = x.dim()
    dim = dim if dim >= 0 else n_dims + dim
    dim_size = x.shape[dim]
    if isinstance(split_size_or_sections, int):
        sizes = []
        remaining = dim_size
        while remaining > 0:
            sizes.append(min(split_size_or_sections, remaining))
            remaining -= sizes[-1]
    else:
        sizes = list(split_size_or_sections)
    results = []
    offset = 0
    for size in sizes:
        idx = [slice(None)] * n_dims
        idx[dim] = slice(offset, offset + size)
        results.append(x[tuple(idx)])
        offset += size
    return results


# ===========================================================================
#  9. chunk — 均匀拆分
# ===========================================================================

def custom_chunk(x, chunks, dim=0):
    """chunk = 均匀分割为 N 块。

    核心原理：
    - chunk_size = ceil(dim_size / chunks)
    - 调用 split，最后一块可能不足 chunk_size
    """
    dim = dim if dim >= 0 else x.dim() + dim
    chunk_size = math.ceil(x.shape[dim] / chunks)
    return custom_split(x, chunk_size, dim)


# ===========================================================================
#  10. gather — 按索引取值
# ===========================================================================

def custom_gather(x, dim, index):
    """gather: 沿 dim 维度用 index 选取元素。

    核心原理：
        dim=0: out[i][j][k] = x[ index[i][j][k] ][ j ][ k ]
        dim=1: out[i][j][k] = x[ i ][ index[i][j][k] ][ k ]
    即：只在 dim 维度用 index 中的值替换坐标，其余维度保持原位。

    向量化技巧：
    - 为每个非 dim 维度构建坐标网格 (arange + reshape + expand)
    - 在 dim 维度用 index 本身
    - 组合成完整索引元组，一次 advanced indexing 取出所有值
    """
    dim = dim if dim >= 0 else x.dim() + dim
    idx_list = []
    for d in range(x.dim()):
        if d == dim:
            idx_list.append(index.long())
        else:
            shape = [1] * x.dim()
            shape[d] = index.shape[d]
            coord = torch.arange(index.shape[d], device=x.device).reshape(shape)
            idx_list.append(coord.expand_as(index))
    return x[tuple(idx_list)]


# ===========================================================================
#  11. scatter — 按索引写值 (gather 的逆操作)
# ===========================================================================

def custom_scatter(x, dim, index, src):
    """scatter: 将 src 的值按 index 写入 x。

    核心原理：
        dim=0: out[ index[i][j][k] ][ j ][ k ] = src[i][j][k]
        dim=1: out[ i ][ index[i][j][k] ][ k ] = src[i][j][k]
    相当于 gather 的"反方向"——gather 是按 index 读，scatter 是按 index 写。
    """
    dim = dim if dim >= 0 else x.dim() + dim
    out = x.clone()
    idx_list = []
    for d in range(x.dim()):
        if d == dim:
            idx_list.append(index.long())
        else:
            shape = [1] * x.dim()
            shape[d] = index.shape[d]
            coord = torch.arange(index.shape[d], device=x.device).reshape(shape)
            idx_list.append(coord.expand_as(index))
    out[tuple(idx_list)] = src
    return out


# ===========================================================================
#  12. matmul — 矩阵乘法 (2D)
# ===========================================================================

def custom_matmul(a, b):
    """matmul: C[i,j] = Σ_k A[i,k] * B[k,j]。

    核心原理（半向量化写法）：
    - 外层遍历输出的行 i
    - 内层遍历公共维度 k：把 A 的一个标量乘以 B 的一整行，累加到 out 的一行
    - 这等价于三重循环，但把最内层 j 维度交给 torch 向量化

    为什么是 i-k-j 而不是 i-j-k？
    - b[kk] 是按行连续访问，缓存友好
    - 如果用 i-j-k 顺序，b[kk, j] 是按列跳跃访问，缓存不友好
    """
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, f"内维不匹配: {k} vs {k2}"
    out = torch.zeros(m, n, dtype=a.dtype, device=a.device)
    for i in range(m):
        for kk in range(k):
            out[i] += a[i, kk] * b[kk]
    return out


# ===========================================================================
#  13. bmm — 批量矩阵乘法
# ===========================================================================

def custom_bmm(a, b):
    """bmm: 对 batch 维度独立做 2D matmul。

    核心原理：
    - out[b] = a[b] @ b[b]，每个 batch 独立互不影响
    - 半向量化：最内层 n 维度交给 torch 向量化
    """
    batch, m, k = a.shape
    _, k2, n = b.shape
    assert k == k2
    out = torch.zeros(batch, m, n, dtype=a.dtype, device=a.device)
    for bi in range(batch):
        for i in range(m):
            for kk in range(k):
                out[bi, i] += a[bi, i, kk] * b[bi, kk]
    return out


# ===========================================================================
#  14. linear — 全连接层
# ===========================================================================

def custom_linear(x, weight, bias=None):
    """linear: y = x @ Wᵀ + b。

    核心原理：
    - weight: (out_features, in_features)，每一行对应一个输出神经元
    - 本质: 输入向量与每行权重做点积，再加偏置

    等价理解：
        y[..., j] = Σ_i x[..., i] * W[j, i] + b[j]
    """
    in_f = x.shape[-1]
    out_f = weight.shape[0]
    x_2d = x.reshape(-1, in_f)
    n = x_2d.shape[0]
    w_t = weight.t()                                       # (in, out)
    out_2d = torch.zeros(n, out_f, dtype=x.dtype, device=x.device)
    for i in range(n):
        for kk in range(in_f):
            out_2d[i] += x_2d[i, kk] * w_t[kk]
    if bias is not None:
        out_2d = out_2d + bias
    return out_2d.reshape(*x.shape[:-1], out_f)


# ===========================================================================
#  15. conv2d — 2D 卷积 (img2col → matmul)
# ===========================================================================

def custom_conv2d(x, weight, bias=None, stride=1, padding=0):
    """conv2d 三步法: pad → img2col → matmul。

    核心原理:
    1. Padding:  在 H/W 边界填零
    2. img2col:  把每个卷积窗口"拉直"为一列
       (N, C_in, H, W) → (N, C_in*kH*kW, out_H*out_W)
    3. matmul:   W_reshaped @ col_matrix
       (C_out, C_in*kH*kW) @ (N, C_in*kH*kW, oH*oW) → (N, C_out, oH*oW)
    4. reshape + bias → (N, C_out, out_H, out_W)

    为什么用 img2col？
    - 将卷积转化为大矩阵乘法，可以直接调用高度优化的 BLAS/GEMM
    - 代价是展开后有冗余内存（重叠区域被重复存储）
    """
    n, c_in, h, w = x.shape
    c_out, c_in_w, kh, kw = weight.shape
    assert c_in == c_in_w

    # Step 1: Zero Padding
    if padding > 0:
        x_pad = torch.zeros(n, c_in, h + 2 * padding, w + 2 * padding,
                            dtype=x.dtype, device=x.device)
        x_pad[:, :, padding:padding + h, padding:padding + w] = x
    else:
        x_pad = x

    _, _, h_p, w_p = x_pad.shape
    out_h = (h_p - kh) // stride + 1
    out_w = (w_p - kw) // stride + 1

    # Step 2: img2col — 将每个 kH×kW 窗口展开为列向量
    col = torch.empty(n, c_in * kh * kw, out_h * out_w,
                      dtype=x.dtype, device=x.device)
    for ki in range(kh):
        for kj in range(kw):
            h_end = ki + out_h * stride
            w_end = kj + out_w * stride
            patches = x_pad[:, :, ki:h_end:stride, kj:w_end:stride]  # (N,C,oH,oW)
            for ci in range(c_in):
                row = ci * kh * kw + ki * kw + kj
                col[:, row, :] = patches[:, ci].reshape(n, -1)

    # Step 3: matmul (用 torch.bmm)
    w_2d = weight.reshape(c_out, c_in * kh * kw)
    out = torch.bmm(
        w_2d.unsqueeze(0).expand(n, -1, -1),
        col,
    )

    # Step 4: reshape + bias
    out = out.reshape(n, c_out, out_h, out_w)
    if bias is not None:
        out = out + bias.reshape(1, c_out, 1, 1)
    return out


# ===========================================================================
#  16. relu — ReLU 激活
# ===========================================================================

def custom_relu(x):
    """ReLU: f(x) = max(0, x)。

    核心原理：
    - 最简单的非线性激活：负值归零，正值不变
    - 梯度: x > 0 → 1, x ≤ 0 → 0（所以叫 Rectified Linear Unit）
    """
    out = x.clone()
    out[out < 0] = 0
    return out


# ===========================================================================
#  17. softmax — 概率归一化
# ===========================================================================

def custom_softmax(x, dim=-1):
    """softmax: p_i = exp(x_i) / Σ_j exp(x_j)。

    核心原理：
    1. 减去最大值（数值稳定性：避免 exp 上溢）
    2. 对每个元素取 exp
    3. 除以该维度的 exp 之和

    为什么要减最大值？
    - exp(1000) = inf，但 exp(1000 - 1000) = 1.0
    - 数学上 softmax(x) == softmax(x - c)，结果完全一样
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


# ===========================================================================
#  18. layer_norm — 层归一化
# ===========================================================================

def custom_layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    """layer_norm: y = (x - μ) / √(σ² + ε) * γ + β。

    核心原理：
    1. 对 normalized_shape 对应的最后几个维度计算均值和方差
    2. 标准化: (x - mean) / sqrt(var + eps)
    3. 仿射变换: * weight + bias (可学习的缩放和平移)

    与 batch_norm 的区别：
    - layer_norm: 对每个样本的特征维度归一化 (不依赖 batch 统计)
    - batch_norm: 对 batch 维度归一化 (训练/推理行为不同)
    """
    dims = list(range(x.dim() - len(normalized_shape), x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    return x_norm


# ===========================================================================
#  19. embedding — 嵌入查表
# ===========================================================================

def custom_embedding(weight, indices):
    """embedding: 用整数索引查表获取向量。

    核心原理：
    - weight: (vocab_size, embed_dim) 嵌入矩阵
    - indices: 任意形状的整数 tensor
    - output: (*indices.shape, embed_dim)
    - 本质就是 weight[indices] —— 用整数做行索引

    为什么 embedding 不是简单的查表？
    - 因为 weight 参与反向传播：梯度只流向被选中的行
    - 可以看作"稀疏全连接层": one-hot(indices) @ weight
    """
    return weight[indices.long()]


# ===========================================================================
#  20. cross_entropy — 交叉熵损失
# ===========================================================================

def custom_cross_entropy(logits, targets):
    """cross_entropy = log_softmax + NLL_loss。

    核心原理：
    1. log_softmax(x_i) = x_i - log(Σ_j exp(x_j))
       （数值稳定版: x_i - max - log(Σ_j exp(x_j - max))）
    2. NLL_loss = -log_softmax[target_class]
       （取出目标类别对应的 log 概率，取负号）
    3. 对 batch 取平均

    为什么不分开算 softmax 再取 log？
    - log(softmax) = log(exp(x)/sum) = x - log(sum)，避免了先 exp 再 log 的数值精度损失
    """
    x_max = logits.max(dim=1, keepdim=True).values
    log_sum_exp = (
        torch.log(torch.exp(logits - x_max).sum(dim=1, keepdim=True)) + x_max
    )
    log_softmax = logits - log_sum_exp
    batch_size = logits.shape[0]
    nll = -log_softmax[torch.arange(batch_size, device=logits.device), targets]
    return nll.mean()


# ===========================================================================
#  21-22. getitem / setitem — 基于索引的读取与赋值
# ===========================================================================
#
# PyTorch 索引系统的两条核心路径:
#
#   路径1 — 基础索引 (int / slice / None / Ellipsis)
#       只修改 (storage_offset, shape, strides) 三元组，底层数据 **零拷贝**。
#       这是理解 PyTorch 内存模型的关键。
#
#   路径2 — 高级索引 (tensor / bool mask)
#       **必须拷贝数据** 到新存储。原理是计算每个目标元素在 storage 中的偏移，
#       然后逐个 gather (读) 或 scatter (写)。
#
# 索引系统示意图:
#
#     x[key]
#       ├─ 全部是 int/slice/None/... → as_strided()  → 零拷贝视图
#       └─ 含有 tensor/bool          → 计算 flat offset → 新存储
#

def _expand_ellipsis(key, ndim):
    """将 ``...`` 展开为对应数量的 ``slice(None)``。

    示例: x 是 4D, key = (0, ..., 1) → (0, slice(None), slice(None), 1)
    """
    if not isinstance(key, tuple):
        key = (key,)
    n_ellipsis = sum(1 for k in key if k is Ellipsis)
    if n_ellipsis == 0:
        return key
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis (...)")
    n_real = sum(1 for k in key if k is not Ellipsis and k is not None)
    n_fill = ndim - n_real
    result = []
    for k in key:
        if k is Ellipsis:
            result.extend([slice(None)] * n_fill)
        else:
            result.append(k)
    return tuple(result)


def _normalize_key(key, ndim):
    """标准化索引: 确保是 tuple, 展开 Ellipsis。"""
    if not isinstance(key, tuple):
        key = (key,)
    return _expand_ellipsis(key, ndim)


def _has_advanced_index(key):
    """判断索引中是否包含 tensor 或 bool mask。"""
    return any(isinstance(k, torch.Tensor) for k in key)


# ---- 路径1: 基础索引 (零拷贝) ----

def _basic_getitem(x, key):
    """基础索引核心: 遍历 key, 逐维调整 offset / shape / stride。

    处理规则:
        int i   → offset += i * stride[dim],  该维度从输出中消失
        slice s → offset += start * stride[dim],
                  new_size  = len(range(start, stop, step)),
                  new_stride = stride[dim] * step
        None    → 插入 size=1 维度 (不消耗 x 的维度)

    最终用 as_strided 构造零拷贝视图。

    注: as_strided 不支持负 stride, 对 step<0 的 slice 需特殊处理:
        先取正步长视图 (从最后一个元素开始), 再用 flip 翻转该维度。
        flip 内部使用负 stride 实现, 仍为零拷贝视图。
    """
    offset = x.storage_offset()
    new_shape = []
    new_strides = []
    old_shape = list(x.shape)
    old_strides = list(x.stride())
    dim = 0
    flip_dims = []
    out_dim = 0
    for k in key:
        if k is None:
            new_shape.append(1)
            s = old_strides[dim] * old_shape[dim] if dim < len(old_shape) else 1
            new_strides.append(s)
            out_dim += 1
        elif isinstance(k, int):
            if k < 0:
                k += old_shape[dim]
            offset += k * old_strides[dim]
            dim += 1
        elif isinstance(k, slice):
            start, stop, step = k.indices(old_shape[dim])
            n_elem = len(range(start, stop, step))
            if step < 0:
                last_elem = start + (n_elem - 1) * step
                offset += last_elem * old_strides[dim]
                new_shape.append(n_elem)
                new_strides.append(old_strides[dim] * (-step))
                flip_dims.append(out_dim)
            else:
                offset += start * old_strides[dim]
                new_shape.append(n_elem)
                new_strides.append(old_strides[dim] * step)
            out_dim += 1
            dim += 1
    for d in range(dim, len(old_shape)):
        new_shape.append(old_shape[d])
        new_strides.append(old_strides[d])
    result = torch.as_strided(x, new_shape, new_strides, offset)
    for d in flip_dims:
        result = result.flip(d)
    return result


# ---- 路径2: 高级索引 (数据拷贝) ----

def _storage_1d(x):
    """获取 x 底层 storage 的一维视图 (通过 as_strided, 避免 .storage() 弃用)。

    原理: tensor 的所有元素都存储在一段连续内存中,
    用 as_strided 从 offset=0 开始、stride=1 创建一维视图即可覆盖全部有效区域。
    """
    max_off = x.storage_offset()
    for i in range(x.dim()):
        max_off += (x.shape[i] - 1) * x.stride()[i]
    return torch.as_strided(x, (max_off + 1,), (1,), 0)


def _advanced_offsets(x, key):
    """为高级索引计算每个目标元素在 x.storage() 中的偏移。

    返回 (flat_offsets, out_shape, squeeze_dims, unsqueeze_dims):
        flat_offsets  : LongTensor, 每个输出元素对应的 storage offset
        out_shape     : 广播后的原始输出形状
        squeeze_dims  : int 索引对应的维度 (需要在最终输出中去掉)
        unsqueeze_dims: None 对应的维度 (需要在最终输出中加上)
    """
    idx_per_dim = []
    squeeze_dims = []
    unsqueeze_dims = []
    out_pos = 0
    src_dim = 0
    for k in key:
        if k is None:
            unsqueeze_dims.append(out_pos)
            out_pos += 1
        elif isinstance(k, int):
            val = k if k >= 0 else k + x.shape[src_dim]
            idx_per_dim.append(
                torch.tensor([val], device=x.device, dtype=torch.long)
            )
            squeeze_dims.append(out_pos)
            out_pos += 1
            src_dim += 1
        elif isinstance(k, slice):
            start, stop, step = k.indices(x.shape[src_dim])
            idx_per_dim.append(
                torch.arange(start, stop, step, device=x.device)
            )
            out_pos += 1
            src_dim += 1
        elif isinstance(k, torch.Tensor):
            if k.dtype == torch.bool:
                idx_per_dim.append(k.nonzero(as_tuple=False).squeeze(-1))
            else:
                idx_per_dim.append(k.long().flatten())
            out_pos += 1
            src_dim += 1
    for d in range(src_dim, x.dim()):
        idx_per_dim.append(torch.arange(x.shape[d], device=x.device))

    ndim = len(idx_per_dim)
    grids = []
    for i, idx in enumerate(idx_per_dim):
        shape = [1] * ndim
        shape[i] = idx.shape[0]
        grids.append(idx.reshape(shape))
    bcast = [max(g.shape[d] for g in grids) for d in range(ndim)]
    expanded = [g.expand(bcast) for g in grids]

    strides_t = torch.tensor(x.stride(), device=x.device, dtype=torch.long)
    flat_offsets = torch.full(bcast, x.storage_offset(),
                              device=x.device, dtype=torch.long)
    for i, eg in enumerate(expanded):
        flat_offsets = flat_offsets + eg * strides_t[i]

    return flat_offsets, bcast, squeeze_dims, unsqueeze_dims


def _advanced_getitem(x, key):
    """高级索引读取: 计算 storage offset → 从 storage 一维视图逐个取值。"""
    s1d = _storage_1d(x)

    if (len(key) == 1 and isinstance(key[0], torch.Tensor)
            and key[0].dtype == torch.bool):
        mask = key[0]
        coords = mask.nonzero(as_tuple=False)
        n = coords.shape[0]
        out = torch.empty(n, dtype=x.dtype, device=x.device)
        strides = x.stride()
        base = x.storage_offset()
        for i in range(n):
            fi = base
            for d in range(x.dim()):
                fi += coords[i, d].item() * strides[d]
            out[i] = s1d[fi]
        return out

    flat_offsets, bcast, sq, unsq = _advanced_offsets(x, key)
    out = torch.empty(bcast, dtype=x.dtype, device=x.device)
    fo_flat = flat_offsets.reshape(-1)
    out_flat = out.reshape(-1)
    for i in range(fo_flat.numel()):
        out_flat[i] = s1d[fo_flat[i].item()]

    for d in sorted(sq, reverse=True):
        out = out.squeeze(d)
    for d in unsq:
        out = out.unsqueeze(d)
    return out


def _advanced_setitem(x, key, value):
    """高级索引赋值: 计算 storage offset → 通过 storage 一维视图逐个写入。"""
    s1d = _storage_1d(x)

    if (len(key) == 1 and isinstance(key[0], torch.Tensor)
            and key[0].dtype == torch.bool):
        mask = key[0]
        coords = mask.nonzero(as_tuple=False)
        n = coords.shape[0]
        strides = x.stride()
        base = x.storage_offset()
        is_scalar = not isinstance(value, torch.Tensor) or value.dim() == 0
        v_flat = None if is_scalar else value.flatten()
        for i in range(n):
            fi = base
            for d in range(x.dim()):
                fi += coords[i, d].item() * strides[d]
            s1d[fi] = float(value) if is_scalar else v_flat[i].item()
        return

    flat_offsets, bcast, _, _ = _advanced_offsets(x, key)
    fo_flat = flat_offsets.reshape(-1)
    is_scalar = not isinstance(value, torch.Tensor) or value.dim() == 0
    v_flat = None if is_scalar else value.flatten()
    for i in range(fo_flat.numel()):
        s1d[fo_flat[i].item()] = (
            float(value) if is_scalar else v_flat[i].item()
        )


# ---- 对外接口 ----

def custom_getitem(x, key):
    """x[key] 手动实现 — 展示 PyTorch 索引的底层原理。

    路径1 — 基础索引 (int / slice / None / Ellipsis):
        只修改 (offset, shape, strides) 元数据，零拷贝！
        ┌─────────────────────────────────────────────┐
        │  storage: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   │
        │  x = as_strided(shape=(2,5), stride=(5,1))  │
        │  x[1, 1:4] → offset=6, shape=(3,), stride=(1,) │
        │  → 视图指向 storage[6], [7], [8]            │
        └─────────────────────────────────────────────┘

    路径2 — 高级索引 (tensor / bool mask):
        必须拷贝数据！计算每个目标元素的 storage offset, 逐个取值。
        ┌─────────────────────────────────────────────┐
        │  x[torch.tensor([0,2,4])] →                 │
        │  offset_0 = 0*stride, offset_2 = 2*stride   │
        │  → 从 storage 中 gather 到新 tensor          │
        └─────────────────────────────────────────────┘
    """
    key = _normalize_key(key, x.dim())
    if not _has_advanced_index(key):
        return _basic_getitem(x, key)
    return _advanced_getitem(x, key)


def custom_setitem(x, key, value):
    """x[key] = value 手动实现。

    路径1 — 基础索引:
        构建 as_strided 视图 → fill_() 或 copy_() → 直接写入原始 storage。
        视图与原 tensor 共享内存, 修改视图 = 修改原 tensor。

    路径2 — 高级索引:
        计算每个目标位置的 storage offset → 逐个写入 storage。
    """
    key = _normalize_key(key, x.dim())
    if not _has_advanced_index(key):
        view = _basic_getitem(x, key)
        if isinstance(value, torch.Tensor):
            view.copy_(
                value.expand(view.shape) if value.shape != view.shape else value
            )
        else:
            view.fill_(float(value))
        return
    _advanced_setitem(x, key, value)
