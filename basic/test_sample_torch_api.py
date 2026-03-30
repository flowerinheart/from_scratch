"""Unit tests for sample_torch_api.py — correctness vs PyTorch native + performance."""

import time

import torch
import torch.nn.functional as F
import pytest

from sample_torch_api import (
    custom_bmm,
    custom_cat,
    custom_chunk,
    custom_conv2d,
    custom_cross_entropy,
    custom_embedding,
    custom_gather,
    custom_getitem,
    custom_layer_norm,
    custom_linear,
    custom_matmul,
    custom_permute,
    custom_relu,
    custom_reshape,
    custom_scatter,
    custom_setitem,
    custom_softmax,
    custom_split,
    custom_squeeze,
    custom_stack,
    custom_transpose,
    custom_unsqueeze,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _close(a, b, atol=1e-5):
    """Assert two tensors are element-wise close."""
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    assert torch.allclose(a.float(), b.float(), atol=atol), (
        f"max diff = {(a.float() - b.float()).abs().max().item()}"
    )


def _timer(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


# ===========================================================================
#  1. reshape
# ===========================================================================

class TestReshape:
    def test_basic(self):
        x = torch.arange(12).float()
        _close(custom_reshape(x, (3, 4)), x.reshape(3, 4))

    def test_minus_one(self):
        x = torch.arange(24).float()
        _close(custom_reshape(x, (4, -1)), x.reshape(4, -1))
        _close(custom_reshape(x, (-1, 3, 2)), x.reshape(-1, 3, 2))

    def test_non_contiguous(self):
        x = torch.arange(12).float().reshape(3, 4).t()  # non-contiguous
        _close(custom_reshape(x, (12,)), x.reshape(12))

    def test_shares_memory_if_contiguous(self):
        x = torch.arange(6).float()
        y = custom_reshape(x, (2, 3))
        y[0, 0] = 99
        assert x[0] == 99


# ===========================================================================
#  2. transpose
# ===========================================================================

class TestTranspose:
    def test_2d(self):
        x = torch.arange(6).float().reshape(2, 3)
        _close(custom_transpose(x, 0, 1), x.transpose(0, 1))

    def test_3d(self):
        x = torch.randn(2, 3, 4)
        _close(custom_transpose(x, 0, 2), x.transpose(0, 2))

    def test_negative_dim(self):
        x = torch.randn(2, 3, 4)
        _close(custom_transpose(x, -2, -1), x.transpose(-2, -1))

    def test_shares_memory(self):
        x = torch.arange(6).float().reshape(2, 3)
        y = custom_transpose(x, 0, 1)
        y[0, 0] = 99
        assert x[0, 0] == 99


# ===========================================================================
#  3. permute
# ===========================================================================

class TestPermute:
    def test_nchw_to_nhwc(self):
        x = torch.randn(2, 3, 4, 5)
        _close(custom_permute(x, [0, 2, 3, 1]), x.permute(0, 2, 3, 1))

    def test_3d(self):
        x = torch.randn(2, 3, 4)
        _close(custom_permute(x, [2, 0, 1]), x.permute(2, 0, 1))

    def test_identity(self):
        x = torch.randn(3, 4, 5)
        _close(custom_permute(x, [0, 1, 2]), x)


# ===========================================================================
#  4. squeeze
# ===========================================================================

class TestSqueeze:
    def test_all(self):
        x = torch.randn(1, 3, 1, 5)
        _close(custom_squeeze(x), x.squeeze())

    def test_specific_dim(self):
        x = torch.randn(1, 3, 1, 5)
        _close(custom_squeeze(x, 0), x.squeeze(0))
        _close(custom_squeeze(x, 2), x.squeeze(2))

    def test_no_effect(self):
        x = torch.randn(2, 3)
        _close(custom_squeeze(x, 0), x.squeeze(0))

    def test_scalar_result(self):
        x = torch.tensor([[[5.0]]])
        result = custom_squeeze(x)
        assert result.dim() == 0
        assert result.item() == 5.0


# ===========================================================================
#  5. unsqueeze
# ===========================================================================

class TestUnsqueeze:
    def test_basic(self):
        x = torch.randn(3, 4)
        _close(custom_unsqueeze(x, 0), x.unsqueeze(0))
        _close(custom_unsqueeze(x, 1), x.unsqueeze(1))
        _close(custom_unsqueeze(x, 2), x.unsqueeze(2))

    def test_negative(self):
        x = torch.randn(3, 4)
        _close(custom_unsqueeze(x, -1), x.unsqueeze(-1))

    def test_preserves_contiguity(self):
        x = torch.randn(3, 4)
        assert custom_unsqueeze(x, 0).is_contiguous()
        assert custom_unsqueeze(x, 1).is_contiguous()
        assert custom_unsqueeze(x, 2).is_contiguous()


# ===========================================================================
#  6. cat
# ===========================================================================

class TestCat:
    def test_dim0(self):
        a = torch.randn(2, 3)
        b = torch.randn(4, 3)
        _close(custom_cat([a, b], 0), torch.cat([a, b], 0))

    def test_dim1(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 5)
        _close(custom_cat([a, b], 1), torch.cat([a, b], 1))

    def test_three_tensors(self):
        ts = [torch.randn(2, 3) for _ in range(3)]
        _close(custom_cat(ts, 0), torch.cat(ts, 0))

    def test_negative_dim(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 4)
        _close(custom_cat([a, b], -1), torch.cat([a, b], -1))


# ===========================================================================
#  7. stack
# ===========================================================================

class TestStack:
    def test_dim0(self):
        ts = [torch.randn(3, 4) for _ in range(5)]
        _close(custom_stack(ts, 0), torch.stack(ts, 0))

    def test_dim1(self):
        ts = [torch.randn(3, 4) for _ in range(5)]
        _close(custom_stack(ts, 1), torch.stack(ts, 1))

    def test_1d(self):
        ts = [torch.tensor([1.0, 2, 3]), torch.tensor([4.0, 5, 6])]
        _close(custom_stack(ts, 0), torch.stack(ts, 0))


# ===========================================================================
#  8. split
# ===========================================================================

class TestSplit:
    def test_equal(self):
        x = torch.arange(12).float().reshape(4, 3)
        custom_parts = custom_split(x, 2, dim=0)
        native_parts = torch.split(x, 2, dim=0)
        for c, n in zip(custom_parts, native_parts):
            _close(c, n)

    def test_unequal(self):
        x = torch.arange(10).float()
        custom_parts = custom_split(x, 3)
        native_parts = torch.split(x, 3)
        assert len(custom_parts) == len(native_parts)
        for c, n in zip(custom_parts, native_parts):
            _close(c, n)

    def test_sections_list(self):
        x = torch.arange(10).float()
        custom_parts = custom_split(x, [2, 3, 5])
        native_parts = torch.split(x, [2, 3, 5])
        for c, n in zip(custom_parts, native_parts):
            _close(c, n)

    def test_returns_views(self):
        x = torch.arange(6).float().reshape(2, 3)
        parts = custom_split(x, 1, dim=0)
        parts[0][0, 0] = 99
        assert x[0, 0] == 99


# ===========================================================================
#  9. chunk
# ===========================================================================

class TestChunk:
    def test_even(self):
        x = torch.arange(12).float()
        custom_parts = custom_chunk(x, 3)
        native_parts = torch.chunk(x, 3)
        for c, n in zip(custom_parts, native_parts):
            _close(c, n)

    def test_uneven(self):
        x = torch.arange(10).float()
        custom_parts = custom_chunk(x, 3)
        native_parts = torch.chunk(x, 3)
        assert len(custom_parts) == len(native_parts)
        for c, n in zip(custom_parts, native_parts):
            _close(c, n)

    def test_dim1(self):
        x = torch.randn(3, 8)
        custom_parts = custom_chunk(x, 3, dim=1)
        native_parts = torch.chunk(x, 3, dim=1)
        for c, n in zip(custom_parts, native_parts):
            _close(c, n)


# ===========================================================================
#  10. gather
# ===========================================================================

class TestGather:
    def test_2d_dim0(self):
        x = torch.tensor([[1, 2], [3, 4], [5, 6]]).float()
        idx = torch.tensor([[0, 1], [2, 0]])
        _close(custom_gather(x, 0, idx), torch.gather(x, 0, idx))

    def test_2d_dim1(self):
        x = torch.tensor([[10, 20, 30], [40, 50, 60]]).float()
        idx = torch.tensor([[2, 0], [1, 2]])
        _close(custom_gather(x, 1, idx), torch.gather(x, 1, idx))

    def test_3d(self):
        torch.manual_seed(42)
        x = torch.randn(2, 3, 4)
        idx = torch.randint(0, 4, (2, 3, 2))
        _close(custom_gather(x, 2, idx), torch.gather(x, 2, idx))


# ===========================================================================
#  11. scatter
# ===========================================================================

class TestScatter:
    def test_2d_dim0(self):
        x = torch.zeros(3, 3)
        idx = torch.tensor([[0, 1, 2], [2, 0, 1]])
        src = torch.tensor([[1.0, 2, 3], [4, 5, 6]])
        _close(custom_scatter(x, 0, idx, src), x.scatter(0, idx, src))

    def test_2d_dim1(self):
        x = torch.zeros(2, 4)
        idx = torch.tensor([[3, 0], [1, 2]])
        src = torch.tensor([[10.0, 20], [30, 40]])
        _close(custom_scatter(x, 1, idx, src), x.scatter(1, idx, src))

    def test_3d(self):
        torch.manual_seed(0)
        x = torch.zeros(2, 3, 4)
        idx = torch.randint(0, 4, (2, 3, 2))
        src = torch.randn(2, 3, 2)
        _close(custom_scatter(x, 2, idx, src), x.scatter(2, idx, src))


# ===========================================================================
#  12. matmul
# ===========================================================================

class TestMatmul:
    def test_basic(self):
        a = torch.tensor([[1, 2], [3, 4], [5, 6]]).float()
        b = torch.tensor([[7, 8, 9], [10, 11, 12]]).float()
        _close(custom_matmul(a, b), a @ b)

    def test_identity(self):
        x = torch.randn(4, 4)
        eye = torch.eye(4)
        _close(custom_matmul(x, eye), x)

    def test_random(self):
        torch.manual_seed(1)
        a = torch.randn(8, 16)
        b = torch.randn(16, 12)
        _close(custom_matmul(a, b), a @ b, atol=1e-4)


# ===========================================================================
#  13. bmm
# ===========================================================================

class TestBmm:
    def test_basic(self):
        torch.manual_seed(2)
        a = torch.randn(4, 3, 5)
        b = torch.randn(4, 5, 7)
        _close(custom_bmm(a, b), torch.bmm(a, b), atol=1e-4)

    def test_single_batch(self):
        a = torch.randn(1, 4, 4)
        b = torch.randn(1, 4, 4)
        _close(custom_bmm(a, b), torch.bmm(a, b), atol=1e-4)


# ===========================================================================
#  14. linear
# ===========================================================================

class TestLinear:
    def test_without_bias(self):
        torch.manual_seed(3)
        x = torch.randn(4, 8)
        w = torch.randn(5, 8)
        _close(custom_linear(x, w), F.linear(x, w), atol=1e-4)

    def test_with_bias(self):
        torch.manual_seed(4)
        x = torch.randn(4, 8)
        w = torch.randn(5, 8)
        b = torch.randn(5)
        _close(custom_linear(x, w, b), F.linear(x, w, b), atol=1e-4)

    def test_batched(self):
        torch.manual_seed(5)
        x = torch.randn(2, 3, 8)
        w = torch.randn(5, 8)
        b = torch.randn(5)
        _close(custom_linear(x, w, b), F.linear(x, w, b), atol=1e-4)


# ===========================================================================
#  15. conv2d
# ===========================================================================

class TestConv2d:
    def test_no_padding(self):
        torch.manual_seed(6)
        x = torch.randn(1, 1, 5, 5)
        w = torch.randn(1, 1, 3, 3)
        _close(custom_conv2d(x, w), F.conv2d(x, w), atol=1e-4)

    def test_with_padding(self):
        torch.manual_seed(7)
        x = torch.randn(1, 1, 5, 5)
        w = torch.randn(1, 1, 3, 3)
        _close(custom_conv2d(x, w, padding=1), F.conv2d(x, w, padding=1),
               atol=1e-4)

    def test_stride2(self):
        torch.manual_seed(8)
        x = torch.randn(1, 1, 8, 8)
        w = torch.randn(1, 1, 3, 3)
        _close(custom_conv2d(x, w, stride=2), F.conv2d(x, w, stride=2),
               atol=1e-4)

    def test_multi_channel_with_bias(self):
        torch.manual_seed(9)
        x = torch.randn(2, 3, 8, 8)
        w = torch.randn(4, 3, 3, 3)
        b = torch.randn(4)
        _close(
            custom_conv2d(x, w, bias=b, stride=1, padding=1),
            F.conv2d(x, w, bias=b, stride=1, padding=1),
            atol=1e-4,
        )


# ===========================================================================
#  16. relu
# ===========================================================================

class TestRelu:
    def test_basic(self):
        x = torch.tensor([-3.0, -1, 0, 1, 3])
        _close(custom_relu(x), F.relu(x))

    def test_2d(self):
        x = torch.randn(4, 5)
        _close(custom_relu(x), F.relu(x))

    def test_all_negative(self):
        x = torch.tensor([-5.0, -3, -1])
        result = custom_relu(x)
        assert (result == 0).all()


# ===========================================================================
#  17. softmax
# ===========================================================================

class TestSoftmax:
    def test_1d(self):
        x = torch.tensor([1.0, 2, 3])
        _close(custom_softmax(x, dim=0), F.softmax(x, dim=0))

    def test_2d(self):
        x = torch.randn(3, 5)
        _close(custom_softmax(x, dim=-1), F.softmax(x, dim=-1))

    def test_sums_to_one(self):
        x = torch.randn(4, 6)
        result = custom_softmax(x, dim=1)
        _close(result.sum(dim=1), torch.ones(4))

    def test_large_values(self):
        x = torch.tensor([1000.0, 1001, 1002])
        result = custom_softmax(x, dim=0)
        expected = F.softmax(x, dim=0)
        _close(result, expected)


# ===========================================================================
#  18. layer_norm
# ===========================================================================

class TestLayerNorm:
    def test_basic(self):
        x = torch.randn(2, 3, 4)
        _close(
            custom_layer_norm(x, [4]),
            F.layer_norm(x, [4]),
            atol=1e-4,
        )

    def test_with_affine(self):
        torch.manual_seed(10)
        x = torch.randn(2, 3, 4)
        w = torch.randn(4)
        b = torch.randn(4)
        _close(
            custom_layer_norm(x, [4], weight=w, bias=b),
            F.layer_norm(x, [4], weight=w, bias=b),
            atol=1e-4,
        )

    def test_multi_dim_norm(self):
        torch.manual_seed(11)
        x = torch.randn(2, 3, 4, 5)
        _close(
            custom_layer_norm(x, [4, 5]),
            F.layer_norm(x, [4, 5]),
            atol=1e-4,
        )


# ===========================================================================
#  19. embedding
# ===========================================================================

class TestEmbedding:
    def test_basic(self):
        weight = torch.randn(10, 4)
        idx = torch.tensor([1, 3, 5, 7])
        _close(custom_embedding(weight, idx),
               F.embedding(idx, weight))

    def test_2d_indices(self):
        weight = torch.randn(20, 8)
        idx = torch.tensor([[0, 2, 4], [1, 3, 5]])
        _close(custom_embedding(weight, idx),
               F.embedding(idx, weight))


# ===========================================================================
#  20. cross_entropy
# ===========================================================================

class TestCrossEntropy:
    def test_basic(self):
        torch.manual_seed(12)
        logits = torch.randn(4, 10)
        targets = torch.tensor([3, 7, 0, 5])
        _close(
            custom_cross_entropy(logits, targets),
            F.cross_entropy(logits, targets),
            atol=1e-5,
        )

    def test_single_sample(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        targets = torch.tensor([2])
        _close(
            custom_cross_entropy(logits, targets),
            F.cross_entropy(logits, targets),
        )

    def test_large_logits(self):
        logits = torch.tensor([[1000.0, 1001.0, 999.0]])
        targets = torch.tensor([1])
        _close(
            custom_cross_entropy(logits, targets),
            F.cross_entropy(logits, targets),
        )


# ===========================================================================
#  Performance benchmarks
# ===========================================================================

class TestPerformance:
    """Timing: custom vs native PyTorch. Always passes; prints comparison."""

    def _report(self, name, t_custom, t_native):
        ratio = t_custom / max(t_native, 1e-9)
        print(
            f"  [PERF] {name:32s}  "
            f"custom={t_custom*1000:8.2f}ms  "
            f"native={t_native*1000:8.2f}ms  "
            f"ratio={ratio:6.1f}x"
        )

    def test_perf_reshape(self):
        x = torch.randn(1000, 1000)
        _, tc = _timer(custom_reshape, x, (500, 2000))
        _, tn = _timer(lambda: x.reshape(500, 2000))
        self._report("reshape(1M)", tc, tn)

    def test_perf_transpose(self):
        x = torch.randn(500, 500)
        _, tc = _timer(custom_transpose, x, 0, 1)
        _, tn = _timer(lambda: x.transpose(0, 1))
        self._report("transpose(500x500)", tc, tn)

    def test_perf_permute(self):
        x = torch.randn(8, 16, 32, 64)
        _, tc = _timer(custom_permute, x, [0, 2, 3, 1])
        _, tn = _timer(lambda: x.permute(0, 2, 3, 1))
        self._report("permute(NCHW→NHWC)", tc, tn)

    def test_perf_squeeze(self):
        x = torch.randn(1, 100, 1, 100)
        _, tc = _timer(custom_squeeze, x)
        _, tn = _timer(lambda: x.squeeze())
        self._report("squeeze", tc, tn)

    def test_perf_unsqueeze(self):
        x = torch.randn(100, 100)
        _, tc = _timer(custom_unsqueeze, x, 0)
        _, tn = _timer(lambda: x.unsqueeze(0))
        self._report("unsqueeze", tc, tn)

    def test_perf_cat(self):
        ts = [torch.randn(100, 100) for _ in range(10)]
        _, tc = _timer(custom_cat, ts, 0)
        _, tn = _timer(torch.cat, ts, 0)
        self._report("cat(10x 100x100)", tc, tn)

    def test_perf_stack(self):
        ts = [torch.randn(100, 100) for _ in range(10)]
        _, tc = _timer(custom_stack, ts, 0)
        _, tn = _timer(torch.stack, ts, 0)
        self._report("stack(10x 100x100)", tc, tn)

    def test_perf_split(self):
        x = torch.randn(1000, 100)
        _, tc = _timer(custom_split, x, 100, 0)
        _, tn = _timer(torch.split, x, 100, 0)
        self._report("split(1000x100, sz=100)", tc, tn)

    def test_perf_chunk(self):
        x = torch.randn(1000, 100)
        _, tc = _timer(custom_chunk, x, 10, 0)
        _, tn = _timer(torch.chunk, x, 10, 0)
        self._report("chunk(1000x100, n=10)", tc, tn)

    def test_perf_gather(self):
        x = torch.randn(100, 100)
        idx = torch.randint(0, 100, (100, 50))
        _, tc = _timer(custom_gather, x, 1, idx)
        _, tn = _timer(torch.gather, x, 1, idx)
        self._report("gather(100x100, idx=100x50)", tc, tn)

    def test_perf_scatter(self):
        x = torch.zeros(100, 100)
        idx = torch.randint(0, 100, (100, 50))
        src = torch.randn(100, 50)
        _, tc = _timer(custom_scatter, x, 1, idx, src)
        _, tn = _timer(lambda: x.scatter(1, idx, src))
        self._report("scatter(100x100)", tc, tn)

    def test_perf_matmul(self):
        torch.manual_seed(0)
        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        _, tc = _timer(custom_matmul, a, b)
        _, tn = _timer(lambda: a @ b)
        self._report("matmul(32x32)", tc, tn)

    def test_perf_bmm(self):
        torch.manual_seed(0)
        a = torch.randn(4, 16, 16)
        b = torch.randn(4, 16, 16)
        _, tc = _timer(custom_bmm, a, b)
        _, tn = _timer(torch.bmm, a, b)
        self._report("bmm(4x16x16)", tc, tn)

    def test_perf_linear(self):
        torch.manual_seed(0)
        x = torch.randn(8, 32)
        w = torch.randn(16, 32)
        b = torch.randn(16)
        _, tc = _timer(custom_linear, x, w, b)
        _, tn = _timer(F.linear, x, w, b)
        self._report("linear(8x32→16)", tc, tn)

    def test_perf_conv2d(self):
        torch.manual_seed(0)
        x = torch.randn(1, 3, 16, 16)
        w = torch.randn(8, 3, 3, 3)
        _, tc = _timer(custom_conv2d, x, w, padding=1)
        _, tn = _timer(F.conv2d, x, w, padding=1)
        self._report("conv2d(1,3,16,16 co=8 k=3)", tc, tn)

    def test_perf_relu(self):
        x = torch.randn(1000, 1000)
        _, tc = _timer(custom_relu, x)
        _, tn = _timer(F.relu, x)
        self._report("relu(1Mx1M)", tc, tn)

    def test_perf_softmax(self):
        x = torch.randn(256, 1000)
        _, tc = _timer(custom_softmax, x, -1)
        _, tn = _timer(F.softmax, x, -1)
        self._report("softmax(256x1000)", tc, tn)

    def test_perf_layer_norm(self):
        x = torch.randn(32, 128, 64)
        _, tc = _timer(custom_layer_norm, x, [64])
        _, tn = _timer(F.layer_norm, x, [64])
        self._report("layer_norm(32x128x64)", tc, tn)

    def test_perf_embedding(self):
        weight = torch.randn(10000, 256)
        idx = torch.randint(0, 10000, (32, 128))
        _, tc = _timer(custom_embedding, weight, idx)
        _, tn = _timer(F.embedding, idx, weight)
        self._report("embedding(32x128, V=10k)", tc, tn)

    def test_perf_cross_entropy(self):
        logits = torch.randn(256, 100)
        targets = torch.randint(0, 100, (256,))
        _, tc = _timer(custom_cross_entropy, logits, targets)
        _, tn = _timer(F.cross_entropy, logits, targets)
        self._report("cross_entropy(256x100)", tc, tn)


# ===========================================================================
#  21. getitem — 基于索引的读取
# ===========================================================================


class TestGetitem:
    """custom_getitem(x, key) vs x[key]."""

    # ---- 基础索引: int ----

    def test_single_int(self):
        x = torch.arange(12).float().reshape(3, 4)
        _close(custom_getitem(x, 1), x[1])

    def test_negative_int(self):
        x = torch.arange(12).float().reshape(3, 4)
        _close(custom_getitem(x, -1), x[-1])

    def test_multi_int(self):
        x = torch.arange(24).float().reshape(2, 3, 4)
        result = custom_getitem(x, (1, 2, 3))
        assert result.item() == x[1, 2, 3].item()

    # ---- 基础索引: slice ----

    def test_simple_slice(self):
        x = torch.arange(10).float()
        _close(custom_getitem(x, slice(2, 7)), x[2:7])

    def test_step_slice(self):
        x = torch.arange(20).float().reshape(4, 5)
        _close(custom_getitem(x, (slice(None), slice(0, 5, 2))), x[:, 0:5:2])

    def test_negative_step(self):
        x = torch.arange(10).float()
        expected = x.flip(0)
        _close(custom_getitem(x, slice(None, None, -1)), expected)

    def test_combined_int_slice(self):
        x = torch.arange(24).float().reshape(2, 3, 4)
        _close(custom_getitem(x, (0, slice(1, 3))), x[0, 1:3])

    def test_multi_slice(self):
        x = torch.arange(60).float().reshape(3, 4, 5)
        key = (slice(1, 3), slice(None), slice(0, 5, 2))
        _close(custom_getitem(x, key), x[1:3, :, 0:5:2])

    # ---- 基础索引: None (newaxis) ----

    def test_none_leading(self):
        x = torch.arange(6).float().reshape(2, 3)
        result = custom_getitem(x, (None,))
        _close(result, x[None])
        assert result.shape == (1, 2, 3)

    def test_none_middle(self):
        x = torch.arange(6).float().reshape(2, 3)
        result = custom_getitem(x, (slice(None), None, slice(None)))
        _close(result, x[:, None, :])
        assert result.shape == (2, 1, 3)

    def test_none_trailing(self):
        x = torch.arange(6).float().reshape(2, 3)
        result = custom_getitem(x, (slice(None), slice(None), None))
        _close(result, x[:, :, None])
        assert result.shape == (2, 3, 1)

    # ---- 基础索引: Ellipsis ----

    def test_ellipsis_leading(self):
        x = torch.arange(24).float().reshape(2, 3, 4)
        _close(custom_getitem(x, (..., 2)), x[..., 2])

    def test_ellipsis_middle(self):
        x = torch.arange(24).float().reshape(2, 3, 4)
        _close(custom_getitem(x, (0, ..., 1)), x[0, ..., 1])

    # ---- 基础索引: 零拷贝验证 ----

    def test_basic_shares_memory(self):
        x = torch.arange(12).float().reshape(3, 4)
        view = custom_getitem(x, (slice(1, 3),))
        view[0, 0] = 999
        assert x[1, 0] == 999

    def test_slice_is_view(self):
        x = torch.arange(20).float()
        view = custom_getitem(x, slice(5, 10))
        view[0] = -1
        assert x[5] == -1

    # ---- 高级索引: bool mask ----

    def test_bool_mask_1d(self):
        x = torch.tensor([10, 20, 30, 40, 50]).float()
        mask = torch.tensor([True, False, True, False, True])
        _close(custom_getitem(x, mask), x[mask])

    def test_bool_mask_2d(self):
        x = torch.arange(12).float().reshape(3, 4)
        mask = x > 5
        _close(custom_getitem(x, mask), x[mask])

    # ---- 高级索引: tensor index ----

    def test_tensor_index_1d(self):
        x = torch.tensor([10, 20, 30, 40, 50]).float()
        idx = torch.tensor([4, 0, 2])
        _close(custom_getitem(x, idx), x[idx])

    def test_tensor_index_rows(self):
        x = torch.arange(12).float().reshape(3, 4)
        idx = torch.tensor([2, 0])
        _close(custom_getitem(x, idx), x[idx])

    def test_tensor_index_cols(self):
        x = torch.arange(12).float().reshape(3, 4)
        key = (slice(None), torch.tensor([0, 3]))
        _close(custom_getitem(x, key), x[:, torch.tensor([0, 3])])

    def test_int_and_tensor_mixed(self):
        x = torch.arange(24).float().reshape(2, 3, 4)
        key = (1, torch.tensor([0, 2]))
        _close(custom_getitem(x, key), x[1, torch.tensor([0, 2])])


# ===========================================================================
#  22. setitem — 基于索引的赋值
# ===========================================================================


class TestSetitem:
    """custom_setitem(x, key, value) vs x[key] = value."""

    # ---- 基础索引: 标量赋值 ----

    def test_scalar_to_element(self):
        x = torch.arange(12).float().reshape(3, 4)
        y = x.clone()
        custom_setitem(x, (1, 2), 99.0)
        y[1, 2] = 99.0
        _close(x, y)

    def test_scalar_to_slice(self):
        x = torch.arange(12).float().reshape(3, 4)
        y = x.clone()
        custom_setitem(x, (slice(None), 0), 0.0)
        y[:, 0] = 0.0
        _close(x, y)

    def test_scalar_broadcast_row(self):
        x = torch.arange(12).float().reshape(3, 4)
        y = x.clone()
        custom_setitem(x, 1, 0.0)
        y[1] = 0.0
        _close(x, y)

    # ---- 基础索引: tensor 赋值 ----

    def test_tensor_to_slice(self):
        x = torch.zeros(3, 4)
        y = x.clone()
        val = torch.arange(4).float()
        custom_setitem(x, 1, val)
        y[1] = val
        _close(x, y)

    def test_tensor_to_multi_slice(self):
        x = torch.zeros(4, 6)
        y = x.clone()
        val = torch.ones(2, 3)
        custom_setitem(x, (slice(1, 3), slice(0, 3)), val)
        y[1:3, 0:3] = val
        _close(x, y)

    def test_step_slice_setitem(self):
        x = torch.zeros(10)
        y = x.clone()
        val = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        custom_setitem(x, slice(0, 10, 2), val)
        y[0:10:2] = val
        _close(x, y)

    # ---- 基础索引: 验证写穿原始 tensor ----

    def test_writes_through(self):
        x = torch.zeros(3, 4)
        custom_setitem(x, (0, slice(None)), 7.0)
        assert (x[0] == 7.0).all()
        assert (x[1] == 0.0).all()

    # ---- 高级索引: bool mask 赋值 ----

    def test_bool_mask_scalar(self):
        x = torch.tensor([1.0, -2, 3, -4, 5])
        y = x.clone()
        mask = x < 0
        custom_setitem(x, mask, 0.0)
        y[mask] = 0.0
        _close(x, y)

    def test_bool_mask_tensor(self):
        x = torch.arange(6).float().reshape(2, 3)
        y = x.clone()
        mask = x > 3
        vals = torch.tensor([99.0, 88.0])
        custom_setitem(x, mask, vals)
        y[mask] = vals
        _close(x, y)

    # ---- 高级索引: tensor index 赋值 ----

    def test_tensor_index_scalar(self):
        x = torch.zeros(5)
        y = x.clone()
        idx = torch.tensor([1, 3])
        custom_setitem(x, idx, 9.0)
        y[idx] = 9.0
        _close(x, y)

    def test_tensor_index_values(self):
        x = torch.zeros(5)
        y = x.clone()
        idx = torch.tensor([0, 2, 4])
        vals = torch.tensor([10.0, 20.0, 30.0])
        custom_setitem(x, idx, vals)
        y[idx] = vals
        _close(x, y)

    def test_tensor_index_cols(self):
        x = torch.zeros(3, 4)
        y = x.clone()
        key = (slice(None), torch.tensor([0, 3]))
        vals = torch.ones(3, 2)
        custom_setitem(x, key, vals)
        y[:, torch.tensor([0, 3])] = vals
        _close(x, y)


# ===========================================================================
#  getitem / setitem 性能测试
# ===========================================================================


class TestGetitemSetitemPerformance:
    def _report(self, name, t_custom, t_native):
        ratio = t_custom / max(t_native, 1e-9)
        print(
            f"  [PERF] {name:36s}  "
            f"custom={t_custom*1000:8.3f}ms  "
            f"native={t_native*1000:8.3f}ms  "
            f"ratio={ratio:6.1f}x"
        )

    def test_perf_basic_slice(self):
        x = torch.randn(1000, 1000)
        _, tc = _timer(custom_getitem, x, (slice(100, 900), slice(200, 800)))
        _, tn = _timer(lambda: x[100:900, 200:800])
        self._report("getitem basic slice(1000x1000)", tc, tn)

    def test_perf_basic_int_slice(self):
        x = torch.randn(100, 200, 300)
        key = (50, slice(10, 190))
        _, tc = _timer(custom_getitem, x, key)
        _, tn = _timer(lambda: x[50, 10:190])
        self._report("getitem int+slice(100x200x300)", tc, tn)

    def test_perf_bool_mask(self):
        x = torch.randn(100, 100)
        mask = x > 0
        _, tc = _timer(custom_getitem, x, mask)
        _, tn = _timer(lambda: x[mask])
        self._report("getitem bool mask(100x100)", tc, tn)

    def test_perf_tensor_index(self):
        x = torch.randn(1000, 64)
        idx = torch.randint(0, 1000, (200,))
        _, tc = _timer(custom_getitem, x, idx)
        _, tn = _timer(lambda: x[idx])
        self._report("getitem tensor idx(1000x64)", tc, tn)

    def test_perf_setitem_slice(self):
        x = torch.randn(1000, 1000)
        val = torch.ones(500, 600)
        key = (slice(100, 600), slice(200, 800))
        _, tc = _timer(custom_setitem, x, key, val)
        _, tn = _timer(lambda: x.__setitem__(key, val))
        self._report("setitem basic slice(1000x1000)", tc, tn)

    def test_perf_setitem_bool(self):
        x = torch.randn(100, 100)
        mask = x > 0
        _, tc = _timer(custom_setitem, x, mask, 0.0)
        x2 = x.clone()
        _, tn = _timer(lambda: x2.__setitem__(mask, 0.0))
        self._report("setitem bool mask(100x100)", tc, tn)
