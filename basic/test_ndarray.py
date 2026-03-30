"""Unit tests for NDArray — correctness (vs numpy) and performance."""

from __future__ import annotations

import time

import numpy as np
import pytest

from ndarray import (
    NDArray,
    arange,
    cat,
    chunk,
    conv2d,
    ones,
    split,
    stack,
    zeros,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _to_ndarray(np_arr: np.ndarray) -> NDArray:
    """Convert a numpy array to an NDArray."""
    return NDArray(np_arr.tolist())


def _flatten_nested(obj):
    """Recursively flatten a nested list to a flat list of floats."""
    if isinstance(obj, (list, tuple)):
        result = []
        for item in obj:
            result.extend(_flatten_nested(item))
        return result
    return [float(obj)]


def _assert_close(nd: NDArray, np_arr: np.ndarray, atol: float = 1e-6) -> None:
    """Assert NDArray matches numpy array within tolerance."""
    assert nd.shape == tuple(np_arr.shape), (
        f"shape mismatch: {nd.shape} vs {np_arr.shape}"
    )
    nd_flat = _flatten_nested(nd.tolist())
    np_flat = _flatten_nested(np_arr.tolist())
    assert nd_flat == pytest.approx(np_flat, abs=atol)


def _timer(func, *args, **kwargs):
    """Run *func* and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# ===========================================================================
#  1. Creation & Indexing
# ===========================================================================


class TestCreationAndIndexing:
    def test_from_scalar(self):
        a = NDArray(5.0)
        assert a.shape == ()
        assert a.ndim == 0
        assert a.size == 1
        assert a.tolist() == 5.0

    def test_from_1d(self):
        a = NDArray([1, 2, 3])
        assert a.shape == (3,)
        assert a.tolist() == [1.0, 2.0, 3.0]

    def test_from_2d(self):
        data = [[1, 2, 3], [4, 5, 6]]
        a = NDArray(data)
        np_a = np.array(data, dtype=float)
        assert a.shape == (2, 3)
        _assert_close(a, np_a)

    def test_from_3d(self):
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        a = NDArray(data)
        np_a = np.array(data, dtype=float)
        assert a.shape == (2, 2, 2)
        _assert_close(a, np_a)

    def test_integer_index(self):
        a = NDArray([[10, 20, 30], [40, 50, 60]])
        assert a[0, 1] == 20.0
        assert a[1, 2] == 60.0
        assert a[-1, -1] == 60.0

    def test_row_index(self):
        a = NDArray([[1, 2], [3, 4], [5, 6]])
        row = a[1]
        assert isinstance(row, NDArray)
        assert row.shape == (2,)
        assert row.tolist() == [3.0, 4.0]

    def test_slice_index(self):
        np_a = np.arange(24, dtype=float).reshape(4, 6)
        a = _to_ndarray(np_a)
        _assert_close(a[1:3], np_a[1:3])
        _assert_close(a[:, 2:5], np_a[:, 2:5])
        _assert_close(a[::2], np_a[::2])
        _assert_close(a[1:3, ::2], np_a[1:3, ::2])

    def test_negative_step(self):
        np_a = np.arange(10, dtype=float)
        a = _to_ndarray(np_a)
        _assert_close(a[::-1], np_a[::-1])

    def test_setitem_scalar(self):
        a = NDArray([[1, 2], [3, 4]])
        a[0, 1] = 99
        assert a[0, 1] == 99.0

    def test_setitem_broadcast_scalar(self):
        a = NDArray([[1, 2, 3], [4, 5, 6]])
        a[0] = 0
        assert a.tolist() == [[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]]

    def test_contiguous(self):
        a = NDArray([[1, 2, 3], [4, 5, 6]])
        assert a.is_contiguous()
        col = a[:, ::2]  # shape (2, 2), strides (3, 2) — not contiguous
        assert not col.is_contiguous()
        cc = col.contiguous()
        assert cc.is_contiguous()
        assert cc.tolist() == col.tolist()

    def test_strides_and_offset(self):
        a = NDArray([[1, 2, 3], [4, 5, 6]])
        assert a.strides == (3, 1)
        assert a.offset == 0
        sub = a[1]
        assert sub.offset == 3

    def test_factory_zeros(self):
        z = zeros(2, 3)
        assert z.shape == (2, 3)
        assert z.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    def test_factory_ones(self):
        o = ones(3)
        assert o.tolist() == [1.0, 1.0, 1.0]

    def test_factory_arange(self):
        a = arange(5)
        assert a.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


# ===========================================================================
#  2. View
# ===========================================================================


class TestView:
    def test_basic(self):
        a = arange(12)
        v = a.view(3, 4)
        np_a = np.arange(12, dtype=float).reshape(3, 4)
        _assert_close(v, np_a)

    def test_shares_data(self):
        a = arange(6)
        v = a.view(2, 3)
        v[0, 0] = 99
        assert a[0] == 99.0

    def test_infer_minus_one(self):
        a = arange(24)
        v = a.view(4, -1)
        assert v.shape == (4, 6)
        v2 = a.view(-1, 3, 2)
        assert v2.shape == (4, 3, 2)

    def test_non_contiguous_raises(self):
        a = arange(12).view(3, 4)
        nc = a[:, ::2]
        with pytest.raises(RuntimeError, match="contiguous"):
            nc.view(-1)

    def test_size_mismatch_raises(self):
        a = arange(12)
        with pytest.raises(ValueError):
            a.view(5, 5)


# ===========================================================================
#  3. Reshape
# ===========================================================================


class TestReshape:
    def test_contiguous_no_copy(self):
        a = arange(12)
        r = a.reshape(3, 4)
        r[0, 0] = 99
        assert a[0] == 99.0

    def test_non_contiguous_copies(self):
        a = arange(12).view(3, 4)
        nc = a[:, ::2]                 # (3,2), strides (4,2)
        r = nc.reshape(6)
        assert r.shape == (6,)
        np_ref = np.arange(12, dtype=float).reshape(3, 4)[:, ::2].reshape(6)
        _assert_close(r, np_ref)

    def test_infer_minus_one(self):
        a = arange(24)
        assert a.reshape(4, -1).shape == (4, 6)
        assert a.reshape(-1).shape == (24,)


# ===========================================================================
#  4. Squeeze / Unsqueeze
# ===========================================================================


class TestSqueezeUnsqueeze:
    def test_squeeze_all(self):
        a = arange(6).view(1, 6, 1)
        s = a.squeeze()
        assert s.shape == (6,)
        assert s.tolist() == list(range(6))

    def test_squeeze_specific_dim(self):
        a = arange(6).view(1, 2, 1, 3)
        assert a.squeeze(0).shape == (2, 1, 3)
        assert a.squeeze(2).shape == (1, 2, 3)

    def test_squeeze_no_effect(self):
        a = arange(6).view(2, 3)
        s = a.squeeze(0)
        assert s.shape == (2, 3)

    def test_unsqueeze(self):
        a = arange(6).view(2, 3)
        u = a.unsqueeze(0)
        assert u.shape == (1, 2, 3)
        u1 = a.unsqueeze(1)
        assert u1.shape == (2, 1, 3)
        u2 = a.unsqueeze(2)
        assert u2.shape == (2, 3, 1)
        u_neg = a.unsqueeze(-1)
        assert u_neg.shape == (2, 3, 1)

    def test_unsqueeze_contiguous(self):
        a = arange(6).view(2, 3)
        assert a.is_contiguous()
        assert a.unsqueeze(0).is_contiguous()
        assert a.unsqueeze(1).is_contiguous()
        assert a.unsqueeze(2).is_contiguous()

    def test_round_trip(self):
        np_a = np.arange(6, dtype=float).reshape(2, 3)
        a = _to_ndarray(np_a)
        rt = a.unsqueeze(1).squeeze(1)
        _assert_close(rt, np_a)


# ===========================================================================
#  5. Cat
# ===========================================================================


class TestCat:
    def test_dim0(self):
        a = _to_ndarray(np.array([[1, 2], [3, 4]], dtype=float))
        b = _to_ndarray(np.array([[5, 6]], dtype=float))
        result = cat([a, b], dim=0)
        expected = np.concatenate(
            [np.array([[1, 2], [3, 4]]), np.array([[5, 6]])], axis=0
        )
        _assert_close(result, expected)

    def test_dim1(self):
        a = _to_ndarray(np.ones((2, 3), dtype=float))
        b = _to_ndarray(np.zeros((2, 2), dtype=float))
        result = cat([a, b], dim=1)
        expected = np.concatenate(
            [np.ones((2, 3)), np.zeros((2, 2))], axis=1
        )
        _assert_close(result, expected)

    def test_three_tensors(self):
        arrs = [_to_ndarray(np.full((2, 2), i, dtype=float)) for i in range(3)]
        result = cat(arrs, dim=0)
        expected = np.concatenate(
            [np.full((2, 2), i) for i in range(3)], axis=0
        )
        _assert_close(result, expected)

    def test_negative_dim(self):
        a = _to_ndarray(np.ones((2, 3), dtype=float))
        b = _to_ndarray(np.zeros((2, 4), dtype=float))
        result = cat([a, b], dim=-1)
        expected = np.concatenate(
            [np.ones((2, 3)), np.zeros((2, 4))], axis=-1
        )
        _assert_close(result, expected)


# ===========================================================================
#  6. Stack
# ===========================================================================


class TestStack:
    def test_dim0(self):
        a = _to_ndarray(np.array([1, 2, 3], dtype=float))
        b = _to_ndarray(np.array([4, 5, 6], dtype=float))
        result = stack([a, b], dim=0)
        expected = np.stack([np.array([1, 2, 3.0]), np.array([4, 5, 6.0])], axis=0)
        _assert_close(result, expected)

    def test_dim1(self):
        a = _to_ndarray(np.array([1, 2, 3], dtype=float))
        b = _to_ndarray(np.array([4, 5, 6], dtype=float))
        result = stack([a, b], dim=1)
        expected = np.stack([np.array([1, 2, 3.0]), np.array([4, 5, 6.0])], axis=1)
        _assert_close(result, expected)

    def test_2d(self):
        np_a = np.arange(6, dtype=float).reshape(2, 3)
        np_b = np.arange(6, 12, dtype=float).reshape(2, 3)
        result = stack([_to_ndarray(np_a), _to_ndarray(np_b)], dim=0)
        expected = np.stack([np_a, np_b], axis=0)
        _assert_close(result, expected)


# ===========================================================================
#  7. Split
# ===========================================================================


class TestSplit:
    def test_equal_split(self):
        a = arange(12).view(4, 3)
        parts = split(a, 2, dim=0)
        assert len(parts) == 2
        assert parts[0].shape == (2, 3)
        assert parts[1].shape == (2, 3)
        np_ref = np.arange(12, dtype=float).reshape(4, 3)
        _assert_close(parts[0], np_ref[:2])
        _assert_close(parts[1], np_ref[2:])

    def test_unequal_split(self):
        a = arange(10)
        parts = split(a, 3, dim=0)
        assert len(parts) == 4
        assert [p.shape[0] for p in parts] == [3, 3, 3, 1]

    def test_split_by_sections(self):
        a = arange(10)
        parts = split(a, [2, 3, 5], dim=0)
        assert len(parts) == 3
        assert parts[0].tolist() == [0.0, 1.0]
        assert parts[1].tolist() == [2.0, 3.0, 4.0]
        assert parts[2].tolist() == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_split_returns_views(self):
        a = arange(6).view(2, 3)
        parts = split(a, 1, dim=0)
        parts[0][0, 0] = 99
        assert a[0, 0] == 99.0


# ===========================================================================
#  8. Chunk
# ===========================================================================


class TestChunk:
    def test_even(self):
        a = arange(12)
        parts = chunk(a, 3)
        assert len(parts) == 3
        for p in parts:
            assert p.shape == (4,)

    def test_uneven(self):
        a = arange(10)
        parts = chunk(a, 3)
        assert len(parts) == 3
        sizes = [p.shape[0] for p in parts]
        assert sizes == [4, 4, 2]

    def test_dim1(self):
        np_a = np.arange(12, dtype=float).reshape(3, 4)
        a = _to_ndarray(np_a)
        parts = chunk(a, 2, dim=1)
        assert len(parts) == 2
        _assert_close(parts[0], np_a[:, :2])
        _assert_close(parts[1], np_a[:, 2:])


# ===========================================================================
#  9. Gather
# ===========================================================================


class TestGather:
    def test_1d(self):
        src = NDArray([10, 20, 30, 40, 50])
        idx = NDArray([4, 0, 2])
        result = src.unsqueeze(0).gather(1, idx.unsqueeze(0))
        assert result.squeeze(0).tolist() == [50.0, 10.0, 30.0]

    def test_2d_dim0(self):
        np_src = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        np_idx = np.array([[0, 1], [2, 0]], dtype=float)
        src = _to_ndarray(np_src)
        idx = _to_ndarray(np_idx)
        result = src.gather(0, idx)
        expected = np.take_along_axis(np_src, np_idx.astype(int), axis=0)
        _assert_close(result, expected)

    def test_2d_dim1(self):
        np_src = np.array([[10, 20, 30], [40, 50, 60]], dtype=float)
        np_idx = np.array([[2, 0], [1, 2]], dtype=float)
        src = _to_ndarray(np_src)
        idx = _to_ndarray(np_idx)
        result = src.gather(1, idx)
        expected = np.take_along_axis(np_src, np_idx.astype(int), axis=1)
        _assert_close(result, expected)

    def test_3d(self):
        np.random.seed(42)
        np_src = np.random.rand(2, 3, 4)
        np_idx = np.random.randint(0, 4, size=(2, 3, 2)).astype(float)
        src = _to_ndarray(np_src)
        idx = _to_ndarray(np_idx)
        result = src.gather(2, idx)
        expected = np.take_along_axis(np_src, np_idx.astype(int), axis=2)
        _assert_close(result, expected)


# ===========================================================================
#  10. Matmul
# ===========================================================================


class TestMatmul:
    def test_2d(self):
        np_a = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        np_b = np.array([[7, 8, 9], [10, 11, 12]], dtype=float)
        a = _to_ndarray(np_a)
        b = _to_ndarray(np_b)
        result = a.matmul(b)
        _assert_close(result, np_a @ np_b)

    def test_at_operator(self):
        np_a = np.eye(3, dtype=float)
        np_b = np.arange(9, dtype=float).reshape(3, 3)
        a = _to_ndarray(np_a)
        b = _to_ndarray(np_b)
        _assert_close(a @ b, np_a @ np_b)

    def test_non_contiguous(self):
        np_a = np.arange(12, dtype=float).reshape(3, 4)
        np_b = np.arange(12, dtype=float).reshape(4, 3)
        a = _to_ndarray(np_a)[:, :2]      # (3,2) non-contiguous
        b = _to_ndarray(np_b)[:2, :]       # (2,3) contiguous
        np_a2 = np_a[:, :2]
        np_b2 = np_b[:2, :]
        result = a.contiguous().matmul(b)
        _assert_close(result, np_a2 @ np_b2)

    def test_batched(self):
        np.random.seed(7)
        np_a = np.random.rand(2, 3, 4)
        np_b = np.random.rand(2, 4, 5)
        a = _to_ndarray(np_a)
        b = _to_ndarray(np_b)
        result = a.matmul(b)
        _assert_close(result, np_a @ np_b, atol=1e-5)

    def test_batched_broadcast(self):
        np.random.seed(8)
        np_a = np.random.rand(3, 2, 4)
        np_b = np.random.rand(1, 4, 5)
        a = _to_ndarray(np_a)
        b = _to_ndarray(np_b)
        result = a.matmul(b)
        _assert_close(result, np_a @ np_b, atol=1e-5)


# ===========================================================================
#  11. Conv2D
# ===========================================================================


def _numpy_conv2d(inp, weight, bias=None, stride=1, padding=0):
    """Reference conv2d with pure numpy loops."""
    n, c_in, h, w = inp.shape
    c_out, _, kh, kw = weight.shape
    if padding > 0:
        inp = np.pad(
            inp, ((0, 0), (0, 0), (padding, padding), (padding, padding))
        )
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1
    out = np.zeros((n, c_out, out_h, out_w))
    for ni in range(n):
        for co in range(c_out):
            for oh in range(out_h):
                for ow in range(out_w):
                    val = 0.0
                    for ci in range(c_in):
                        for ki in range(kh):
                            for kj in range(kw):
                                val += (
                                    inp[ni, ci, oh * stride + ki, ow * stride + kj]
                                    * weight[co, ci, ki, kj]
                                )
                    if bias is not None:
                        val += bias[co]
                    out[ni, co, oh, ow] = val
    return out


class TestConv2d:
    def test_no_padding_stride1(self):
        np.random.seed(1)
        inp_np = np.random.rand(1, 1, 5, 5)
        w_np = np.random.rand(1, 1, 3, 3)
        expected = _numpy_conv2d(inp_np, w_np)
        result = conv2d(_to_ndarray(inp_np), _to_ndarray(w_np))
        _assert_close(result, expected, atol=1e-5)

    def test_with_padding(self):
        np.random.seed(2)
        inp_np = np.random.rand(1, 1, 4, 4)
        w_np = np.random.rand(1, 1, 3, 3)
        expected = _numpy_conv2d(inp_np, w_np, padding=1)
        result = conv2d(_to_ndarray(inp_np), _to_ndarray(w_np), padding=1)
        _assert_close(result, expected, atol=1e-5)

    def test_stride2(self):
        np.random.seed(3)
        inp_np = np.random.rand(1, 1, 6, 6)
        w_np = np.random.rand(1, 1, 3, 3)
        expected = _numpy_conv2d(inp_np, w_np, stride=2)
        result = conv2d(_to_ndarray(inp_np), _to_ndarray(w_np), stride=2)
        _assert_close(result, expected, atol=1e-5)

    def test_multi_channel(self):
        np.random.seed(4)
        inp_np = np.random.rand(2, 3, 8, 8)
        w_np = np.random.rand(4, 3, 3, 3)
        b_np = np.random.rand(4)
        expected = _numpy_conv2d(inp_np, w_np, bias=b_np, stride=1, padding=1)
        result = conv2d(
            _to_ndarray(inp_np),
            _to_ndarray(w_np),
            bias=NDArray(b_np.tolist()),
            stride=1,
            padding=1,
        )
        _assert_close(result, expected, atol=1e-4)

    def test_no_bias(self):
        np.random.seed(5)
        inp_np = np.random.rand(1, 2, 5, 5)
        w_np = np.random.rand(3, 2, 3, 3)
        expected = _numpy_conv2d(inp_np, w_np, stride=1, padding=0)
        result = conv2d(_to_ndarray(inp_np), _to_ndarray(w_np))
        _assert_close(result, expected, atol=1e-5)


# ===========================================================================
#  12. Arithmetic
# ===========================================================================


class TestArithmetic:
    def test_add(self):
        a = NDArray([1, 2, 3])
        b = NDArray([4, 5, 6])
        _assert_close(a + b, np.array([5, 7, 9], dtype=float))

    def test_add_scalar(self):
        a = NDArray([1, 2, 3])
        _assert_close(a + 10, np.array([11, 12, 13], dtype=float))

    def test_sub(self):
        a = NDArray([10, 20, 30])
        b = NDArray([1, 2, 3])
        _assert_close(a - b, np.array([9, 18, 27], dtype=float))

    def test_mul(self):
        a = NDArray([2, 3, 4])
        b = NDArray([5, 6, 7])
        _assert_close(a * b, np.array([10, 18, 28], dtype=float))

    def test_neg(self):
        a = NDArray([1, -2, 3])
        _assert_close(-a, np.array([-1, 2, -3], dtype=float))


# ===========================================================================
#  13. Performance benchmarks
# ===========================================================================


class TestPerformance:
    """Timing benchmarks — always pass, but print elapsed time."""

    def _report(self, name: str, elapsed: float) -> None:
        print(f"  [PERF] {name:30s}  {elapsed*1000:8.2f} ms")

    # -- creation / indexing --

    def test_perf_creation(self):
        data = list(range(10000))
        _, t = _timer(NDArray, data)
        self._report("creation(10k)", t)

    def test_perf_slice(self):
        a = arange(10000).view(100, 100)
        _, t = _timer(lambda: a[10:90, 20:80])
        self._report("slice(100x100)", t)

    # -- view / reshape --

    def test_perf_view(self):
        a = arange(10000)
        _, t = _timer(lambda: a.view(100, 100))
        self._report("view(10k→100x100)", t)

    def test_perf_reshape_copy(self):
        a = arange(10000).view(100, 100)[:, ::2]   # non-contiguous
        _, t = _timer(lambda: a.reshape(5000))
        self._report("reshape+copy(5k)", t)

    # -- squeeze / unsqueeze --

    def test_perf_unsqueeze(self):
        a = arange(1000)
        _, t = _timer(lambda: a.unsqueeze(0))
        self._report("unsqueeze", t)

    # -- cat / stack --

    def test_perf_cat(self):
        parts = [arange(100).view(10, 10) for _ in range(10)]
        _, t = _timer(cat, parts, dim=0)
        self._report("cat(10x 10x10, dim=0)", t)

    def test_perf_stack(self):
        parts = [arange(100).view(10, 10) for _ in range(10)]
        _, t = _timer(stack, parts, dim=0)
        self._report("stack(10x 10x10)", t)

    # -- split / chunk --

    def test_perf_split(self):
        a = arange(10000).view(100, 100)
        _, t = _timer(split, a, 10, 0)
        self._report("split(100x100, sz=10)", t)

    def test_perf_chunk(self):
        a = arange(10000).view(100, 100)
        _, t = _timer(chunk, a, 5, 0)
        self._report("chunk(100x100, n=5)", t)

    # -- gather --

    def test_perf_gather(self):
        src = arange(100).view(10, 10)
        idx = _to_ndarray(np.random.randint(0, 10, size=(10, 10)).astype(float))
        _, t = _timer(lambda: src.gather(1, idx))
        self._report("gather(10x10)", t)

    # -- matmul --

    def test_perf_matmul_small(self):
        np.random.seed(0)
        a = _to_ndarray(np.random.rand(32, 32))
        b = _to_ndarray(np.random.rand(32, 32))
        _, t = _timer(lambda: a @ b)
        self._report("matmul(32x32)", t)

    def test_perf_matmul_medium(self):
        np.random.seed(0)
        a = _to_ndarray(np.random.rand(64, 64))
        b = _to_ndarray(np.random.rand(64, 64))
        _, t = _timer(lambda: a @ b)
        self._report("matmul(64x64)", t)

    # -- conv2d --

    def test_perf_conv2d_small(self):
        np.random.seed(0)
        inp = _to_ndarray(np.random.rand(1, 1, 8, 8))
        w = _to_ndarray(np.random.rand(1, 1, 3, 3))
        _, t = _timer(conv2d, inp, w, stride=1, padding=1)
        self._report("conv2d(1,1,8,8 k=3 p=1)", t)

    def test_perf_conv2d_multi(self):
        np.random.seed(0)
        inp = _to_ndarray(np.random.rand(1, 3, 8, 8))
        w = _to_ndarray(np.random.rand(4, 3, 3, 3))
        _, t = _timer(conv2d, inp, w, stride=1, padding=1)
        self._report("conv2d(1,3,8,8 co=4 k=3 p=1)", t)
