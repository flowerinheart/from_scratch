"""NDArray: a PyTorch-like multi-dimensional array backed by a flat list with offset + strides."""

from __future__ import annotations

import itertools
from functools import reduce
from typing import Optional, Union


def _prod(iterable) -> int:
    """Product of all elements; returns 1 for empty sequence."""
    return reduce(lambda a, b: a * b, iterable, 1)


# ---------------------------------------------------------------------------
#  NDArray core
# ---------------------------------------------------------------------------

class NDArray:
    """Multi-dimensional array backed by a 1-D ``list[float]``.

    Multiple NDArrays can share the same underlying ``_data`` list; the
    combination of ``_offset``, ``_shape`` and ``_strides`` determines
    which elements each view refers to.

    Addressing formula::

        element(i0, i1, ..., in) = _data[_offset + sum(ik * _strides[k])]
    """

    # ------------------------------------------------------------------ init

    def __init__(self, data: Union[list, float, int]) -> None:
        """Create an NDArray from a (nested) Python list or scalar."""
        self._shape = self._infer_shape(data)
        self._data = self._flatten(data)
        self._strides = self._compute_strides(self._shape)
        self._offset = 0

    @classmethod
    def _make(
        cls,
        data: list[float],
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        offset: int,
    ) -> NDArray:
        """Create a view over existing storage — no copy."""
        obj = object.__new__(cls)
        obj._data = data
        obj._shape = tuple(shape)
        obj._strides = tuple(strides)
        obj._offset = offset
        return obj

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        """Row-major (C-order) strides for *shape*."""
        if not shape:
            return ()
        strides = [0] * len(shape)
        strides[-1] = 1
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    @staticmethod
    def _infer_shape(data) -> tuple[int, ...]:
        shape: list[int] = []
        cur = data
        while isinstance(cur, (list, tuple)):
            shape.append(len(cur))
            if len(cur) == 0:
                break
            cur = cur[0]
        return tuple(shape)

    @staticmethod
    def _flatten(data) -> list[float]:
        if not isinstance(data, (list, tuple)):
            return [float(data)]
        result: list[float] = []
        for item in data:
            if isinstance(item, (list, tuple)):
                result.extend(NDArray._flatten(item))
            else:
                result.append(float(item))
        return result

    # ------------------------------------------------------------- properties

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return _prod(self._shape) if self._shape else 1

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides

    @property
    def offset(self) -> int:
        return self._offset

    # ----------------------------------------------------------- contiguity

    def is_contiguous(self) -> bool:
        """True when strides match the default row-major layout."""
        return self._strides == self._compute_strides(self._shape)

    def contiguous(self) -> NDArray:
        """Return self if already contiguous, else a fresh contiguous copy."""
        if self.is_contiguous():
            return self
        new_data = list(self._iter_elements())
        return NDArray._make(
            new_data, self._shape, self._compute_strides(self._shape), 0
        )

    # -------------------------------------------------------- element access

    def _flat_index(self, idx: tuple[int, ...]) -> int:
        return self._offset + sum(i * s for i, s in zip(idx, self._strides))

    def _iter_indices(self):
        if self.ndim == 0:
            yield ()
            return
        yield from itertools.product(*(range(s) for s in self._shape))

    def _iter_elements(self):
        for idx in self._iter_indices():
            yield self._data[self._flat_index(idx)]

    # ------------------------------------------------------------ indexing

    def _resolve_key(self, key):
        """Return ``(offset, shape, strides)`` for an index *key*."""
        if not isinstance(key, tuple):
            key = (key,)
        offset = self._offset
        new_shape: list[int] = []
        new_strides: list[int] = []
        dim = 0
        for k in key:
            if isinstance(k, int):
                if k < 0:
                    k += self._shape[dim]
                if k < 0 or k >= self._shape[dim]:
                    raise IndexError(
                        f"index {k} out of range for dim {dim} "
                        f"with size {self._shape[dim]}"
                    )
                offset += k * self._strides[dim]
                dim += 1
            elif isinstance(k, slice):
                start, stop, step = k.indices(self._shape[dim])
                offset += start * self._strides[dim]
                new_shape.append(len(range(start, stop, step)))
                new_strides.append(self._strides[dim] * step)
                dim += 1
            else:
                raise TypeError(f"Invalid index type: {type(k)}")
        for d in range(dim, self.ndim):
            new_shape.append(self._shape[d])
            new_strides.append(self._strides[d])
        return offset, tuple(new_shape), tuple(new_strides)

    def __getitem__(self, key) -> Union[NDArray, float]:
        offset, shape, strides = self._resolve_key(key)
        if not shape:
            return self._data[offset]
        return NDArray._make(self._data, shape, strides, offset)

    def __setitem__(self, key, value) -> None:
        offset, shape, strides = self._resolve_key(key)
        if not shape:
            self._data[offset] = float(value)
            return
        target = NDArray._make(self._data, shape, strides, offset)
        if isinstance(value, NDArray):
            for idx in target._iter_indices():
                target._data[target._flat_index(idx)] = value._data[
                    value._flat_index(idx)
                ]
        else:
            val = float(value)
            for idx in target._iter_indices():
                target._data[target._flat_index(idx)] = val

    # ----------------------------------------------------------- conversion

    def tolist(self):
        """Convert to a (nested) Python list.  Scalars become plain float."""
        if self.ndim == 0:
            return self._data[self._offset]
        if self.ndim == 1:
            return [
                self._data[self._offset + i * self._strides[0]]
                for i in range(self._shape[0])
            ]
        return [self[i].tolist() for i in range(self._shape[0])]

    def __repr__(self) -> str:
        return f"NDArray({self.tolist()})"

    # ============================================================ Shape ops
    # ============================================================

    def _resolve_shape(self, shape) -> tuple[int, ...]:
        """Resolve a shape spec that may contain one ``-1``."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        neg_count = sum(1 for s in shape if s == -1)
        if neg_count > 1:
            raise ValueError("At most one dimension can be -1")
        if neg_count == 1:
            known = _prod(s for s in shape if s != -1)
            if known == 0:
                raise ValueError("Cannot infer dimension with 0-size tensor")
            shape = tuple(self.size // known if s == -1 else s for s in shape)
        return shape

    def view(self, *shape) -> NDArray:
        """New shape over the *same* storage (requires contiguous)."""
        shape = self._resolve_shape(shape)
        if not self.is_contiguous():
            raise RuntimeError("view requires a contiguous tensor")
        if _prod(shape) != self.size:
            raise ValueError(
                f"shape {shape} is invalid for size {self.size}"
            )
        return NDArray._make(
            self._data, shape, self._compute_strides(shape), self._offset
        )

    def reshape(self, *shape) -> NDArray:
        """Like *view* but copies data when necessary."""
        shape = self._resolve_shape(shape)
        if _prod(shape) != self.size:
            raise ValueError(
                f"shape {shape} is invalid for size {self.size}"
            )
        if self.is_contiguous():
            return NDArray._make(
                self._data, shape, self._compute_strides(shape), self._offset
            )
        return self.contiguous().view(*shape)

    def squeeze(self, dim: Optional[int] = None) -> NDArray:
        """Remove size-1 dimensions (or a specific *dim*)."""
        if dim is not None:
            if dim < 0:
                dim += self.ndim
            if self._shape[dim] != 1:
                return NDArray._make(
                    self._data, self._shape, self._strides, self._offset
                )
            new_shape = self._shape[:dim] + self._shape[dim + 1 :]
            new_strides = self._strides[:dim] + self._strides[dim + 1 :]
        else:
            pairs = [
                (s, st)
                for s, st in zip(self._shape, self._strides)
                if s != 1
            ]
            if pairs:
                new_shape, new_strides = map(tuple, zip(*pairs))
            else:
                new_shape, new_strides = (), ()
        return NDArray._make(self._data, new_shape, new_strides, self._offset)

    def unsqueeze(self, dim: int) -> NDArray:
        """Insert a size-1 dimension at *dim*."""
        if dim < 0:
            dim += self.ndim + 1
        if dim < 0 or dim > self.ndim:
            raise IndexError(
                f"dim {dim} out of range for tensor with {self.ndim} dims"
            )
        new_shape = self._shape[:dim] + (1,) + self._shape[dim:]
        if dim < self.ndim:
            stride_val = self._strides[dim] * self._shape[dim]
        else:
            stride_val = 1
        new_strides = self._strides[:dim] + (stride_val,) + self._strides[dim:]
        return NDArray._make(self._data, new_shape, new_strides, self._offset)

    # ============================================================ Gather
    # ============================================================

    def gather(self, dim: int, index: NDArray) -> NDArray:
        """Gather values along *dim* according to *index*.

        ``out[i][j][k] = self[index[i][j][k]][j][k]``  (when dim == 0)
        """
        if dim < 0:
            dim += self.ndim
        out_shape = index.shape
        out_data = [0.0] * _prod(out_shape)
        out_strides = self._compute_strides(out_shape)
        for idx in index._iter_indices():
            index_val = int(index._data[index._flat_index(idx)])
            src_idx = list(idx)
            src_idx[dim] = index_val
            out_data[
                sum(i * s for i, s in zip(idx, out_strides))
            ] = self._data[self._flat_index(tuple(src_idx))]
        return NDArray._make(out_data, out_shape, out_strides, 0)

    # ============================================================ Matmul
    # ============================================================

    def _matmul_2d(self, other: NDArray) -> NDArray:
        m, k1 = self._shape
        k2, n = other._shape
        if k1 != k2:
            raise ValueError(
                f"matmul inner-dim mismatch: ({m},{k1}) @ ({k2},{n})"
            )
        out_data = [0.0] * (m * n)
        sa0, sa1 = self._strides
        sb0, sb1 = other._strides
        a_off, b_off = self._offset, other._offset
        a_data, b_data = self._data, other._data
        for i in range(m):
            row_base = a_off + i * sa0
            for j in range(n):
                total = 0.0
                col_base = b_off + j * sb1
                for kk in range(k1):
                    total += a_data[row_base + kk * sa1] * b_data[
                        col_base + kk * sb0
                    ]
                out_data[i * n + j] = total
        return NDArray._make(out_data, (m, n), (n, 1), 0)

    def matmul(self, other: NDArray) -> NDArray:
        """Matrix multiplication (2-D and batched)."""
        if self.ndim == 2 and other.ndim == 2:
            return self._matmul_2d(other)
        if self.ndim < 2 or other.ndim < 2:
            raise ValueError("matmul requires tensors with ndim >= 2")
        a_batch = self._shape[:-2]
        b_batch = other._shape[:-2]
        batch_shape = _broadcast_shapes(a_batch, b_batch)
        m = self._shape[-2]
        n = other._shape[-1]
        out_shape = batch_shape + (m, n)
        out_data = [0.0] * _prod(out_shape)
        out_strides = self._compute_strides(out_shape)
        batch_iter = (
            [()]
            if not batch_shape
            else itertools.product(*(range(s) for s in batch_shape))
        )
        for b_idx in batch_iter:
            a_sub = self
            for i in _broadcast_index(b_idx, a_batch):
                a_sub = a_sub[i]
            b_sub = other
            for i in _broadcast_index(b_idx, b_batch):
                b_sub = b_sub[i]
            r2d = a_sub._matmul_2d(b_sub)
            base = sum(bi * s for bi, s in zip(b_idx, out_strides))
            for i in range(m):
                for j in range(n):
                    out_data[base + i * out_strides[-2] + j * out_strides[-1]] = (
                        r2d._data[i * n + j]
                    )
        return NDArray._make(out_data, out_shape, out_strides, 0)

    def __matmul__(self, other: NDArray) -> NDArray:
        return self.matmul(other)

    # ======================================================= Arithmetic
    # =======================================================

    def _elementwise(self, other, op) -> NDArray:
        out_data = [0.0] * self.size
        out_strides = self._compute_strides(self._shape)
        if isinstance(other, NDArray):
            if self.shape != other.shape:
                raise ValueError(
                    f"shape mismatch: {self.shape} vs {other.shape}"
                )
            for idx in self._iter_indices():
                a = self._data[self._flat_index(idx)]
                b = other._data[other._flat_index(idx)]
                out_data[sum(i * s for i, s in zip(idx, out_strides))] = op(a, b)
        else:
            b = float(other)
            for idx in self._iter_indices():
                a = self._data[self._flat_index(idx)]
                out_data[sum(i * s for i, s in zip(idx, out_strides))] = op(a, b)
        return NDArray._make(out_data, self._shape, out_strides, 0)

    def __add__(self, other):
        return self._elementwise(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._elementwise(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._elementwise(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self._elementwise(other, lambda a, b: a / b)

    def __neg__(self):
        out = [-x for x in self._iter_elements()]
        return NDArray._make(out, self._shape, self._compute_strides(self._shape), 0)


# ===========================================================================
#  Module-level tensor operations
# ===========================================================================


def cat(tensors: list[NDArray], dim: int = 0) -> NDArray:
    """Concatenate tensors along an existing dimension."""
    if not tensors:
        raise ValueError("cat requires at least one tensor")
    ndim = tensors[0].ndim
    if dim < 0:
        dim += ndim
    for t in tensors[1:]:
        if t.ndim != ndim:
            raise ValueError("All tensors must have the same ndim")
        for i in range(ndim):
            if i != dim and t.shape[i] != tensors[0].shape[i]:
                raise ValueError(
                    f"Dim {i} size mismatch: "
                    f"{t.shape[i]} vs {tensors[0].shape[i]}"
                )
    cat_size = sum(t.shape[dim] for t in tensors)
    out_shape = list(tensors[0].shape)
    out_shape[dim] = cat_size
    out_shape = tuple(out_shape)
    out_data = [0.0] * _prod(out_shape)
    out_strides = NDArray._compute_strides(out_shape)
    dim_off = 0
    for t in tensors:
        for idx in t._iter_indices():
            out_idx = list(idx)
            out_idx[dim] += dim_off
            out_data[sum(i * s for i, s in zip(out_idx, out_strides))] = t._data[
                t._flat_index(idx)
            ]
        dim_off += t.shape[dim]
    return NDArray._make(out_data, out_shape, out_strides, 0)


def stack(tensors: list[NDArray], dim: int = 0) -> NDArray:
    """Stack tensors along a *new* dimension."""
    if not tensors:
        raise ValueError("stack requires at least one tensor")
    return cat([t.unsqueeze(dim) for t in tensors], dim=dim)


def split(
    tensor: NDArray,
    split_size_or_sections: Union[int, list[int]],
    dim: int = 0,
) -> list[NDArray]:
    """Split a tensor into chunks (returns views when possible)."""
    if dim < 0:
        dim += tensor.ndim
    dim_size = tensor.shape[dim]
    if isinstance(split_size_or_sections, int):
        sections: list[int] = []
        remaining = dim_size
        while remaining > 0:
            sections.append(min(split_size_or_sections, remaining))
            remaining -= sections[-1]
    else:
        sections = list(split_size_or_sections)
    result: list[NDArray] = []
    offset = 0
    for size in sections:
        slices = [slice(None)] * tensor.ndim
        slices[dim] = slice(offset, offset + size)
        result.append(tensor[tuple(slices)])
        offset += size
    return result


def chunk(tensor: NDArray, chunks: int, dim: int = 0) -> list[NDArray]:
    """Split a tensor into *chunks* approximately-equal pieces."""
    if dim < 0:
        dim += tensor.ndim
    dim_size = tensor.shape[dim]
    chunk_size = (dim_size + chunks - 1) // chunks
    return split(tensor, chunk_size, dim)


# ===========================================================================
#  Conv2D  (img2col → matmul)
# ===========================================================================


def _zero_pad(x: NDArray, padding: int) -> NDArray:
    """Pad ``(N, C, H, W)`` with zeros on the H / W axes."""
    if padding == 0:
        return x
    n, c, h, w = x.shape
    nh, nw = h + 2 * padding, w + 2 * padding
    out_shape = (n, c, nh, nw)
    out_data = [0.0] * _prod(out_shape)
    os = NDArray._compute_strides(out_shape)
    for ni in range(n):
        for ci in range(c):
            for hi in range(h):
                for wi in range(w):
                    out_data[
                        ni * os[0]
                        + ci * os[1]
                        + (hi + padding) * os[2]
                        + (wi + padding) * os[3]
                    ] = x._data[x._flat_index((ni, ci, hi, wi))]
    return NDArray._make(out_data, out_shape, os, 0)


def _img2col(x: NDArray, kh: int, kw: int, stride: int) -> NDArray:
    """Unfold ``(N, C, H, W)`` → ``(N, C*kH*kW, out_H*out_W)``."""
    n, c, h, w = x.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    col_shape = (n, c * kh * kw, out_h * out_w)
    col_data = [0.0] * _prod(col_shape)
    cs = NDArray._compute_strides(col_shape)
    for ni in range(n):
        for ci in range(c):
            for ki in range(kh):
                for kj in range(kw):
                    row = ci * kh * kw + ki * kw + kj
                    for oh in range(out_h):
                        for ow in range(out_w):
                            col = oh * out_w + ow
                            col_data[ni * cs[0] + row * cs[1] + col * cs[2]] = (
                                x._data[
                                    x._flat_index(
                                        (ni, ci, oh * stride + ki, ow * stride + kj)
                                    )
                                ]
                            )
    return NDArray._make(col_data, col_shape, cs, 0)


def conv2d(
    input_arr: NDArray,
    weight: NDArray,
    bias: Optional[NDArray] = None,
    stride: int = 1,
    padding: int = 0,
) -> NDArray:
    """2-D convolution via img2col + matmul.

    Parameters
    ----------
    input_arr : (N, C_in, H, W)
    weight    : (C_out, C_in, kH, kW)
    bias      : (C_out,) or None
    stride, padding : int
    """
    n, c_in, h, w = input_arr.shape
    c_out, c_in_w, kh, kw = weight.shape
    if c_in != c_in_w:
        raise ValueError(
            f"Channel mismatch: input {c_in}, weight {c_in_w}"
        )
    padded = _zero_pad(input_arr, padding)
    col = _img2col(padded, kh, kw, stride)       # (N, C_in*kH*kW, oH*oW)
    w2d = weight.reshape(c_out, c_in * kh * kw)  # (C_out, C_in*kH*kW)
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1
    out_shape = (n, c_out, out_h, out_w)
    out_data = [0.0] * _prod(out_shape)
    os = NDArray._compute_strides(out_shape)
    for ni in range(n):
        col_2d = col[ni]                           # (C_in*kH*kW, oH*oW)
        r2d = w2d._matmul_2d(col_2d)               # (C_out, oH*oW)
        for ci in range(c_out):
            bias_val = 0.0
            if bias is not None:
                bias_val = bias._data[bias._offset + ci * bias._strides[0]]
            for oi in range(out_h * out_w):
                out_data[
                    ni * os[0]
                    + ci * os[1]
                    + (oi // out_w) * os[2]
                    + (oi % out_w) * os[3]
                ] = r2d._data[r2d._flat_index((ci, oi))] + bias_val
    return NDArray._make(out_data, out_shape, os, 0)


# ===========================================================================
#  Broadcast helpers  (for batched matmul)
# ===========================================================================


def _broadcast_shapes(
    shape_a: tuple[int, ...], shape_b: tuple[int, ...]
) -> tuple[int, ...]:
    nd = max(len(shape_a), len(shape_b))
    a = (1,) * (nd - len(shape_a)) + shape_a
    b = (1,) * (nd - len(shape_b)) + shape_b
    out: list[int] = []
    for sa, sb in zip(a, b):
        if sa == sb:
            out.append(sa)
        elif sa == 1:
            out.append(sb)
        elif sb == 1:
            out.append(sa)
        else:
            raise ValueError(
                f"Cannot broadcast {shape_a} and {shape_b}"
            )
    return tuple(out)


def _broadcast_index(
    idx: tuple[int, ...], shape: tuple[int, ...]
) -> tuple[int, ...]:
    diff = len(idx) - len(shape)
    return tuple(0 if s == 1 else idx[i + diff] for i, s in enumerate(shape))


# ===========================================================================
#  Factory helpers
# ===========================================================================


def zeros(*shape) -> NDArray:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    data = [0.0] * _prod(shape)
    return NDArray._make(data, shape, NDArray._compute_strides(shape), 0)


def ones(*shape) -> NDArray:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    data = [1.0] * _prod(shape)
    return NDArray._make(data, shape, NDArray._compute_strides(shape), 0)


def arange(n: int) -> NDArray:
    data = [float(i) for i in range(n)]
    return NDArray._make(data, (n,), (1,), 0)
