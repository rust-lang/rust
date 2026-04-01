Adds a signed offset to a pointer.

`count` is in units of T; e.g., a `count` of 3 represents a pointer
offset of `3 * size_of::<T>()` bytes.

# Safety

If any of the following conditions are violated, the result is Undefined Behavior:

* The offset in bytes, `count * size_of::<T>()`, computed on mathematical integers (without
"wrapping around"), must fit in an `isize`.

* If the computed offset is non-zero, then `self` must be [derived from][crate::ptr#provenance] a pointer to some
[allocation], and the entire memory range between `self` and the result must be in
bounds of that allocation. In particular, this range must not "wrap around" the edge
of the address space. Note that "range" here refers to a half-open range as usual in Rust,
i.e., `self..result` for non-negative offsets and `result..self` for negative offsets.

Allocations can never be larger than `isize::MAX` bytes, so if the computed offset
stays in bounds of the allocation, it is guaranteed to satisfy the first requirement.
This implies, for instance, that `vec.as_ptr().add(vec.len())` (for `vec: Vec<T>`) is always
safe.

Consider using [`wrapping_offset`] instead if these constraints are
difficult to satisfy. The only advantage of this method is that it
enables more aggressive compiler optimizations.

[`wrapping_offset`]: #method.wrapping_offset
[allocation]: crate::ptr#allocation
