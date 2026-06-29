Subtracts an unsigned offset from a pointer.

This can only move the pointer backward (or not move it). If you need to move forward or
backward depending on the value, then you might want [`offset`](#method.offset) instead
which takes a signed offset.

`count` is in units of T; e.g., a `count` of 3 represents a pointer
offset of `3 * size_of::<T>()` bytes.

# Safety

If any of the following conditions are violated, the result is Undefined Behavior:

* The offset in bytes, `count * size_of::<T>()`, computed on mathematical integers (without
  "wrapping around"), must fit in an `isize`.

* Let `result` be `self.addr() - count * size_of::<T>()`, computed on mathematical integers.
This must fit in a `usize`.

* If the computed offset is non-zero, then `self` must be [derived from][crate::ptr#provenance] a pointer to some
[allocation], and the entire memory range between `self` and `result`
(i.e., `result..self.addr()`) must be in bounds of that allocation.

Allocations can never be larger than `isize::MAX` bytes and they can only contain addresses
representable by `usize`, so technically the last condition implies the first two.

[allocation]: crate::ptr#allocation
