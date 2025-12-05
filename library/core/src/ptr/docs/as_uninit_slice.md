Returns `None` if the pointer is null, or else returns a shared slice to
the value wrapped in `Some`. In contrast to [`as_ref`], this does not require
that the value has to be initialized.

[`as_ref`]: #method.as_ref

# Safety

When calling this method, you have to ensure that *either* the pointer is null *or*
all of the following is true:

* The pointer must be [valid] for reads for `ptr.len() * size_of::<T>()` many bytes,
  and it must be properly aligned. This means in particular:

* The entire memory range of this slice must be contained within a single [allocation]!
  Slices can never span across multiple allocations.

* The pointer must be aligned even for zero-length slices. One
  reason for this is that enum layout optimizations may rely on references
  (including slices of any length) being aligned and non-null to distinguish
  them from other data. You can obtain a pointer that is usable as `data`
  for zero-length slices using [`NonNull::dangling()`].

* The total size `ptr.len() * size_of::<T>()` of the slice must be no larger than `isize::MAX`.
  See the safety documentation of [`pointer::offset`].

* You must enforce Rust's aliasing rules, since the returned lifetime `'a` is
  arbitrarily chosen and does not necessarily reflect the actual lifetime of the data.
  In particular, while this reference exists, the memory the pointer points to must
  not get mutated (except inside `UnsafeCell`).

This applies even if the result of this method is unused!

See also [`slice::from_raw_parts`][].

[valid]: crate::ptr#safety
[allocation]: crate::ptr#allocation

# Panics during const evaluation

This method will panic during const evaluation if the pointer cannot be
determined to be null or not. See [`is_null`] for more information.

[`is_null`]: #method.is_null
