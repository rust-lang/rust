Returns `None` if the pointer is null, or else returns a shared reference to
the value wrapped in `Some`. If the value may be uninitialized, [`as_uninit_ref`]
must be used instead.

# Safety

When calling this method, you have to ensure that *either* the pointer is null *or*
the pointer is [convertible to a reference](crate::ptr#pointer-to-reference-conversion).

# Panics during const evaluation

This method will panic during const evaluation if the pointer cannot be
determined to be null or not. See [`is_null`] for more information.

# Null-unchecked version

If you are sure the pointer can never be null and are looking for some kind of
`as_ref_unchecked` that returns the `&T` instead of `Option<&T>`, know that you can
dereference the pointer directly.
