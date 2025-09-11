Returns `None` if the pointer is null, or else returns a shared reference to
the value wrapped in `Some`. In contrast to [`as_ref`], this does not require
that the value has to be initialized.

# Safety

When calling this method, you have to ensure that *either* the pointer is null *or*
the pointer is [convertible to a reference](crate::ptr#pointer-to-reference-conversion).
Note that because the created reference is to `MaybeUninit<T>`, the
source pointer can point to uninitialized memory.

# Panics during const evaluation

This method will panic during const evaluation if the pointer cannot be
determined to be null or not. See [`is_null`] for more information.
