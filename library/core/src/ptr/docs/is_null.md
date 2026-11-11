Returns `true` if the pointer is null.

Note that unsized types have many possible null pointers, as only the
raw data pointer is considered, not their length, vtable, etc.
Therefore, two pointers that are null may still not compare equal to
each other.

# Panics during const evaluation

If this method is used during const evaluation, and `self` is a pointer
that is offset beyond the bounds of the memory it initially pointed to,
then there might not be enough information to determine whether the
pointer is null. This is because the absolute address in memory is not
known at compile time. If the nullness of the pointer cannot be
determined, this method will panic.

In-bounds pointers are never null, so the method will never panic for
such pointers.
