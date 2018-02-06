Equivalent to C's `unsigned long` type.

This type will always be [`u32`] or [`u64`]. Most notably, many Linux-based systems assume an `u64`, but Windows assumes `u32`. The C standard technically only requires that this type be an unsigned integer with the size of a [`long`], although in practice, no system would have a `ulong` that is neither a `u32` nor `u64`.

[`long`]: type.c_long.html
[`u32`]: ../../primitive.u32.html
[`u64`]: ../../primitive.u64.html
