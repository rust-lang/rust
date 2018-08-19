Equivalent to C's `unsigned long long` type.

This type will almost always be [`u64`], but may differ on some systems. The C standard technically only requires that this type be an unsigned integer with the size of a [`long long`], although in practice, no system would have a `long long` that is not a `u64`, as most systems do not have a standardised [`u128`] type.

[`long long`]: type.c_longlong.html
[`u64`]: ../../primitive.u64.html
[`u128`]: ../../primitive.u128.html
