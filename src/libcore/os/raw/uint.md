Equivalent to C's `unsigned int` type.

This type will almost always be [`u32`], but may differ on some esoteric systems. The C standard technically only requires that this type be an unsigned integer with the same size as an [`int`]; some systems define it as a [`u16`], for example.

[`int`]: type.c_int.html
[`u32`]: ../../primitive.u32.html
[`u16`]: ../../primitive.u16.html
