Equivalent to C's `signed long` (`long`) type.

This type will always be [`i32`] or [`i64`]. Most notably, many Linux-based systems assume an `i64`, but Windows assumes `i32`. The C standard technically only requires that this type be a signed integer that is at least 32 bits and at least the size of an [`int`], although in practice, no system would have a `long` that is neither an `i32` nor `i64`.

[`int`]: c_int
