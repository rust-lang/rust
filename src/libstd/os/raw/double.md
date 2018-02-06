Equivalent to C's `double` type.

This type will almost always be [`f64`], which is guaranteed to be an [IEEE-754 double-precision float] in Rust. That said, the standard technically only guarantees that it be a floating-point number with at least the precision of a [`float`], and it may be `f32` or something entirely different from the IEEE-754 standard.

[IEEE-754 double-precision float]: https://en.wikipedia.org/wiki/IEEE_754
[`float`]: type.c_float.html
[`f64`]: ../../primitive.f64.html
