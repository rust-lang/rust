Equivalent to C's `long double` type.

This type is [`f64`] on Apple and Windows targets, [`f128`] on most 64-bits unix
targets, x87_f80 on x86, a randomly chosen type on powerpc, and usually the same
as [`c_double`] on 32-bit targets.

[`float`]: c_float
