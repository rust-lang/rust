# Unsafety

Unsafe operations are those that potentially violate the memory-safety
guarantees of Rust's static semantics.

The following language level features cannot be used in the safe subset of
Rust:

- Dereferencing a [raw pointer](#pointer-types).
- Reading or writing a [mutable static variable](#mutable-statics).
- Calling an unsafe function (including an intrinsic or foreign function).
