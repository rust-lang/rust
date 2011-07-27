


// Regression test for issue #152.
fn main() { let b: uint = 1u; while b <= 32u { 0u << b; b <<= 1u; log b; } }