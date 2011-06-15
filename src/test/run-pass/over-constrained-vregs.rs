


// Regression test for issue #152.
fn main() { let uint b = 1u; while (b <= 32u) { 0u << b; b <<= 1u; log b; } }