// This is exactly like `msvc-prefix-good.rs`, except that it should always fail.

//@ should-fail

// MSVC: text that should not match
// NONMSVC: text that should not match
fn main() {}
