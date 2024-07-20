// This is exactly like `msvc-prefix-good.rs`, except that it should always fail.

//@ should-fail

// CHECK-MSVC: text that should not match
// CHECK-NONMSVC: text that should not match
fn main() {}
