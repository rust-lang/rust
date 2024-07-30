// One of MSVC or NONMSVC should always be defined, so this test should pass.

// (one of these should always be present)

// CHECK-MSVC: main
// CHECK-NONMSVC: main
fn main() {}
