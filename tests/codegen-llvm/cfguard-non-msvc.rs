//@ compile-flags: -C control-flow-guard
//@ ignore-msvc

#![crate_type = "lib"]

// A basic test function.
pub fn test() {}

// Ensure the cfguard module flag is not added for non-MSVC targets.
// CHECK-NOT: !"cfguard"
