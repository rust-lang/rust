//@ compile-flags: -C control-flow-guard=checks
//@ only-msvc

#![crate_type = "lib"]

// A basic test function.
pub fn test() {}

// Ensure the module flag cfguard=2 is present
// CHECK: !"cfguard", i32 2
