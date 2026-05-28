//@ compile-flags: -C control-flow-guard=nochecks
//@ only-msvc

#![crate_type = "lib"]

// A basic test function.
pub fn test() {}

// Ensure the module flag cfguard=1 is present
// CHECK: !"cfguard", i32 1
