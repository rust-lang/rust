//@ compile-flags: -C control-flow-guard=no
//@ only-msvc

#![crate_type = "lib"]

// A basic test function.
pub fn test() {}

// Ensure the module flag cfguard is not present
// CHECK-NOT: !"cfguard"
