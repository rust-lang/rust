// compile-flags: -Z control-flow-guard=checks

#![crate_type = "lib"]

// A basic test function.
pub fn test() {
}

// Ensure the module flag cfguard=2 is present
// CHECK: !"cfguard", i32 2
