// compile-flags: -Z control_flow_guard=disabled

#![crate_type = "lib"]

// A basic test function.
pub fn test() {
}

// Ensure the module flag cfguard is not present
// CHECK-NOT: !"cfguard"
