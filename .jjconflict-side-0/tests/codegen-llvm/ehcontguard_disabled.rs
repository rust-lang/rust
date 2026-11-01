#![crate_type = "lib"]

// A basic test function.
pub fn test() {}

// Ensure the module flag ehcontguard is not present
// CHECK-NOT: !"ehcontguard"
