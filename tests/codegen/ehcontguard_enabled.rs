//@ compile-flags: -Z ehcont-guard

#![crate_type = "lib"]

// A basic test function.
pub fn test() {}

// Ensure the module flag ehcontguard=1 is present
// CHECK: !"ehcontguard", i32 1
