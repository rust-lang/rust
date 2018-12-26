// compile-flags: -O

#![crate_type = "lib"]

// CHECK: Function Attrs: norecurse nounwind
pub extern fn foo() {}
