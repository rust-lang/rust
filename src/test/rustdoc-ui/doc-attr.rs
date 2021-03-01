#![crate_type = "lib"]
#![doc(as_ptr)] //~ ERROR

#[doc(as_ptr)] //~ ERROR
pub fn foo() {}
