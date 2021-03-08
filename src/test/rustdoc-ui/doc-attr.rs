#![crate_type = "lib"]
#![deny(warnings)]
#![doc(as_ptr)]
//~^ ERROR unknown `doc` attribute
//~^^ WARN

#[doc(as_ptr)]
//~^ ERROR unknown `doc` attribute
//~^^ WARN
pub fn foo() {}
