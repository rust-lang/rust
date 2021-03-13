#![crate_type = "lib"]
#![deny(warnings)]
#![doc(as_ptr)]
//~^ ERROR unknown `doc` attribute
//~^^ WARN

#[doc(as_ptr)]
//~^ ERROR unknown `doc` attribute
//~^^ WARN
pub fn foo() {}

#[doc(123)]
//~^ ERROR unknown `doc` attribute
//~| WARN
#[doc("hello", "bar")]
//~^ ERROR unknown `doc` attribute
//~| WARN
fn bar() {}
