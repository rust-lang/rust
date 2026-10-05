#![deny(invalid_doc_attributes)]
#![crate_type = "lib"]
#![doc(as_ptr)]
//~^ ERROR reserved `doc` attribute

#[doc(as_ptr)]
//~^ ERROR reserved `doc` attribute
pub fn foo() {}

#[doc(123)]
//~^ ERROR
//~| WARN
#[doc("hello", "bar")]
//~^ ERROR
//~| ERROR
//~| WARN
//~| WARN
#[doc(foo::bar, crate::bar::baz = "bye")]
//~^ ERROR reserved `doc` attribute
//~| ERROR reserved `doc` attribute
fn bar() {}
