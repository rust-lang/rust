#![crate_type = "lib"]
#![doc(as_ptr)]
//~^ ERROR unknown `doc` attribute

#[doc(as_ptr)]
//~^ ERROR unknown `doc` attribute
pub fn foo() {}

#[doc(123)]
//~^ ERROR
#[doc("hello", "bar")]
//~^ ERROR
//~| ERROR
#[doc(foo::bar, crate::bar::baz = "bye")]
//~^ ERROR unknown `doc` attribute
//~| ERROR unknown `doc` attribute
fn bar() {}
