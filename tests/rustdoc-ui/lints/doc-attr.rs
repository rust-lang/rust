#![crate_type = "lib"]
#![doc(as_ptr)]
//~^ ERROR unknown `doc` attribute

#[doc(as_ptr)]
//~^ ERROR unknown `doc` attribute
pub fn foo() {}

#[doc(123)]
//~^ ERROR invalid `doc` attribute
#[doc("hello", "bar")]
//~^ ERROR invalid `doc` attribute
//~| ERROR invalid `doc` attribute
#[doc(foo::bar, crate::bar::baz = "bye")]
//~^ ERROR unknown `doc` attribute
//~| ERROR unknown `doc` attribute
fn bar() {}
