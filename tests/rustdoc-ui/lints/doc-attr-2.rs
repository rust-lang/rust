#![doc(as_ptr)]
//~^ ERROR unknown `doc` attribute `as_ptr`

#[doc(as_ptr)]
//~^ ERROR unknown `doc` attribute `as_ptr`
pub fn foo() {}

#[doc(foo::bar, crate::bar::baz = "bye")]
//~^ ERROR unknown `doc` attribute `foo::bar`
//~| ERROR unknown `doc` attribute `crate::bar::baz`
fn bar() {}
