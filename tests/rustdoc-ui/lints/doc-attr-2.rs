#![doc(as_ptr)]
//~^ ERROR reserved `doc` attribute `as_ptr`

#[doc(as_ptr)]
//~^ ERROR reserved `doc` attribute `as_ptr`
pub fn foo() {}

#[doc(foo::bar, crate::bar::baz = "bye")]
//~^ ERROR reserved `doc` attribute `foo::bar`
//~| ERROR reserved `doc` attribute `crate::bar::baz`
fn bar() {}
