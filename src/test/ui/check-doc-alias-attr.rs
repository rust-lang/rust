#![crate_type = "lib"]

#[doc(alias = "foo")] // ok!
pub struct Bar;

#[doc(alias)] //~ ERROR
#[doc(alias = 0)] //~ ERROR
#[doc(alias("bar"))] //~ ERROR
#[doc(alias = "\"")] //~ ERROR
#[doc(alias = "\n")] //~ ERROR
#[doc(alias = "
")] //~^ ERROR
#[doc(alias = "\t")] //~ ERROR
#[doc(alias = " hello")] //~ ERROR
#[doc(alias = "hello ")] //~ ERROR
pub struct Foo;
