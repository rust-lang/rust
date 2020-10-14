#![crate_type = "lib"]
#![feature(doc_alias)]

#[doc(alias = "foo")] // ok!
pub struct Bar;

#[doc(alias)] //~ ERROR
#[doc(alias = 0)] //~ ERROR
#[doc(alias("bar"))] //~ ERROR
#[doc(alias = "\"")] //~ ERROR
#[doc(alias = "\n")] //~ ERROR
#[doc(alias = "
")] //~^ ERROR
#[doc(alias = " ")] //~ ERROR
#[doc(alias = "\t")] //~ ERROR
pub struct Foo;
