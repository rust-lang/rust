#![crate_type = "lib"]

#[doc(alias = "Foo")] //~ ERROR
pub struct Foo;
