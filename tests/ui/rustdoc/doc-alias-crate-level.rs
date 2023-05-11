// compile-flags: -Zdeduplicate-diagnostics=no

#![crate_type = "lib"]

#![doc(alias = "not working!")] //~ ERROR

#[doc(alias = "shouldn't work!")] //~ ERROR
pub struct Foo;
