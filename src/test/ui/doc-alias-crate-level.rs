// compile-flags: -Zdeduplicate-diagnostics=no

#![feature(doc_alias)]

#![crate_type = "lib"]

#![doc(alias = "not working!")] //~ ERROR

#[doc(alias = "shouldn't work!")] //~ ERROR
pub struct Foo;
