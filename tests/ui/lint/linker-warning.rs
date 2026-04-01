//@ check-pass
#![crate_type = "lib"]
#![warn(unused_attributes)]
#![allow(linker_messages)]
//~^ WARNING unused attribute

#[allow(linker_messages)]
//~^ WARNING should be an inner attribute
fn foo() {}
