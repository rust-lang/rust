//@ check-pass
#![crate_type = "lib"]
#![warn(unused_attributes)]
#![allow(linker_messages)]
//~^ WARN unused attribute

#[allow(linker_messages)]
//~^ WARN allow(linker_messages) is ignored unless specified at crate level
fn foo() {}
