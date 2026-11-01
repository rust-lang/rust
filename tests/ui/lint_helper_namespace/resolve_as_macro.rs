//@revisions: registered unregistered
//@[unregistered] check-pass

#![crate_type = "lib"]
#![cfg_attr(unregistered, feature(rustc_attrs))]
#![cfg_attr(registered, feature(register_tool))]
#![cfg_attr(registered, register_tool(lint))]

#[lint::as_ptr] //[registered]~ERROR attribute macro `lint::as_ptr` is ambiguous
fn foo() {}

mod lint {
    pub use std::prelude::v1::test as as_ptr;
}
