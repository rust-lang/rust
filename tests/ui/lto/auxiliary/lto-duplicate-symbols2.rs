//@ no-prefer-dynamic

#![crate_type = "rlib"]

#[no_mangle]
pub extern "C" fn foo() {}
