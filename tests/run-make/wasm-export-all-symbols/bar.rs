#![crate_type = "rlib"]

#[no_mangle]
pub extern fn foo() {}

#[no_mangle]
pub static FOO: u64 = 42;
