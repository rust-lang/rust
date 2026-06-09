//@ build-fail

#![crate_type="rlib"]
#![allow(warnings)]

#[export_name="fail"]
static HELLO: u8 = 0;

#[export_name="fail"]
static HELLO_TWICE: u16 = 0;
//~^ ERROR symbol `fail` is already defined
