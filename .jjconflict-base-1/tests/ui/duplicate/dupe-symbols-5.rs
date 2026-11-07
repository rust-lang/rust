//@ build-fail

//
#![crate_type="rlib"]
#![allow(warnings)]

#[export_name="fail"]
static HELLO: u8 = 0;

#[export_name="fail"]
pub fn b() {
//~^ ERROR symbol `fail` is already defined
}
