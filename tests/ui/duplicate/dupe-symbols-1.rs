//@ build-fail

//
#![crate_type="rlib"]
#![allow(warnings)]

#[export_name="fail"]
pub fn a() {
}

#[export_name="fail"]
pub fn b() {
//~^ ERROR symbol `fail` is already defined
}
