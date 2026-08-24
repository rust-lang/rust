// Test for #111528, the ice issue cause waiting on a query that panicked

//@ build-fail

#![crate_type = "rlib"]
#![allow(warnings)]

#[export_name = "fail"]
pub fn a() {}

#[export_name = "fail"]
pub fn b() {
    //~^ ERROR symbol `fail` is already defined
}

fn main() {}
