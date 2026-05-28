// Test for #111528, the ice issue cause waiting on a query that panicked
//
//@ compile-flags: -Z threads=16
//@ build-fail
//@ compare-output-by-lines

#![crate_type = "rlib"]
#![allow(warnings)]

#[export_name = "fail"]
pub fn a() {}

#[export_name = "fail"]
pub fn b() {
    //~^ ERROR symbol `fail` is already defined
}

fn main() {}
