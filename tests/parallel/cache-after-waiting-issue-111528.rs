// Test for #111528, the ice issue cause waiting on a query that panicked
//
//@ compile-flags: -Z threads=16

#![crate_type = "rlib"]
#![allow(warnings)]

#[export_name = "fail"]
pub fn a() {}

#[export_name = "fail"]
pub fn b() {}

fn main() {}
