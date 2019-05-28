// aux-build:test-macros.rs

#![feature(rustc_attrs)]
#![warn(unused_extern_crates)]

extern crate test_macros;
//~^ WARN unused extern crate

#[rustc_error]
fn main() {} //~ ERROR compilation successful
