// aux-build:derive-a.rs

#![feature(rustc_attrs)]
#![warn(unused_extern_crates)]

extern crate derive_a;
//~^ WARN unused extern crate

#[rustc_error]
fn main() {} //~ ERROR compilation successful
