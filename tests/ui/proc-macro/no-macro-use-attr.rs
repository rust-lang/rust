//@ proc-macro: test-macros.rs

#![feature(rustc_attrs)]
#![warn(unused_extern_crates)]

extern crate test_macros;
//~^ WARN unused extern crate

#[rustc_error]
fn main() {} //~ ERROR fatal error triggered by #[rustc_error]
