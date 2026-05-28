//@ check-pass
//@ proc-macro: test-macros.rs

#![warn(unused_extern_crates)]

extern crate test_macros;
//~^ WARN unused extern crate

fn main() {}
