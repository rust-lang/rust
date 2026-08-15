//@ run-pass

#![allow(unused_imports)]
//@ proc-macro: empty-crate.rs

#[macro_use]
extern crate empty_crate;

fn main() {}
