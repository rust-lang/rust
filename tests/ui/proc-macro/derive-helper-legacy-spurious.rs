//@ proc-macro: test-macros.rs

#![dummy] //~ ERROR cannot find attribute `dummy` in this scope

#[macro_use]
extern crate test_macros;

#[derive(Empty)]
#[empty_helper]
struct Foo {}

fn main() {}
