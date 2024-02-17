//@ aux-build:test-macros.rs

#![dummy] //~ ERROR cannot find attribute `dummy` in this scope

#[macro_use]
extern crate test_macros;

#[derive(Empty)]
#[empty_helper] //~ ERROR cannot find attribute `empty_helper` in this scope
struct Foo {}

fn main() {}
