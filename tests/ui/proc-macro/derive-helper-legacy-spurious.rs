//@ aux-build:test-macros.rs

#![dummy] //~ ERROR cannot find attribute `dummy`

#[macro_use]
extern crate test_macros;

#[derive(Empty)]
#[empty_helper] //~ ERROR cannot find attribute `empty_helper`
struct Foo {}

fn main() {}
