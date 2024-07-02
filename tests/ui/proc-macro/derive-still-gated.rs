//@ aux-build:test-macros.rs

#[macro_use]
extern crate test_macros;

#[derive_Empty] //~ ERROR cannot find attribute `derive_Empty`
struct A;

fn main() {}
