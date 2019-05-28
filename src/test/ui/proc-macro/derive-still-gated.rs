// aux-build:test-macros.rs

#[macro_use]
extern crate test_macros;

#[derive_Empty] //~ ERROR attribute `derive_Empty` is currently unknown
struct A;

fn main() {}
