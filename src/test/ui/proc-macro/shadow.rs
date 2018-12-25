// aux-build:derive-a.rs

#[macro_use]
extern crate derive_a;
#[macro_use]
extern crate derive_a; //~ ERROR the name `derive_a` is defined multiple times

fn main() {}
