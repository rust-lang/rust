//@ aux-build:m1.rs
//@ aux-build:m2.rs
//@ reference: items.extern-crate.as


extern crate m1;
extern crate m2 as m1; //~ ERROR is defined multiple times

fn main() {}
