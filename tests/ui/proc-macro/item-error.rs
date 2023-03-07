// aux-build:derive-b.rs

#![allow(warnings)]

#[macro_use]
extern crate derive_b;

#[derive(B)]
struct A {
    a: &u64
//~^ ERROR: missing lifetime specifier
}

fn main() {
}
