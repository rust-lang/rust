// aux-build:derive-a.rs

#![allow(warnings)]

#[macro_use]
extern crate derive_a;

#[derive_A] //~ ERROR: attributes of the form `#[derive_*]` are reserved for the compiler
struct A;

fn main() {}
