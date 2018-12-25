// aux-build:derive-a.rs

#![allow(warnings)]

#[macro_use]
extern crate derive_a;

#[derive_A] //~ ERROR attribute `derive_A` is currently unknown
struct A;

fn main() {}
