// aux-build:derive-unstable-2.rs
// ignore-stage1

#![allow(warnings)]

#[macro_use]
extern crate derive_unstable_2;

#[derive(Unstable)]
//~^ ERROR: reserved for internal compiler
struct A;

fn main() {
    foo();
}
