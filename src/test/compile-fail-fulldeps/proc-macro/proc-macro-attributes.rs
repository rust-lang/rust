// aux-build:derive-b.rs
// ignore-stage1

#![allow(warnings)]

#[macro_use]
extern crate derive_b;

#[derive(B)]
#[B] //~ ERROR `B` is a derive mode
#[C]
#[B(D)]
#[B(E = "foo")]
#[B(arbitrary tokens)]
struct B;

fn main() {}
