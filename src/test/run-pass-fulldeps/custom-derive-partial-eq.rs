// aux-build:custom-derive-partial-eq.rs
// ignore-stage1
#![feature(plugin)]
#![plugin(custom_derive_partial_eq)]
#![allow(unused)]

#[derive_CustomPartialEq] // Check that this is not a stability error.
enum E { V1, V2 }

fn main() {}
