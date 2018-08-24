// aux-build:custom_derive_partial_eq.rs
// ignore-stage1
#![feature(plugin, custom_derive)]
#![plugin(custom_derive_partial_eq)]
#![allow(unused)]

#[derive(CustomPartialEq)] // Check that this is not a stability error.
enum E { V1, V2 }

fn main() {}
