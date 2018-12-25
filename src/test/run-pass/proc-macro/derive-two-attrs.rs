#![allow(dead_code)]
// aux-build:derive-two-attrs.rs

extern crate derive_two_attrs as foo;

use foo::A;

#[derive(A)]
#[b]
#[b]
struct B;

fn main() {}
