//@ known-bug: #110395
#![feature(derive_const)]

pub struct A;

impl Default for A {
    fn default() -> A { A }
}

#[derive_const(Default)]
pub struct S(A);

fn main() {}
