//@ known-bug: #110395
#![feature(derive_const)]

pub struct A;

impl std::fmt::Debug for A {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>)  -> Result<(), std::fmt::Error> {
        panic!()
    }
}

#[derive_const(Debug)]
pub struct S(A);

fn main() {}
