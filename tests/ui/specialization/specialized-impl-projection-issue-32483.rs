//@ check-pass

#![allow(incomplete_features)]
#![feature(specialization)]

// Tests that we allow some projections in specialized impls.
// Regression test for issue #32483.

pub trait Foo {
    type TypeA;
    type TypeB: Bar<Self::TypeA>;
}

pub trait Bar<T> {
}

pub struct ImplsBar;
impl<T> Bar<T> for ImplsBar {
}

impl<T> Foo for T {
    type TypeA = u8;
    // WF checking `TypeB` here requires us to project `Self::TypeA`
    default type TypeB = ImplsBar;
}

fn main() {}
