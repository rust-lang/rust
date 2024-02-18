//! Regression test for revealing associated types through specialization during const eval.
//@ check-pass
#![feature(specialization)]
//~^ WARNING the feature `specialization` is incomplete and may not be safe to use

trait Foo {
    const ASSOC: usize;
}

impl Foo for u32 {
    default const ASSOC: usize = 0;
}

fn foo() -> [u8; 0] {
    [0; <u32 as Foo>::ASSOC]
}

fn main() {}
