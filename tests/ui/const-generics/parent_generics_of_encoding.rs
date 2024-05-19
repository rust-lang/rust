//@ aux-build:generics_of_parent.rs
//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

extern crate generics_of_parent;

use generics_of_parent::{Foo, S};

fn main() {
    // regression test for #87603
    const N: usize = 2;
    let x: S<u8, N> = S::test();
}

// regression test for #87674
fn new<U>(a: U) -> U {
    a
}
fn foo<const N: usize>(bar: &mut Foo<N>)
where
    [(); N + 1]: ,
{
    *bar = new(loop {});
}
