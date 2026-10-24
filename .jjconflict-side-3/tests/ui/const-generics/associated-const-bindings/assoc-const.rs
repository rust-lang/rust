//@ run-pass
#![feature(gca_min_const_items)]
#![allow(unused, incomplete_features)]

use std::gca;

pub trait Foo {
    #[rustc_always_gca]
    const N: usize;
}

pub struct Bar;

impl Foo for Bar {
    const N: usize = gca!(3);
}

fn foo<F: Foo<N = 3usize>>() {}

fn main() {
    foo::<Bar>()
}
