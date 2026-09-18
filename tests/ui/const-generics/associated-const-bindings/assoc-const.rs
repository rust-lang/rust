//@ run-pass
#![feature(min_generic_const_args)]
#![allow(unused, incomplete_features)]

pub trait Foo {
    #[rustc_always_gca]
    const N: usize;
}

pub struct Bar;

impl Foo for Bar {
    const N: usize = core::direct_const_arg!(3);
}

fn foo<F: Foo<N = 3usize>>() {}

fn main() {
    foo::<Bar>()
}
