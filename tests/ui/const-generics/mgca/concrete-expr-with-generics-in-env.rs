//@ check-pass

#![expect(incomplete_features)]
#![feature(min_generic_const_args, generic_const_items)]

extern crate core;
use core::direct_const_arg;

pub trait Tr<const X: usize> {
    #[rustc_always_gca]
    const N1<T>: usize;
    #[rustc_always_gca]
    const N2<const I: usize>: usize;
    #[rustc_always_gca]
    const N3: usize;
}

pub struct S;

impl<const X: usize> Tr<X> for S {
    const N1<T>: usize = core::direct_const_arg!(0);
    const N2<const I: usize>: usize = core::direct_const_arg!(1);
    const N3: usize = core::direct_const_arg!(2);
}

fn main() {}
