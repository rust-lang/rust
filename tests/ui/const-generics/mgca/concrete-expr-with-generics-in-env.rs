//@ check-pass

#![expect(incomplete_features)]
#![feature(gca_min_const_items, generic_const_items)]

use std::gca;

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
    const N1<T>: usize = gca!(0);
    const N2<const I: usize>: usize = gca!(1);
    const N3: usize = gca!(2);
}

fn main() {}
