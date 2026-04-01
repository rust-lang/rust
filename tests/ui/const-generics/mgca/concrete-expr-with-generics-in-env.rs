//@ check-pass

#![expect(incomplete_features)]
#![feature(min_generic_const_args, generic_const_items)]

pub trait Tr<const X: usize> {
    type const N1<T>: usize;
    type const N2<const I: usize>: usize;
    type const N3: usize;
}

pub struct S;

impl<const X: usize> Tr<X> for S {
    type const N1<T>: usize = 0;
    type const N2<const I: usize>: usize = 1;
    type const N3: usize = 2;
}

fn main() {}
