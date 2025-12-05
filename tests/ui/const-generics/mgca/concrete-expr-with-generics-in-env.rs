//@ check-pass

#![expect(incomplete_features)]
#![feature(min_generic_const_args, generic_const_items)]

pub trait Tr<const X: usize> {
    #[type_const]
    const N1<T>: usize;
    #[type_const]
    const N2<const I: usize>: usize;
    #[type_const]
    const N3: usize;
}

pub struct S;

impl<const X: usize> Tr<X> for S {
    #[type_const]
    const N1<T>: usize = 0;
    #[type_const]
    const N2<const I: usize>: usize = 1;
    #[type_const]
    const N3: usize = 2;
}

fn main() {}
