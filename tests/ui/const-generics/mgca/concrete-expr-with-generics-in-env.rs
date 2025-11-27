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
    const N1<T>: usize = const { 0 };
    #[type_const]
    const N2<const I: usize>: usize = const { 1 };
    #[type_const]
    const N3: usize = const { 2 };
}

fn main() {}
