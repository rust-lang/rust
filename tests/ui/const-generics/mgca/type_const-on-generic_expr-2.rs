#![expect(incomplete_features)]
#![feature(min_generic_const_args, generic_const_items)]

pub trait Tr<const X: usize> {
    type const N1<T>: usize;
    type const N2<const I: usize>: usize;
    type const N3: usize;
}

pub struct S;

impl<const X: usize> Tr<X> for S {
    type const N1<T>: usize = const { std::mem::size_of::<T>() };
    //~^ ERROR generic parameters may not be used in const operations
    type const N2<const I: usize>: usize = const { I + 1 };
    //~^ ERROR generic parameters may not be used in const operations
    type const N3: usize = const { 2 & X };
    //~^ ERROR generic parameters may not be used in const operations
}

fn main() {}
