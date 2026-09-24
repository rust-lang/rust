#![expect(incomplete_features)]
#![feature(min_generic_const_args, generic_const_items)]

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
    const N1<T>: usize = gca!(const { std::mem::size_of::<T>() });
    //~^ ERROR generic parameters may not be used in const operations
    const N2<const I: usize>: usize = gca!(const { I + 1 });
    //~^ ERROR generic parameters may not be used in const operations
    const N3: usize = gca!(const { 2 & X });
    //~^ ERROR generic parameters may not be used in const operations
}

fn main() {}
