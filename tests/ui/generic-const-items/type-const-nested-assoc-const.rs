//@ check-pass

#![feature(generic_const_items, min_generic_const_args)]
#![allow(incomplete_features)]

type const CT<T: ?Sized>: usize = { <T as Trait>::N };

trait Trait {
    type const N: usize;
}

impl<T: ?Sized> Trait for T {
    type const N:usize = 0;
}

fn f(_x: [(); CT::<()>]) {}

fn main() {}
