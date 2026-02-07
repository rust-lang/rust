//@ check-pass

#![feature(generic_const_items, min_generic_const_args)]
#![allow(incomplete_features)]

#[type_const]
const CT<T: ?Sized>: usize = { <T as Trait>::N };

trait Trait {
    #[type_const]
    const N: usize;
}

impl<T: ?Sized> Trait for T {
    #[type_const]
    const N: usize = 0;
}

fn f(_x: [(); CT::<()>]) {}

fn main() {}
