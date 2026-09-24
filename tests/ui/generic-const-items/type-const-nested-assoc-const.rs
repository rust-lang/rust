//@ check-pass

#![feature(generic_const_items, min_generic_const_args)]
#![allow(incomplete_features)]

use std::gca;

const CT<T: ?Sized>: usize = gca!(<T as Trait>::N);

trait Trait {
    #[rustc_always_gca]
    const N: usize;
}

impl<T: ?Sized> Trait for T {
    const N: usize = gca!(0);
}

fn f(_x: [(); CT::<()>]) {}

fn main() {}
