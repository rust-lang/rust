//@ check-pass

#![feature(generic_const_items, min_generic_const_args)]
#![allow(incomplete_features)]

const CT<T: ?Sized>: usize = core::direct_const_arg!(<T as Trait>::N);

trait Trait {
    #[rustc_always_gca]
    const N: usize;
}

impl<T: ?Sized> Trait for T {
    const N: usize = core::direct_const_arg!(0);
}

fn f(_x: [(); CT::<()>]) {}

fn main() {}
