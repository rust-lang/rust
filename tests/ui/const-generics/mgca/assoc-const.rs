//@ check-pass

#![feature(gca_min_const_items, gca_macroless_args)]
#![allow(incomplete_features)]

pub trait Tr<X> {
    #[rustc_always_gca]
    const SIZE: usize;
}

fn mk_array<T: Tr<bool>>(_x: T) -> [(); <T as Tr<bool>>::SIZE] {
    [(); T::SIZE]
}

fn main() {}
