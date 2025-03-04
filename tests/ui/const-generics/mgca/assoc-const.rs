//@ check-pass

#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

pub trait Tr<X> {
    #[type_const]
    const SIZE: usize;
}

fn mk_array<T: Tr<bool>>(_x: T) -> [(); <T as Tr<bool>>::SIZE] {
    [(); T::SIZE]
}

fn main() {}
