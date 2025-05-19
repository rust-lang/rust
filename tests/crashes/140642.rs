//@ known-bug: #140642
#![feature(min_generic_const_args)]

pub trait Tr<A> {
    const SIZE: usize;
}

fn mk_array(_x: T) -> [(); <T as Tr<bool>>::SIZE] {}
