#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(specialization)]

pub trait Trait {
    type Type;
}

impl<T: ?Sized> Trait for T {
    default type Type = [u8; 1];
}

impl<T: Trait> Trait for *const T {
    type Type = [u8; std::mem::size_of::<<T as Trait>::Type>()];
    //~^ ERROR: unconstrained generic constant
}

fn main() {}
