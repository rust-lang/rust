#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use std::mem::size_of;
use std::marker::PhantomData;

struct Foo<T>(PhantomData<T>);

fn test<T>() -> [u8; size_of::<T>()] {
    [0; size_of::<Foo<T>>()]
    //~^ ERROR unconstrained generic constant
    //~| ERROR mismatched types
}

fn main() {
    test::<u32>();
}
