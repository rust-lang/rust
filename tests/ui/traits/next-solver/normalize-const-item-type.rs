//@ compile-flags: -Znext-solver
#![feature(generic_const_items)]
#![feature(min_generic_const_args)]
#![feature(generic_const_args)]

use std::marker::PhantomData;

trait Project1<'a> {
    type Assoc1;
}

impl<'a, T> Project1<'a> for T {
    type Assoc1 = ();
}

trait Project2 {
    type Assoc2;
}

impl<T: Project1<'static, Assoc1 = ()>> Project2 for PhantomData<T> {
    type Assoc2 = usize;
}

const N<T>: <PhantomData::<T> as Project2>::Assoc2 = 2_usize;

fn func(_: [(); core::direct_const_arg!(N::<u32>)])
//~^ ERROR: type mismatch resolving `N<u32> == _` [E0271]
//~| ERROR: the type `[(); N::<u32>]` is not well-formed
//~| ERROR: type mismatch resolving `N<u32> == _` [E0271]
where
    for<'a> u32: Project1<'a>,
{}

fn main() {}
