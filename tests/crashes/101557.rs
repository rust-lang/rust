//@ known-bug: #101557
//@ compile-flags: -Copt-level=0
#![feature(generic_const_exprs)]
use std::marker::PhantomData;

trait Trait {
    const CONST: usize;
}

struct A<T: Trait> {
    _marker: PhantomData<T>,
}

impl<const N: usize> Trait for [i8; N] {
    const CONST: usize = N;
}

impl<const N: usize> From<usize> for A<[i8; N]> {
    fn from(_: usize) -> Self {
        todo!()
    }
}

impl<T: Trait> From<A<[i8; T::CONST]>> for A<T> {
    fn from(_: A<[i8; T::CONST]>) -> Self {
        todo!()
    }
}

fn f<T: Trait>() -> A<T>
where
    [(); T::CONST]:,
{
    // Usage of `0` is arbitrary
    let a = A::<[i8; T::CONST]>::from(0);
    A::<T>::from(a)
}

fn main() {
    // Usage of `1` is arbitrary
    f::<[i8; 1]>();
}
