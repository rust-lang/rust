// Regression test for #152752.

#![allow(dead_code)]

use std::marker::PhantomData;

trait Trait {}

struct Number<const N: usize> {}

impl Trait for Number<1> {}

fn func<const N: usize>(_: Number<N>) where Number<N>: Trait {}

fn test_const<const N: usize>() {
    func(Number::<N> {})
    //~^ ERROR the trait bound `Number<N>: Trait` is not satisfied
}

fn infers_const_from_argument() {
    func(Number::<1> {});
}

struct Wrap<T>(PhantomData<T>);

impl Trait for Wrap<u8> {}

fn generic<T>(_: Wrap<T>) where Wrap<T>: Trait {}

fn test_type<T>(x: Wrap<T>) {
    generic(x)
    //~^ ERROR the trait bound `Wrap<T>: Trait` is not satisfied
}

fn test_type_ctor<T>() {
    generic(Wrap::<T>(PhantomData))
    //~^ ERROR the trait bound `Wrap<T>: Trait` is not satisfied
}

fn infers_type_from_argument() {
    generic(Wrap::<u8>(PhantomData));
}

fn main() {}
