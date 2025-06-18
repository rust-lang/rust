#![feature(unsize, dispatch_from_dyn)]
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

use std::marker::Unsize;
use std::ops::DispatchFromDyn;

#[allow(unused)]
struct Foo<'a, T> {
    _inner: &'a &'a T,
}

impl<'a, T: Unsize<U>, U> DispatchFromDyn<Foo<'a, U>> for Foo<'a, T> {}
//~^ ERROR the trait bound `&'a T: Unsize<&'a U>` is not satisfied
//~| NOTE the trait `Unsize<&'a U>` is not implemented for `&'a T`
//~| NOTE all implementations of `Unsize` are provided automatically by the compiler
//~| NOTE required for

fn main() {}
