#![feature(unsize, dispatch_from_dyn)]

use std::marker::Unsize;
use std::ops::DispatchFromDyn;

#[allow(unused)]
struct Foo<'a, T: ?Sized> {
    _inner: &'a &'a T,
}

impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Foo<'a, U>> for Foo<'a, T> {}
//~^ ERROR trait `Unsize<&U>` is not implemented for `&T`
//~| NOTE the trait `Unsize<&U>` is not implemented for `&T`
//~| NOTE all implementations of `Unsize` are provided automatically by the compiler
//~| NOTE required for

fn main() {}
