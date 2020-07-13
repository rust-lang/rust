// Ensure that we don't ICE if associated type impl trait is used in an impl
// with an unconstrained type parameter.

#![feature(type_alias_impl_trait)]

trait X {
    type I;
    fn f() -> Self::I;
}

impl<T> X for () {
    type I = impl Sized;
    //~^ ERROR could not find defining uses
    fn f() -> Self::I {}
    //~^ ERROR type annotations needed
}

fn main() {}
