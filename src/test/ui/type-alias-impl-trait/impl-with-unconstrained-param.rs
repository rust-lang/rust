// Ensure that we don't ICE if associated type impl trait is used in an impl
// with an unconstrained type parameter.

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

trait X {
    type I;
    fn f() -> Self::I;
}

impl<T> X for () {
    //~^ ERROR the type parameter `T` is not constrained
    type I = impl Sized;
    fn f() -> Self::I {}
}

fn main() {}
