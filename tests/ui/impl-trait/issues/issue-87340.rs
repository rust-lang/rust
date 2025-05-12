#![feature(impl_trait_in_assoc_type)]

trait X {
    type I;
    fn f() -> Self::I;
}

impl<T> X for () {
    //~^ ERROR `T` is not constrained by the impl trait, self type, or predicates
    type I = impl Sized;
    fn f() -> Self::I {}
}

fn main() {}
