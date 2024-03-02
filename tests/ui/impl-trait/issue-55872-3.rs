//@ edition:2018

#![feature(impl_trait_in_assoc_type)]

pub trait Bar {
    type E: Copy;

    fn foo<T>() -> Self::E;
}

impl<S> Bar for S {
    type E = impl std::marker::Copy;
    fn foo<T>() -> Self::E {
        //~^ ERROR the trait `Copy` is not implemented for
        async {}
    }
}

fn main() {}
