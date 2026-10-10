//@ edition:2018

#![feature(impl_trait_in_assoc_type)]

pub trait Bar {
    type E: Send;

    fn foo<T>() -> Self::E;
}

impl<S> Bar for S {
    type E = impl std::marker::Send;
    fn foo<T>() -> Self::E {
        //~^ ERROR type parameter `T` is part of concrete type but not used in parameter list for the `impl Trait` type alias
        async {}
    }
}

fn main() {}
