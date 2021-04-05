// edition:2018
// ignore-compare-mode-chalk

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

pub trait Bar {
    type E: Copy;

    fn foo<T>() -> Self::E;
}

impl<S> Bar for S {
    type E = impl std::marker::Copy;
    //~^ ERROR the trait bound `impl Future: Copy` is not satisfied [E0277]
    fn foo<T>() -> Self::E {
        //~^ ERROR type parameter `T` is part of concrete type but not used in parameter list for the `impl Trait` type alias
        async {}
    }
}

fn main() {}
