// edition:2018
// ignore-tidy-linelength
#![feature(async_await, existential_type)]

pub trait Bar {
    type E: Copy;

    fn foo<T>() -> Self::E;
}

impl<S> Bar for S {
    existential type E: Copy;
    //~^ ERROR the trait bound `impl std::future::Future: std::marker::Copy` is not satisfied [E0277]
    fn foo<T>() -> Self::E {
    //~^ ERROR type parameter `T` is part of concrete type but not used in parameter list for existential type
        async {}
    }
}

fn main() {}
