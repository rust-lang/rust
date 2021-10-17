// edition:2018

#![feature(type_alias_impl_trait)]

use std::future::Future;

type G<'a, T> = impl Future<Output = ()>;
//~^ ERROR: type mismatch resolving `<impl Future as Future>::Output == ()`
//~| ERROR: the trait bound `T: Trait` is not satisfied

trait Trait {
    type F: Future<Output = ()>;

    fn f(&self) -> Self::F;

    fn g<'a>(&'a self) -> G<'a, Self>
    where
        Self: Sized,
    {
        async move { self.f().await }
    }
}

fn main() {}
