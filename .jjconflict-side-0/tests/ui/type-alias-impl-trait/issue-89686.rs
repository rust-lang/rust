//@ edition:2018

#![feature(type_alias_impl_trait)]

use std::future::Future;

type G<'a, T> = impl Future<Output = ()>;

trait Trait {
    type F: Future<Output = ()>;

    fn f(&self) -> Self::F;

    #[define_opaque(G)]
    fn g<'a>(&'a self) -> G<'a, Self>
    where
        Self: Sized,
    {
        async move { self.f().await }
        //~^ ERROR: the trait bound `T: Trait` is not satisfied
    }
}

fn main() {}
