//@ edition:2018

#![feature(type_alias_impl_trait)]

use std::future::Future;

type G<'a, T> = impl Future<Output = ()>;

trait Trait {
    type F: Future<Output = ()>;

    fn f(&self) -> Self::F;

    fn g<'a>(&'a self) -> G<'a, Self>
    where
        Self: Sized,
    {
        async move { self.f().await }
        //~^ ERROR trait `Trait` is not implemented for `T`
    }
}

fn main() {}
