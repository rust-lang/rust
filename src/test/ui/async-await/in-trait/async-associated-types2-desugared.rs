// edition: 2021

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

use std::future::Future;

trait MyTrait {
    type Fut<'a>: Future<Output = i32>
    where
        Self: 'a;

    async fn foo(&self) -> Self::Fut<'a>;
    //~^ ERROR use of undeclared lifetime name `'a`
    //~| ERROR the parameter type `Self` may not live long enough
}

impl MyTrait for i32 {
    type Fut<'a> = impl Future + 'a
    where
        Self: 'a;
    //~^^^ ERROR `impl Trait` in type aliases is unstable

    fn foo<'a>(&'a self) -> Self::Fut<'a> {
        async {
            *self
        }
    }
}

fn main() {}
