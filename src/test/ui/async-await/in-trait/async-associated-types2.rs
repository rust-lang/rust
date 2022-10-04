// edition: 2021

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

use std::future::Future;

trait MyTrait {
    type Fut<'a>: Future<Output = i32>
    where
        Self: 'a;

    fn foo(&self) -> Self::Fut<'a>;
    //~^ ERROR use of undeclared lifetime name `'a`
}

impl MyTrait for i32 {
    type Fut<'a> = impl Future + 'a
    where
        Self: 'a;
    //~^^^ ERROR `impl Trait` in type aliases is unstable
    //~| ERROR expected `<i32 as MyTrait>::Fut<'a>` to be a future that resolves to `i32`, but it resolves to `<<i32 as MyTrait>::Fut<'a> as Future>::Output`

    fn foo<'a>(&'a self) -> Self::Fut<'a> {
        //~^ ERROR `impl` item signature doesn't match `trait` item signature
        async {
            *self
        }
    }
}

fn main() {}
