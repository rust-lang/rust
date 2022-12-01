// edition: 2021

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

use std::future::Future;

trait MyTrait {
    async fn foo(&self) -> i32;
}

impl MyTrait for i32 {
    fn foo(&self) -> impl Future<Output = i32> {
        //~^ ERROR `impl Trait` isn't allowed within `impl` method return [E0562]
        async {
            *self
        }
    }
}

fn main() {}
