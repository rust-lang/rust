// check-pass
// edition: 2021

#![feature(async_fn_in_trait)]
#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

use std::future::Future;

trait MyTrait {
    #[allow(async_fn_in_trait)]
    async fn foo(&self) -> i32;
}

impl MyTrait for i32 {
    fn foo(&self) -> impl Future<Output = i32> {
        async { *self }
    }
}

fn main() {}
