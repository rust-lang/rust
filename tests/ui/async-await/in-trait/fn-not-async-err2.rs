// edition: 2021
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

use std::future::Future;

trait MyTrait {
    async fn foo(&self) -> i32;
}

impl MyTrait for i32 {
    fn foo(&self) -> impl Future<Output = i32> {
        //~^ ERROR `impl Trait` only allowed in function and inherent method return types, not in `impl` method return types
        async { *self }
    }
}

fn main() {}
