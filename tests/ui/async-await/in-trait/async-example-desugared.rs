//@ check-pass
//@ edition: 2021

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
