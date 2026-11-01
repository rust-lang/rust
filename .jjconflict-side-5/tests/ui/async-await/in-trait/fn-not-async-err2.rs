//@ edition: 2021
//@ check-pass

use std::future::Future;

trait MyTrait {
    async fn foo(&self) -> i32;
}

impl MyTrait for i32 {
    fn foo(&self) -> impl Future<Output = i32> {
        async { *self }
    }
}

fn main() {}
