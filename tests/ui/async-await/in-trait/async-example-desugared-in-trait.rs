//@ check-pass
//@ edition: 2021

use std::future::Future;

trait MyTrait {
    fn foo(&self) -> impl Future<Output = i32> + '_;
}

impl MyTrait for i32 {
    // This will break once a PR that implements #102745 is merged
    async fn foo(&self) -> i32 {
        *self
    }
}

fn main() {}
