// edition: 2021
// check-pass

use std::future::Future;
use std::pin::Pin;

trait MyTrait {
    async fn foo(&self) -> i32;
}

impl MyTrait for i32 {
    fn foo(&self) -> Pin<Box<dyn Future<Output = i32> + '_>> {
        Box::pin(async { *self })
    }
}

fn main() {}
