// edition: 2021

use std::future::Future;
use std::pin::Pin;

trait MyTrait {
    async fn foo(&self) -> i32;
}

impl MyTrait for i32 {
    fn foo(&self) -> Pin<Box<dyn Future<Output = i32> + '_>> {
        //~^ ERROR method `foo` should be async
        Box::pin(async { *self })
    }
}

fn main() {}
