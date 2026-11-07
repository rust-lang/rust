//@ edition: 2021

use std::future::Future;
use std::pin::Pin;

trait MyTrait {
    fn foo(&self) -> Pin<Box<dyn Future<Output = i32> + '_>>;
}

impl MyTrait for i32 {
    async fn foo(&self) -> i32 {
        //~^ ERROR method `foo` has an incompatible type for trait
        *self
    }
}

fn main() {}
