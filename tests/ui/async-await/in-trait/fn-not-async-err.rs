//@ edition: 2021

#![allow(incomplete_features)]

trait MyTrait {
    async fn foo(&self) -> i32;
}

impl MyTrait for i32 {
    fn foo(&self) -> i32 {
        //~^ ERROR: method should be `async` or return a future, but it is synchronous
        *self
    }
}

fn main() {}
