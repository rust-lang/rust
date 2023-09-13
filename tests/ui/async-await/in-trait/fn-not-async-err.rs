// edition: 2021

#![allow(incomplete_features)]

trait MyTrait {
    async fn foo(&self) -> i32;
}

impl MyTrait for i32 {
    fn foo(&self) -> i32 {
        //~^ ERROR: method `foo` should be async
        *self
    }
}

fn main() {}
