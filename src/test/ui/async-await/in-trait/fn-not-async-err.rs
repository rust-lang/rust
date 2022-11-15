// edition: 2021

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

trait MyTrait {
    async fn foo(&self) -> i32;
}

impl MyTrait for i32 {
    fn foo(&self) -> i32 {
        //~^ ERROR: `i32` is not a future [E0277]
        *self
    }
}

fn main() {}
