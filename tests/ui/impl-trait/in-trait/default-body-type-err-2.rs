// edition:2021

#![allow(incomplete_features)]
#![feature(async_fn_in_trait)]

pub trait Foo {
    async fn woopsie_async(&self) -> String {
        42
        //~^ ERROR mismatched types
    }
}

fn main() {}
