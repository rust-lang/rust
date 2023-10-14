// edition:2021

#![allow(incomplete_features)]

pub trait Foo {
    async fn woopsie_async(&self) -> String {
        42
        //~^ ERROR mismatched types
    }
}

fn main() {}
