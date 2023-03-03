// edition:2021
// ignore-compare-mode-lower-impl-trait-in-trait-to-assoc-ty

#![allow(incomplete_features)]
#![feature(async_fn_in_trait)]

pub trait Foo {
    async fn woopsie_async(&self) -> String {
        42
        //~^ ERROR mismatched types
    }
}

fn main() {}
