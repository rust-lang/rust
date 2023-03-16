// edition:2021
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![allow(incomplete_features)]
#![feature(async_fn_in_trait)]

pub trait Foo {
    async fn woopsie_async(&self) -> String {
        42
        //~^ ERROR mismatched types
    }
}

fn main() {}
