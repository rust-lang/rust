// edition: 2021
// revisions: cfg no

#![feature(async_fn_in_trait)]
//~^ WARN the feature `async_fn_in_trait` is incomplete

trait Trait {
    async fn m();
}

#[cfg(cfg)]
fn foo<T: Trait<m(..): Send>>() {}
//~^ ERROR return type notation is experimental

fn main() {}
