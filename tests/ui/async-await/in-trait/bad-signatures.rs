// edition:2021
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(async_fn_in_trait)]
//~^ WARN the feature `async_fn_in_trait` is incomplete

trait MyTrait {
    async fn bar(&abc self);
    //~^ ERROR expected identifier, found keyword `self`
    //~| ERROR expected one of `:`, `@`, or `|`, found keyword `self`
}

impl MyTrait for () {
    async fn bar(&self) {}
}

fn main() {}
