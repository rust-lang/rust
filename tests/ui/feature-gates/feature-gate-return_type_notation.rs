// edition: 2021

#![feature(async_fn_in_trait)]
//~^ WARN the feature `async_fn_in_trait` is incomplete

trait Trait {
    async fn m();
}

fn foo<T: Trait<m(): Send>>() {}
//~^ ERROR parenthesized generic arguments cannot be used in associated type constraints
//~| ERROR associated type `m` not found for `Trait`
//~| ERROR return type notation is unstable

fn main() {}
