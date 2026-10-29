// Regression test for https://github.com/rust-lang/rust/issues/152887
//@ edition: 2024

#![feature(return_type_notation)]

pub trait Trait {
    async fn func();
}

impl<T: Trait<func(..): Send>> Trait for T {}
//~^ ERROR not all trait items implemented, missing: `func`

fn check(_: impl Trait) {}

fn main() {
    check(());
    //~^ ERROR overflow evaluating the requirement
}
