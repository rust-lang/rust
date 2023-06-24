// edition: 2021
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(return_type_notation, async_fn_in_trait)]
//~^ WARN the feature `return_type_notation` is incomplete

use std::future::Future;

trait Trait {
    async fn method() {}
}

fn test<T: Trait<method() = Box<dyn Future<Output = ()>>>>() {}
//~^ ERROR return type notation is not allowed to use type equality

fn main() {}
