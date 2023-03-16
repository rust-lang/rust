// edition: 2021
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

trait Foo {
    async fn foo<T>();
}

impl Foo for () {
    async fn foo<const N: usize>() {}
    //~^ ERROR: method `foo` has an incompatible generic parameter for trait `Foo` [E0053]
}

fn main() {}
