// check-pass
// edition:2021
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(async_fn_in_trait, return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

pub trait Foo {
    async fn bar<'a: 'a>(&'a mut self);
}

impl Foo for () {
    async fn bar<'a: 'a>(&'a mut self) {}
}

pub trait Foo2 {
    fn bar<'a: 'a>(&'a mut self) -> impl Sized + 'a;
}

impl Foo2 for () {
    fn bar<'a: 'a>(&'a mut self) -> impl Sized + 'a {}
}

fn main() {}
