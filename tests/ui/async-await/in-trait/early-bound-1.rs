// check-pass
// edition:2021
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

pub trait Foo {
    async fn foo(&mut self);
}

struct MyFoo<'a>(&'a mut ());

impl<'a> Foo for MyFoo<'a> {
    async fn foo(&mut self) {}
}

fn main() {}
