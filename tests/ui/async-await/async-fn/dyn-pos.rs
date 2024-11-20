//@ edition:2018

#![feature(async_closure)]

fn foo(x: &dyn async Fn()) {}
//~^ ERROR the trait `AsyncFn` is not yet dyn-compatible

fn main() {}
