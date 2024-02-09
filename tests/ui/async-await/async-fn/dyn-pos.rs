// edition:2018

#![feature(async_closure)]

fn foo(x: &dyn async Fn()) {}
//~^ ERROR the trait `AsyncFn` cannot be made into an object
//~| ERROR the trait `AsyncFn` cannot be made into an object
//~| ERROR the trait `AsyncFn` cannot be made into an object
//~| ERROR the trait `AsyncFn` cannot be made into an object
//~| ERROR the trait `AsyncFnMut` cannot be made into an object
//~| ERROR the trait `AsyncFnMut` cannot be made into an object
//~| ERROR the trait `AsyncFnMut` cannot be made into an object

fn main() {}
