//@ edition:2018

#![feature(async_trait_bounds)]

trait Foo {}

fn test(x: impl async Foo) {}
//~^ ERROR `async` bound modifier only allowed on `Fn`/`FnMut`/`FnOnce` traits

fn main() {}
