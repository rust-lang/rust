//@ edition:2018

#![feature(async_closure)]

trait Foo {}

fn test(x: impl async Foo) {}
//~^ ERROR `async` bound modifier only allowed on `Fn`/`FnMut`/`FnOnce` traits

fn main() {}
