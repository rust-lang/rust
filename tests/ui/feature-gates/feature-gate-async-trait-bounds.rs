//@ edition: 2021

fn test(_: impl async Fn()) {}
//~^ ERROR `async` trait bounds are unstable
//~| ERROR use of unstable library feature `async_closure`

fn main() {}
