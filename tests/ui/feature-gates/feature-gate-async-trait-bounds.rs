//@ edition: 2021

fn test(_: impl async Fn()) {}
//~^ ERROR `async` trait bounds are unstable

fn main() {}
