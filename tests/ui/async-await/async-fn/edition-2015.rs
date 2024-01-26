// FIXME(async_closures): This error message could be made better.

fn foo(x: impl async Fn()) -> impl async Fn() {}
//~^ ERROR expected
//~| ERROR expected
//~| ERROR expected

fn main() {}
