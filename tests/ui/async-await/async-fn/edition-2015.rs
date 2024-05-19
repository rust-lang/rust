fn foo(x: impl async Fn()) -> impl async Fn() { x }
//~^ ERROR `async` trait bounds are only allowed in Rust 2018 or later
//~| ERROR `async` trait bounds are only allowed in Rust 2018 or later
//~| ERROR async closures are unstable
//~| ERROR async closures are unstable

fn main() {}
