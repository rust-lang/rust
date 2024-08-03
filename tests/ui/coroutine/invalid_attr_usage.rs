//! The `coroutine` attribute is only allowed on closures.

#![feature(coroutines)]

#[coroutine]
//~^ ERROR: attribute should be applied to closures
struct Foo;

#[coroutine]
//~^ ERROR: attribute should be applied to closures
fn main() {}
