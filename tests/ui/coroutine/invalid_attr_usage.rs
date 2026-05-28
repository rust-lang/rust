//! The `coroutine` attribute is only allowed on closures.

#![feature(coroutines)]

#[coroutine]
//~^ ERROR: attribute cannot be used on
struct Foo;

#[coroutine]
//~^ ERROR: attribute cannot be used on
fn main() {}
