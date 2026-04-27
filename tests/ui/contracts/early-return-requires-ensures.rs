//@ dont-require-annotations: NOTE
//@ compile-flags: -Zcontract-checks=yes
#![expect(incomplete_features)]
#![feature(contracts)]

extern crate core;
use core::contracts::{ensures, requires};

// Early return in requires takes precedence over ensures clause,
// but now we have two different closure types as candidates for the ensures
// closure, which is not allowed.
#[requires(return |_ret| { true }; true)]
#[ensures(|_ret| { false })]
//~^ ERROR: mismatched types [E0308]
//~| NOTE: no two closures, even if identical, have the same type
fn foo(x: u32) -> u32 {
    x * 2
}

fn main() {
    foo(1);
}
