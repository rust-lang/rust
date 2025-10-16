//@ dont-require-annotations: NOTE
//@ compile-flags: -Zcontract-checks=yes
#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]

extern crate core;
use core::contracts::requires;

#[requires()]
//~^ ERROR mismatched types [E0308]
//~| NOTE expected `bool`, found `()`
fn foo(x: u32) -> u32 {
    x * 2
}

fn main() {
    foo(1);
}
