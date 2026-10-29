//@ dont-require-annotations: NOTE
//@ compile-flags: -Zcontract-checks=yes
#![expect(incomplete_features)]
#![feature(contracts)]

extern crate core;
use core::contracts::requires;

#[requires(let y = 1;)]
//~^ ERROR mismatched types [E0308]
//~| NOTE expected `bool`, found `()`
fn foo(x: u32) -> u32 {
    x * 2
}

fn main() {
    foo(1);
}
