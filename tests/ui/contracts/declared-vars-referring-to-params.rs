//@ run-pass
//@ compile-flags: -Zcontract-checks=yes
#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]

extern crate core;
use core::contracts::{ensures, requires};

// checks that variable declarations are lowered properly, with the ability to
// access function parameters
#[requires(let y = 2 * x; true)]
#[ensures(move |ret| { *ret == y })]
fn foo(x: u32) -> u32 {
    x * 2
}

fn main() {
    foo(1);
}
