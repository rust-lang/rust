//@ run-pass
//@ compile-flags: -Zcontract-checks=yes
#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]

extern crate core;
use core::contracts::{ensures, requires};

#[requires(let y = 1; true)]
#[ensures(move |_ret| { y == 1 })]
fn foo(x: u32) -> u32 {
    x * 2
}

fn main() {
    foo(1);
}
