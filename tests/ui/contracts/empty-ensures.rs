//@ compile-flags: -Zcontract-checks=yes
#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]

extern crate core;
use core::contracts::ensures;

#[ensures()]
//~^ ERROR expected a `Fn(&_)` closure, found `()` [E0277]
fn foo(x: u32) -> u32 {
    x * 2
}

fn main() {
    foo(1);
}
