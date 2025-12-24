//@ run-pass
//@ compile-flags: -Zcontract-checks=yes
#![expect(incomplete_features)]
#![feature(contracts)]

extern crate core;
use core::contracts::{ensures, requires};

// note that when we wrap requires in a block, the return is scoped just to
// requires, not the entire contract, making this early return valid
#[requires({if true { return true }; true})]
#[ensures(|_ret| { true })]
fn foo(x: u32) -> u32 {
    x * 2
}

fn main() {
    foo(1);
}
