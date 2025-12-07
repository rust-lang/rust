//@ run-pass
//@ compile-flags: -Zcontract-checks=yes
#![expect(incomplete_features)]
#![feature(contracts)]

extern crate core;
use core::contracts::{ensures, requires};

// checks that variable declarations are lowered properly, with the ability to
// refer to them *both* in requires and ensures
#[requires(let y = 2 * x; y > 0)]
#[ensures(move |ret| { *ret == y })]
fn foo(x: u32) -> u32 {
    x * 2
}

fn main() {
    foo(1);
}
