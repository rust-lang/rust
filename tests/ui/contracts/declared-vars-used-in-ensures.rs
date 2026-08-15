//@ run-pass
//@ compile-flags: -Zcontract-checks=yes
#![expect(incomplete_features)]
#![feature(contracts)]

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
