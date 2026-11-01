//@ compile-flags: -Zcontract-checks=yes
#![expect(incomplete_features)]
#![feature(contracts)]

extern crate core;
use core::contracts::requires;

#[requires()]
//~^ ERROR `requires` attribute requires an argument
fn foo(x: u32) -> u32 {
    x * 2
}

fn main() {
    foo(1);
}
