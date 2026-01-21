//@ compile-flags: -Zcontract-checks=yes
#![expect(incomplete_features)]
#![feature(contracts)]

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
