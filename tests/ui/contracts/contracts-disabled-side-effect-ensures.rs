//@ run-pass
#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]

extern crate core;
use core::contracts::ensures;

#[ensures(*x = 0; |_ret| true)]
fn buggy_add(x: &mut u32, y: u32) {
    *x = *x + y;
}

fn main() {
    let mut x = 10;
    buggy_add(&mut x, 100);
    assert_eq!(x, 110);
}
