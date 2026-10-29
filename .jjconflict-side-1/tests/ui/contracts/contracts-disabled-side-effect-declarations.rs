//@ run-pass
#![expect(incomplete_features)]
#![feature(contracts)]

extern crate core;
use core::contracts::requires;

#[requires(*x = 0; true)]
fn buggy_add(x: &mut u32, y: u32) {
    *x = *x + y;
}

fn main() {
    let mut x = 10;
    buggy_add(&mut x, 100);
    assert_eq!(x, 110);
}
