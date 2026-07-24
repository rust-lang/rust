//@ run-pass
//! Test using `#[arg_splat]` on tuple arguments of const functions.

#![allow(incomplete_features)]
#![feature(arg_splat)]

const fn sum(#[arg_splat] (a, b): (u32, u32)) -> u32 {
    a + b
}

const RESULT: u32 = sum(1, 2);

fn main() {
    assert_eq!(RESULT, 3);
    assert_eq!(sum(12, 18) , 30);
    assert_eq!(sum(1, 2), 3);
}
