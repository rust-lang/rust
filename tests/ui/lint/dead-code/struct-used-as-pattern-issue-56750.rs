//! Regression test for <https://github.com/rust-lang/rust/issues/56750>

//@ check-pass

#![deny(dead_code)]

use std::mem;

fn main() {
    struct Value {
        a: i32,
        b: i32,
    }

    let Value { a, b } = unsafe { mem::zeroed() };
    println!("{a} {b}");
}
