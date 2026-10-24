//! Regression test for <https://github.com/rust-lang/rust/issues/133797>
//@ check-pass

#![deny(dead_code)]

const fn test() -> usize {
    0
}

trait Test {}

impl Test for [u8; test()] {}

fn test2<T: Test>() {}

fn main() {
    test2::<[u8; 0]>();
}
