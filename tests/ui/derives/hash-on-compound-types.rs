//! Regression test for <https://github.com/rust-lang/rust/issues/21402>.

//@ check-pass
#![allow(dead_code)]

#[derive(Hash)]
struct Foo {
    a: Vec<bool>,
    b: (bool, bool),
    c: [bool; 2],
}

fn main() {}
