//! Regression test for https://github.com/rust-lang/rust/issues/3888

//@ check-pass
#![allow(dead_code)]

fn vec_peek<'r, T>(v: &'r [T]) -> &'r [T] {
    &v[1..5]
}

pub fn main() {}
