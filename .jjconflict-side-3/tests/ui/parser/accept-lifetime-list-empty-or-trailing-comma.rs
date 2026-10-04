//! Regression test for <https://github.com/rust-lang/rust/issues/37733>
//@ build-pass
#![allow(dead_code)]
type A = for<> fn();

type B = for<'a,> fn();

pub fn main() {}
