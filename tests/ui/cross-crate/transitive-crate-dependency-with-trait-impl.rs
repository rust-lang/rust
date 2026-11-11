//@ run-pass
//@ aux-build:transitive-crate-dependency-with-trait-impl-a.rs
//@ aux-build:transitive-crate-dependency-with-trait-impl-b.rs

//! Regression test for https://github.com/rust-lang/rust/issues/2414

extern crate b;

pub fn main() {}
