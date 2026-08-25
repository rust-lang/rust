//@ run-pass
//@ aux-build:cross-crate-type-alias-with-nested-destructors.rs

//! Regression test for https://github.com/rust-lang/rust/issues/2526

#![allow(unused_imports)]

extern crate cross_crate_type_alias_with_nested_destructors;
use cross_crate_type_alias_with_nested_destructors::*;

pub fn main() {}
