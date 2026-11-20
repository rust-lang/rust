//@ run-pass
#![allow(dead_code)]
//@ aux-build:generic-fn-with-supertrait-bound-cross-crate.rs

//! Regression test for https://github.com/rust-lang/rust/issues/4208
extern crate generic_fn_with_supertrait_bound_cross_crate as numeric;
use numeric::{sin, Angle};

fn foo<T, A:Angle<T>>(theta: A) -> T { sin(&theta) }

pub fn main() {}
