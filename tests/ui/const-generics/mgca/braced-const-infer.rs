//! Regression test for https://github.com/rust-lang/rust/issues/153198
#![feature(min_generic_const_args, macroless_generic_const_args)]
#![allow(incomplete_features, rust_2021_compatibility)]

trait Trait<T> {}

impl dyn Trait<{_}> {} //~ ERROR: constant provided when a type was expected

fn main() {}
