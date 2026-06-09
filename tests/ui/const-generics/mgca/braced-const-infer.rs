//! Regression test for https://github.com/rust-lang/rust/issues/153198
#![feature(min_generic_const_args)]
#![allow(incomplete_features, rust_2021_compatibility)]

trait Trait<T> {}

impl dyn Trait<{_}> {} //~ ERROR: the placeholder `_` is not allowed within types on item signatures

fn main() {}
