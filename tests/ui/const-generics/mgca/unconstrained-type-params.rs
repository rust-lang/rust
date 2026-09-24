//! Regression test for https://github.com/rust-lang/rust/issues/146906

#![feature(min_generic_const_args)]

trait Trait {}

impl Trait for [(); N] {}
//~^ ERROR mismatched types

fn N(f: impl FnOnce(f64) -> f64 + Trait) {}

fn main() {}
