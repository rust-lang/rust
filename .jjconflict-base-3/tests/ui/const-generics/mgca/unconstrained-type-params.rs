//! Regression test for https://github.com/rust-lang/rust/issues/146906

#![feature(gca_min_const_items)]

trait Trait {}

impl Trait for [(); N] {}
//~^ ERROR mismatched types

fn N(f: impl FnOnce(f64) -> f64 + Trait) {}

fn main() {}
