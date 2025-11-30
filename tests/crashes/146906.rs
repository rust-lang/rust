//@ known-bug: rust-lang/rust#146906
#![feature(min_generic_const_args)]
trait Trait {}
impl Trait for [(); N] {}

fn N(f: impl FnOnce(f64) -> f64 + Trait) {}
pub fn main() {}
