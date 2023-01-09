#![allow(unused)]
#![warn(clippy::impl_trait_param)]

pub trait Trait {}

// Should warn
pub fn a(_: impl Trait) {}
pub fn c<C: Trait>(_: C, _: impl Trait) {}

// Shouldn't warn

pub fn b<B: Trait>(_: B) {}
fn d<D: Trait>(_: D, _: impl Trait) {}

fn main() {}
