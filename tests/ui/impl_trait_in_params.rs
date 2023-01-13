#![allow(unused)]
#![warn(clippy::impl_trait_in_params)]

pub trait Trait {}
pub trait AnotherTrait<T> {}

// Should warn
pub fn a(_: impl Trait) {}
pub fn c<C: Trait>(_: C, _: impl Trait) {}
fn d(_: impl AnotherTrait<u32>) {}

// Shouldn't warn

pub fn b<B: Trait>(_: B) {}
fn e<T: AnotherTrait<u32>>(_: T) {}

fn main() {}
