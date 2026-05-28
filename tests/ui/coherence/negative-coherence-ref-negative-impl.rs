//! Regression test for <https://github.com/rust-lang/rust/issues/112588>.

//@ check-pass

#![feature(negative_impls, with_negative_coherence)]

trait Trait {}
impl<T: ?Sized> !Trait for &T {}

trait OtherTrait<T> {}

impl<T: Trait> OtherTrait<T> for T {}
impl<T, U> OtherTrait<&U> for &T {}

fn main() {}
