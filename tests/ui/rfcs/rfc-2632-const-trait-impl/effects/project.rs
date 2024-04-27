//@ known-bug: #110395
// FIXME: effects

#![feature(const_trait_impl, effects)]

// This fails because `~const Uwu` doesn't imply (non-const) `Uwu`.

// FIXME: #[const_trait]
pub trait Owo<X = <Self as /* FIXME: ~const */ Uwu>::T> {}

#[const_trait]
pub trait Uwu: Owo {
    type T;
}

fn main() {}
