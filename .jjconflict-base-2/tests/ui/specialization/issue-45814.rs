//@ revisions: current negative
#![feature(specialization)]
#![cfg_attr(negative, feature(with_negative_coherence))]
#![allow(incomplete_features)]

pub trait Trait<T> {}

default impl<T, U> Trait<T> for U {}

impl<T> Trait<<T as Iterator>::Item> for T {}
//~^ ERROR conflicting implementations of trait `Trait<_>`

fn main() {}
