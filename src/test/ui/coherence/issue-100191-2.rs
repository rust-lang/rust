//~ ERROR overflow evaluating the requirement `T: Trait<_>`

#![feature(specialization, with_negative_coherence)]
#![allow(incomplete_features)]

pub trait Trait<T> {}

default impl<T, U> Trait<T> for U {}

impl<T> Trait<<T as Iterator>::Item> for T {}

fn main() {}
