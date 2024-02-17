//@ check-pass

#![feature(negative_impls)]
#![feature(with_negative_coherence)]

trait A {}
trait B: A {}

impl !A for u32 {}
impl !B for u32 {}

fn main() {}
