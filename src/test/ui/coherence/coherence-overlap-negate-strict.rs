// check-pass

#![feature(negative_impls)]
#![feature(rustc_attrs)]
#![feature(trait_alias)]
#![feature(with_negative_coherence)]

trait A {}
trait B {}

impl !A for u32 {}

#[rustc_strict_coherence]
trait C {}
impl<T: A + B> C for T {}
impl C for u32 {}

fn main() {}
