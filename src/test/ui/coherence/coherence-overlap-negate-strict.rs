// check-pass

#![feature(negative_impls)]
#![feature(rustc_attrs)]
#![feature(trait_alias)]

trait A {}
trait B {}

impl !A for u32 {}

trait C {}
#[rustc_strict_coherence]
impl<T: A + B> C for T {}
#[rustc_strict_coherence]
impl C for u32 {}

fn main() {}
