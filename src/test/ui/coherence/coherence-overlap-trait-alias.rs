#![feature(rustc_attrs)]
#![feature(trait_alias)]
#![feature(with_negative_coherence)]

trait A {}
trait B {}
trait AB = A + B;

impl A for u32 {}
impl B for u32 {}

#[rustc_strict_coherence]
trait C {}
impl<T: AB> C for T {}
impl C for u32 {}
//~^ ERROR
// FIXME it's giving an ungreat error but unsure if we care given that it's using an internal rustc
// attribute and an artificial code path for testing purposes

fn main() {}
