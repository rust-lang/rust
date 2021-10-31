#![feature(rustc_attrs)]
#![feature(trait_alias)]

trait A {}
trait B {}
trait AB = A + B;

impl A for u32 {}
impl B for u32 {}

trait C {}
#[rustc_strict_coherence]
impl<T: AB> C for T {}
#[rustc_strict_coherence]
impl C for u32 {}
//~^ ERROR
// FIXME it's giving an ungreat error but unsure if we care given that it's using an internal rustc
// attribute and an artificial code path for testing purposes

fn main() {}
