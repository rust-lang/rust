#![feature(negative_impls)]
#![feature(rustc_attrs)]
#![feature(trait_alias)]

trait A {}
trait B {}
trait AB = A + B;

impl !A for u32 {}

trait C {}
#[rustc_strict_coherence]
impl<T: AB> C for T {}
#[rustc_strict_coherence]
impl C for u32 {}
//~^ ERROR: conflicting implementations of trait `C` for type `u32` [E0119]
// FIXME this should work, we should implement an `assemble_neg_candidates` fn

fn main() {}
