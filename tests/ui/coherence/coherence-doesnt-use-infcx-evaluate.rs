// check-pass
// issue: 113415

// Makes sure that coherence doesn't call any of the `predicate_may_hold`-esque fns,
// since they are using a different infcx which doesn't preserve the intercrate flag.

#![feature(specialization)]
//~^ WARN the feature `specialization` is incomplete

trait Assoc {
    type Output;
}

default impl<T> Assoc for T {
    type Output = bool;
}

impl Assoc for u8 {}

trait Foo {}
impl Foo for u32 {}
impl Foo for <u8 as Assoc>::Output {}

fn main() {}
