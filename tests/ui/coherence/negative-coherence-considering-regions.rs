// revisions: any_lt static_lt
//[static_lt] known-bug: unknown

// This fails because we currently perform negative coherence in coherence mode.
// This means that when looking for a negative predicate, we also assemble a
// coherence-unknowable predicate. Since confirming the negative impl has region
// obligations, we don't prefer the impl over the unknowable predicate
// unconditionally and instead flounder.

#![feature(negative_impls)]
#![feature(with_negative_coherence)]

trait Foo {}

impl<T> !Foo for &'static T {}

trait Bar {}

impl<T> Bar for T where T: Foo {}

#[cfg(any_lt)]
impl<T> Bar for &T {}
//[any_lt]~^ ERROR conflicting implementations of trait `Bar` for type `&_`

#[cfg(static_lt)]
impl<T> Bar for &'static T {}


fn main() {}
