// known-bug: unknown

// This fails because we currently perform negative coherence in coherence mode.
// This means that when looking for a negative predicate, we also assemble a
// coherence-unknowable predicate. Since confirming the negative impl has region
// obligations, we don't prefer the impl over the unknowable predicate
// unconditionally and instead flounder.

#![feature(negative_impls)]
#![feature(rustc_attrs)]
#![feature(with_negative_coherence)]

#[rustc_strict_coherence]
trait Foo {}
impl<T> !Foo for &T where T: 'static {}

#[rustc_strict_coherence]
trait Bar {}
impl<T: Foo> Bar for T {}
impl<T> Bar for &T where T: 'static {}

fn main() {}
