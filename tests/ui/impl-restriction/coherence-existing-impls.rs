//@ revisions: classic next
//@[classic] compile-flags: -Znext-solver=no
//@[next] compile-flags: -Znext-solver=globally
//@ aux-build: coherence-impl-restriction.rs

// Existing impls cause overlap for both kinds of trait.

extern crate coherence_impl_restriction as upstream;

trait KnownRestricted {}
impl<T: upstream::KnownRestricted> KnownRestricted for T {}
impl KnownRestricted for u8 {}
//~^ ERROR conflicting implementations of trait `KnownRestricted` for type `u8`

trait KnownUnrestricted {}
impl<T: upstream::Unrestricted> KnownUnrestricted for T {}
impl KnownUnrestricted for u8 {}
//~^ ERROR conflicting implementations of trait `KnownUnrestricted` for type `u8`

fn main() {}
