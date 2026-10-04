//@ revisions: classic coherence next
//@[classic] compile-flags: -Znext-solver=no
//@[coherence] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally
//@ aux-build: coherence-future-impls.rs

extern crate coherence_future_impls as upstream;

// A known impl must still cause ordinary overlap.
trait Known {}
impl<T: upstream::KnownRestricted> Known for T {}
impl Known for u8 {}
//~^ ERROR conflicting implementations of trait `Known` for type `u8`

fn main() {}
