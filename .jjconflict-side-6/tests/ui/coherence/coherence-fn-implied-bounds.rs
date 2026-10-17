// Test that our leak-check is not smart enough to take implied bounds
// into account (yet). Here we have two types that look like they
// should not be equivalent, but because of the rules on implied
// bounds we ought to know that, in fact, `'a = 'b` must always hold,
// and hence they are.
//
// Rustc can't figure this out and hence it accepts the impls but
// gives a future-compatibility warning (because we'd like to make
// this an error someday).
//
// Note that while we would like to make this a hard error, we also
// give the same warning for `coherence-wasm-bindgen.rs`, which ought
// to be accepted.

#![deny(coherence_leak_check)]

trait Trait {}

impl Trait for for<'a, 'b> fn(&'a &'b u32, &'b &'a u32) -> &'b u32 {}

impl Trait for for<'c> fn(&'c &'c u32, &'c &'c u32) -> &'c u32 {
    //~^ ERROR conflicting implementations
    //~| WARN the behavior may change in a future release
}

fn main() {}
