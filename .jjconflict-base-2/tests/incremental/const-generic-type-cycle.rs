// Verify that we do not ICE when we try to overwrite an anon-const's type because of a trait
// cycle.
//
//@ compile-flags: -Zincremental-ignore-spans
//@ revisions: bpass bfail

#![feature(trait_alias)]
#![crate_type="lib"]

#[cfg(bpass)]
trait Bar<const N: usize> {}

#[cfg(bfail)]
trait Bar<const N: dyn BB> {}
//[bfail]~^ ERROR cycle detected when computing type of `Bar::N`

trait BB = Bar<{ 2 + 1 }>;
