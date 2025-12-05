// Verify that we do not ICE when we try to overwrite an anon-const's type because of a trait
// cycle.
//
//@ compile-flags: -Zincremental-ignore-spans
//@ revisions: cpass cfail

#![feature(trait_alias)]
#![crate_type="lib"]

#[cfg(cpass)]
trait Bar<const N: usize> {}

#[cfg(cfail)]
trait Bar<const N: dyn BB> {}
//[cfail]~^ ERROR cycle detected when computing type of `Bar::N`
//[cfail]~| ERROR cycle detected when computing type of `Bar::N`
//[cfail]~| ERROR cycle detected when computing type of `Bar::N`
//[cfail]~| ERROR `(dyn Bar<{ 2 + 1 }> + 'static)` is forbidden as the type of a const generic parameter

trait BB = Bar<{ 2 + 1 }>;
