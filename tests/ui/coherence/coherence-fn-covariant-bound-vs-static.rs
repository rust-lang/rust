//@ check-pass

// These types were previously considered equal as they are subtypes of each other.
// This has been changed in #118247 and we now consider them to be disjoint.
//
// In our test:
//
// * `for<'r> fn(fn(&'r u32))`
// * `fn(fn(&'a u32)` where `'a` is free
//
// These were considered equal as for `'a = 'static` subtyping succeeds in both
// directions:
//
// * `for<'r> fn(fn(&'r u32)) <: fn(fn(&'static u32))`
//   * true if `exists<'r> { 'r: 'static }` (obviously true)
// * `fn(fn(&'static u32)) <: for<'r> fn(fn(&'r u32))`
//   * true if `forall<'r> { 'static: 'r }` (also true)

trait Trait {}

impl Trait for for<'r> fn(fn(&'r ())) {}
impl<'a> Trait for fn(fn(&'a ())) {}
//~^ WARN conflicting implementations of trait `Trait` for type `for<'r> fn(fn(&'r ()))` [coherence_leak_check]
//~| WARN the behavior may change in a future release

fn main() {}
