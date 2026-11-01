// For historical reasons, trait aliases don't have an erroring pre-expansion feature gate.
// We're now at least issuing a warning for those that only exist before macro expansion.
// FIXME(#154045): Turn this pre-expansion warning into an error and remove the post-expansion gate.
//                 As part of this, move these test cases into `feature-gate-trait-alias.rs`.
//@ check-pass

#[cfg(false)]
trait Trait =;
//~^ WARN trait aliases are experimental
//~| WARN unstable syntax can change at any point in the future

#[cfg(false)]
trait Trait<T> = Bound where T: Bound;
//~^ WARN trait aliases are experimental
//~| WARN unstable syntax can change at any point in the future

fn main() {}
