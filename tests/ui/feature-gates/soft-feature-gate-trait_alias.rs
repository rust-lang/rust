// For historical reasons, trait aliases don't have a proper pre-expansion feature gate.
// We're now at least issuing a *warning* for those that only exist before macro expansion.
// FIXME(#154045): Turn their post-expansion feature gate into a proper pre-expansion one.
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
