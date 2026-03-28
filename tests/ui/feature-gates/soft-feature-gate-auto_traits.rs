// For historical reasons, auto traits don't have a proper pre-expansion feature gate.
// We're now at least issuing a *warning* for those that only exist before macro expansion.
// FIXME(#154045): Turn their post-expansion feature gate into a proper pre-expansion one.
//                 As part of this, move these test cases into `feature-gate-auto-traits.rs`.
//@ check-pass

#[cfg(false)]
auto trait Foo {}
//~^ WARN `auto` traits are unstable
//~| WARN unstable syntax can change at any point in the future

fn main() {}
