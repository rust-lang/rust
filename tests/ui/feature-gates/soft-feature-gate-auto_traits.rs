// For historical reasons, auto traits don't have an erroring pre-expansion feature gate.
// We're now at least issuing a warning for those that only exist before macro expansion.
// FIXME(#154045): Turn this pre-expansion warning into an error and remove the post-expansion gate.
//                 As part of this, move these test cases into `feature-gate-auto-traits.rs`.
//@ check-pass

#[cfg(false)]
auto trait Foo {}
//~^ WARN `auto` traits are unstable
//~| WARN unstable syntax can change at any point in the future

fn main() {}
