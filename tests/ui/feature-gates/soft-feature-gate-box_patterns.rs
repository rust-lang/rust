// For historical reasons, box patterns don't have a proper pre-expansion feature gate.
// We're now at least issuing a *warning* for those that only exist before macro expansion.
// FIXME(#154045): Turn their post-expansion feature gate into a proper pre-expansion one.
//                 As part of this, move these test cases into `feature-gate-box_patterns.rs`.
//@ check-pass

fn main() {
    #[cfg(false)]
    let box x;
    //~^ WARN box pattern syntax is experimental
    //~| WARN unstable syntax can change at any point in the future

    #[cfg(false)]
    let Packet { box x };
    //~^ WARN box pattern syntax is experimental
    //~| WARN unstable syntax can change at any point in the future
}
