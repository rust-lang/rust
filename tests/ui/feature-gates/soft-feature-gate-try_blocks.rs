// For historical reasons, try blocks don't have a proper pre-expansion feature gate.
// We're now at least issuing a *warning* for those that only exist before macro expansion.
// FIXME(#154045): Turn their post-expansion feature gate into a proper pre-expansion one.
//                 As part of this, move these test cases into `feature-gate-try_blocks.rs`.
//@ edition: 2018
//@ check-pass

fn main() {
    #[cfg(false)]
    try {}
    //~^ WARN `try` blocks are unstable
    //~| WARN unstable syntax can change at any point
}
