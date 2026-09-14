// For historical reasons, try blocks don't have an erroring pre-expansion feature gate.
// We're now at least issuing a warning for those that only exist before macro expansion.
// FIXME(#154045): Turn this pre-expansion warning into an error and remove the post-expansion gate.
//                 As part of this, move these test cases into `feature-gate-try_blocks.rs`.
//@ edition: 2018
//@ check-pass

fn main() {
    #[cfg(false)]
    try {}
    //~^ WARN `try` blocks are unstable
    //~| WARN unstable syntax can change at any point
}
