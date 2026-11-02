//@compile-flags: -Zmiri-genmc-weak-cas-spurious-failures

// Translated from GenMC's test `correct/synthetic/weak-cas-spurious`.
// Observable spurious failure of a weak CAS (not used in a loop).

#![no_main]

use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering::*;

static X: AtomicI32 = AtomicI32::new(0);

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    // Either the CAS succeeds, or it fails spuriously and leaves `X` unchanged.
    match X.compare_exchange_weak(0, 1, Relaxed, Relaxed) {
        Ok(0) => assert!(X.load(Relaxed) == 1),
        Err(0) => assert!(X.load(Relaxed) == 0),
        _ => std::process::abort(),
    }
    0
}
