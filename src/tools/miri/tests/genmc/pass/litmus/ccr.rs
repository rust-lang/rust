//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Translated from GenMC's test "litmus/ccr".

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;

static X: AtomicU64 = AtomicU64::new(0);

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    // FIXME(genmc,HACK): remove this initializing write once Miri-GenMC supports mixed atomic-non-atomic accesses.
    X.store(0, Relaxed);

    unsafe {
        spawn_pthread_closure(|| {
            let expected = 0;
            let _ = X.compare_exchange(expected, 42, Relaxed, Relaxed);
        });
        spawn_pthread_closure(|| {
            let expected = 0;
            let _ = X.compare_exchange(expected, 17, Relaxed, Relaxed);
            X.load(Relaxed);
        });
        0
    }
}
