//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Translated from GenMC's "litmus/MP+rels+acqf" test.

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::*;

// Note: the GenMC equivalent of this test (genmc/tests/correct/litmus/MP+rels+acqf/mp+rels+acqf.c) uses non-atomic accesses for `X` with disabled race detection.
static X: AtomicU64 = AtomicU64::new(0);
static Y: AtomicU64 = AtomicU64::new(0);

use crate::genmc::*;

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        spawn_pthread_closure(|| {
            X.store(1, Relaxed);

            Y.store(0, Release);
            Y.store(1, Relaxed);
        });
        spawn_pthread_closure(|| {
            if Y.load(Relaxed) != 0 {
                std::sync::atomic::fence(Acquire);
                let _x = X.load(Relaxed);
            }
        });
        0
    }
}
