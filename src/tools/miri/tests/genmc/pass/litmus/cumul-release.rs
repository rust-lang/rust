//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Translated from GenMC's test "litmus/cumul-release".

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;

static X: AtomicU64 = AtomicU64::new(0);
static Y: AtomicU64 = AtomicU64::new(0);
static Z: AtomicU64 = AtomicU64::new(0);

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let mut a = 1234;
        let mut b = 1234;
        let mut c = 1234;
        let ids = [
            spawn_pthread_closure(|| {
                X.store(1, Relaxed);
                Y.store(1, Release);
            }),
            spawn_pthread_closure(|| {
                a = Y.load(Relaxed);
                Z.store(a, Relaxed);
            }),
            spawn_pthread_closure(|| {
                b = Z.load(Relaxed);
                std::sync::atomic::fence(AcqRel);
                c = X.load(Relaxed);
            }),
        ];
        // Join so we can read the final values.
        join_pthreads(ids);

        // Check that we don't get any unexpected values:
        if !matches!(
            (a, b, c),
            (0, 0, 0) | (0, 0, 1) | (1, 0, 0) | (1, 0, 1) | (1, 1, 0) | (1, 1, 1)
        ) {
            std::process::abort();
        }

        0
    }
}
