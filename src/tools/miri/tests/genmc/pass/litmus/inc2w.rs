//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Translated from GenMC's test "litmus/inc2w".

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;

static X: AtomicU64 = AtomicU64::new(0);

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let ids = [
            spawn_pthread_closure(|| {
                X.fetch_add(1, Relaxed);
            }),
            spawn_pthread_closure(|| {
                X.store(4, Release);
            }),
            spawn_pthread_closure(|| {
                X.fetch_add(2, Relaxed);
            }),
        ];
        // Join so we can read the final values.
        join_pthreads(ids);

        // Check that we don't get any unexpected values:
        let x = X.load(Relaxed);
        if !matches!(x, 4..=7) {
            std::process::abort();
        }

        0
    }
}
