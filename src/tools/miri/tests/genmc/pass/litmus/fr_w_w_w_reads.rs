//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Translated from GenMC's "fr+w+w+w+reads" test.

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
        let mut result = [1234; 4];
        let ids = [
            spawn_pthread_closure(|| {
                X.store(1, Relaxed);
            }),
            spawn_pthread_closure(|| {
                X.store(2, Relaxed);
            }),
            spawn_pthread_closure(|| {
                X.store(3, Relaxed);
            }),
            spawn_pthread_closure(|| {
                result[0] = X.load(Relaxed);
                result[1] = X.load(Relaxed);
                result[2] = X.load(Relaxed);
                result[3] = X.load(Relaxed);
            }),
        ];
        // Join so we can read the final values.
        join_pthreads(ids);

        // Check that we don't get any unexpected values:
        for val in result {
            if !matches!(val, 0..=3) {
                std::process::abort();
            }
        }

        0
    }
}
