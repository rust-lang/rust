//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Translated from GenMC's "CoWR" test.

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
        let mut a = 1234;
        let ids = [
            spawn_pthread_closure(|| {
                X.store(1, Release);
                a = X.load(Acquire);
            }),
            spawn_pthread_closure(|| {
                X.store(2, Release);
            }),
        ];
        // Join so we can read the final values.
        join_pthreads(ids);

        // Check that we don't get any unexpected values (the load cannot read `0`):
        if !matches!(a, 1 | 2) {
            std::process::abort();
        }

        0
    }
}
