//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Translated from GenMC's "CoRR1" test.

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
        let mut b = 1234;
        let mut c = 1234;
        let ids = [
            spawn_pthread_closure(|| {
                X.store(1, Release);
            }),
            spawn_pthread_closure(|| {
                X.store(2, Release);
            }),
            spawn_pthread_closure(|| {
                a = X.load(Acquire);
                b = X.load(Acquire);
            }),
            spawn_pthread_closure(|| {
                c = X.load(Acquire);
            }),
        ];
        // Join so we can read the final values.
        join_pthreads(ids);

        // Check that we don't get any unexpected values (only 0, 1, 2 are allowed):
        if !(matches!(a, 0..=2) && matches!(b, 0..=2) && matches!(c, 0..=2)) {
            std::process::abort();
        }
        // The 36 possible program executions can have 21 different results for (a, b, c).
        // Of the 27 = 3*3*3 total results for (a, b, c),
        // those where `a != 0` and `b == 0` are not allowed by the memory model.
        // Once the load for `a` reads either 1 or 2, the load for `b` must see that store too, so it cannot read 0.
        if a != 0 && b == 0 {
            std::process::abort();
        }

        0
    }
}
