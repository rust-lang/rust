//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows
//@revisions: release1 release2

// Translated from GenMC's test "2+2W+3sc+rel1" and "2+2W+3sc+rel2" (two variants that swap which store is `Release`).
//
// The pass tests "2w2w_3sc_1rel.rs", "2w2w_4rel" and "2w2w_4sc" and the fail test "2w2w_weak.rs" are related.
// Check "2w2w_weak.rs" for a more detailed description.

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;

static X: AtomicU64 = AtomicU64::new(0);
static Y: AtomicU64 = AtomicU64::new(0);

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let ids = [
            spawn_pthread_closure(|| {
                X.store(1, SeqCst);
                Y.store(2, SeqCst);
            }),
            // Variant 1: `Release` goes first.
            #[cfg(release1)]
            spawn_pthread_closure(|| {
                Y.store(1, Release);
                X.store(2, SeqCst);
            }),
            // Variant 2: `Release` goes second.
            #[cfg(not(release1))]
            spawn_pthread_closure(|| {
                Y.store(1, SeqCst);
                X.store(2, Release);
            }),
        ];
        // Join so we can read the final values.
        join_pthreads(ids);

        // Check that we don't get any unexpected values:
        let result = (X.load(Relaxed), Y.load(Relaxed));
        if !matches!(result, (1, 2) | (1, 1) | (2, 2) | (2, 1)) {
            std::process::abort();
        }

        0
    }
}
