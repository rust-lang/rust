//@revisions: weak sc
//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows
//@[sc]compile-flags: -Zmiri-disable-weak-memory-emulation

// Translated from GenMC's test "2+2W".
//
// The pass tests "2w2w_3sc_1rel.rs", "2w2w_4rel" and "2w2w_4sc" and the fail test "2w2w_weak.rs" are related.
// Check "2w2w_weak.rs" for a more detailed description.
//
// The `sc` variant of this test checks that without weak memory emulation, only 3 instead of 4 executions are explored (like the `2w2w_4sc.rs` test).

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
                Y.store(1, Release);
                X.store(2, Release);
            }),
            spawn_pthread_closure(|| {
                X.store(1, Release);
                Y.store(2, Release);
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
