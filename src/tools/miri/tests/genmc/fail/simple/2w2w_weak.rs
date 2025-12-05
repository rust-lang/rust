//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows
//@revisions: sc3_rel1 release4 relaxed4

// The pass tests "2w2w_3sc_1rel.rs", "2w2w_4rel" and "2w2w_4sc" and the fail test "2w2w_weak.rs" are related.
//
// This test has multiple variants using different memory orderings.
// When using any combination of orderings except using all 4 `SeqCst`, the memory model allows the program to result in (X, Y) == (1, 1).
// The "pass" variants only check that we get the expected number of executions (3 for all SC, 4 otherwise),
// and a valid outcome every execution, but do not check that we get all allowed results.
// This "fail" variant ensures we can explore the execution resulting in (1, 1), with an incorrect assumption that the result (1, 1) is impossible.
//
// Miri without GenMC is unable to produce this program execution and thus detect the incorrect assumption, even with `-Zmiri-many-seeds`.
//
// To get good coverage, we test the combination with the strongest orderings allowing this result (3 `SeqCst`, 1 `Release`),
// the weakest orderings (4 `Relaxed`), and one in between (4 `Release`).

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::{self, *};

use crate::genmc::{join_pthreads, spawn_pthread_closure};

static X: AtomicU64 = AtomicU64::new(0);
static Y: AtomicU64 = AtomicU64::new(0);

// Strongest orderings allowing result (1, 1).
#[cfg(seqcst_rel)]
const STORE_ORD_3: Ordering = SeqCst;
#[cfg(seqcst_rel)]
const STORE_ORD_1: Ordering = Release;

// 4 * `Release`.
#[cfg(acqrel)]
const STORE_ORD_3: Ordering = Release;
#[cfg(acqrel)]
const STORE_ORD_1: Ordering = Release;

// Weakest orderings (4 * `Relaxed`).
#[cfg(not(any(acqrel, seqcst_rel)))]
const STORE_ORD_3: Ordering = Relaxed;
#[cfg(not(any(acqrel, seqcst_rel)))]
const STORE_ORD_1: Ordering = Relaxed;

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let ids = [
            spawn_pthread_closure(|| {
                X.store(1, STORE_ORD_3);
                Y.store(2, STORE_ORD_3);
            }),
            spawn_pthread_closure(|| {
                Y.store(1, STORE_ORD_1);
                X.store(2, STORE_ORD_3);
            }),
        ];
        // Join so we can read the final values.
        join_pthreads(ids);

        // We incorrectly assume that the result (1, 1) as unreachable.
        let result = (X.load(Relaxed), Y.load(Relaxed));
        if result == (1, 1) {
            std::process::abort(); //~ ERROR: abnormal termination
        }

        0
    }
}
