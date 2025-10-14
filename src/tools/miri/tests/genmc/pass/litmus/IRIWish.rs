//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Translated from GenMC's "litmus/IRIWish" test.
// This test prints the values read by the different threads to check that we get all the values we expect.

// NOTE: the order of the lines in the output may change with changes to GenMC.
// Before blessing the new output, ensure that only the order of lines in the output changed, and none of the outputs are missing.

// NOTE: GenMC supports instruction caching and does not need replay completed threads.
// This means that an identical test in GenMC may output fewer lines (disable instruction caching to see all results).

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;
#[path = "../../../utils/mod.rs"]
mod utils;

use std::fmt::Write;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;
use crate::utils::*;

static X: AtomicU64 = AtomicU64::new(0);
static Y: AtomicU64 = AtomicU64::new(0);

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let mut results = [1234; 5];
        let ids = [
            spawn_pthread_closure(|| {
                X.store(1, Relaxed);
            }),
            spawn_pthread_closure(|| {
                let r1 = X.load(Relaxed);
                Y.store(r1, Release);
                results[0] = r1;
            }),
            spawn_pthread_closure(|| {
                results[1] = X.load(Relaxed);
                std::sync::atomic::fence(AcqRel);
                results[2] = Y.load(Relaxed);
            }),
            spawn_pthread_closure(|| {
                results[3] = Y.load(Relaxed);
                std::sync::atomic::fence(AcqRel);
                results[4] = X.load(Relaxed);
            }),
        ];
        join_pthreads(ids);

        // Print the values to check that we get all of them:
        writeln!(MiriStderr, "{results:?}").ok();

        0
    }
}
