//@revisions: weak sc
//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows
//@[sc]compile-flags: -Zmiri-disable-weak-memory-emulation

// Translated from GenMC's "litmus/Z6.U" test.
//
// The `sc` variant of this test checks that we get fewer executions when weak memory emulation is disabled.

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
        let mut a = 1234;
        let mut b = 1234;
        let ids = [
            spawn_pthread_closure(|| {
                X.store(1, SeqCst);
                Y.store(1, Release);
            }),
            spawn_pthread_closure(|| {
                Y.fetch_add(1, SeqCst);
                a = Y.load(Relaxed);
            }),
            spawn_pthread_closure(|| {
                Y.store(3, SeqCst);
                b = X.load(SeqCst);
            }),
        ];
        join_pthreads(ids);

        // Print the values to check that we get all of them:
        writeln!(MiriStderr, "a={a}, b={b}, X={}, Y={}", X.load(Relaxed), Y.load(Relaxed)).ok();
        0
    }
}
