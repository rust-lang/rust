//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows
//@revisions: join no_join

// Translated from GenMC's "litmus/detour" test.

// This test has two revisitions to test whether we get the same result
// independent of whether we join the spawned threads or not.

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::atomic::AtomicI64;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;

static X: AtomicI64 = AtomicI64::new(0);
static Y: AtomicI64 = AtomicI64::new(0);
static Z: AtomicI64 = AtomicI64::new(0);

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        // Make these static so we can exit the main thread while the other threads still run.
        // If these are `let mut` like the other tests, this will cause a use-after-free bug.
        static mut A: i64 = 1234;
        static mut B: i64 = 1234;
        static mut C: i64 = 1234;
        let ids = [
            spawn_pthread_closure(|| {
                X.store(1, Relaxed);
            }),
            spawn_pthread_closure(|| {
                A = Z.load(Relaxed);
                X.store(A.wrapping_sub(1), Relaxed);
                B = X.load(Relaxed);
                Y.store(B, Relaxed);
            }),
            spawn_pthread_closure(|| {
                C = Y.load(Relaxed);
                Z.store(C, Relaxed);
            }),
        ];

        // The `no_join` revision doesn't join any of the running threads to test that
        // we still explore the same number of executions in that case.
        if cfg!(no_join) {
            return 0;
        }

        // Join so we can read the final values.
        join_pthreads(ids);

        // Check that we don't get any unexpected values:
        if !matches!((A, B, C), (0, 1, 0) | (0, -1, 0) | (0, 1, 1) | (0, -1, -1)) {
            std::process::abort();
        }

        0
    }
}
