//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Translated from GenMC's "litmus/MPU2+rels+acqf" test.

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

// Note: the GenMC equivalent of this test (genmc/tests/correct/litmus/MPU2+rels+acqf/mpu2+rels+acqf.c) uses non-atomic accesses for `X` with disabled race detection.
static X: AtomicU64 = AtomicU64::new(0);
static Y: AtomicU64 = AtomicU64::new(0);

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let mut a = Ok(1234);
        let mut b = Ok(1234);
        let mut c = 1234;
        let ids = [
            spawn_pthread_closure(|| {
                X.store(1, Relaxed);

                Y.store(0, Release);
                Y.store(1, Relaxed);
            }),
            spawn_pthread_closure(|| {
                a = Y.compare_exchange(2, 3, Relaxed, Relaxed);
            }),
            spawn_pthread_closure(|| {
                b = Y.compare_exchange(1, 2, Relaxed, Relaxed);
            }),
            spawn_pthread_closure(|| {
                c = Y.load(Acquire);
                if c > 2 {
                    std::sync::atomic::fence(Acquire);
                    X.store(2, Relaxed);
                }
            }),
        ];
        join_pthreads(ids);

        // Print the values to check that we get all of them:
        writeln!(
            MiriStderr,
            "X={}, Y={}, a={a:?}, b={b:?}, c={c}",
            X.load(Relaxed),
            Y.load(Relaxed)
        )
        .ok();

        0
    }
}
