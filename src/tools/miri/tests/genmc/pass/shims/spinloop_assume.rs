//@ revisions: bounded123 bounded321 replaced123 replaced321
//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows -Zmiri-genmc-verbose
//@normalize-stderr-test: "Verification took .*s" -> "Verification took [TIME]s"

// This test uses GenMC assume statements to bound or replace spinloops.
// Three threads pass a value to each other, spinning on an atomic FLAG to wait for the previous thread.
//
// There are two variants, one limits the spinloop to three iterations, and one that completely replaces the spin loop.
// Without this loop bounding, this test *cannot* be verified, since GenMC will have to explore infinitely many executions (one per possible number of loop iterations).

// FIXME(genmc): GenMC provides the `--unroll=N` option, which limits all loops to at most N iterations (at the LLVM IR level).
// Such an option for Miri would allow a variant of this test without manual bounding, using this automatic loop bounding instead.

// We use different thread orders to ensure it doesn't just pass by chance (each thread order should give the same result).
// We use verbose output to see the number of explored vs blocked executions.

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;
#[path = "../../../utils/mod.rs"]
mod utils;

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;
use crate::utils::*;

static mut X: u64 = 0;
static FLAG: AtomicU64 = AtomicU64::new(0);

/// Unbounded variant of the spinloop.
/// This function causes GenMC to explore infinite executions.
#[allow(unused)]
fn spin_until_unbounded(value: u64) {
    while FLAG.load(Acquire) != value {
        std::hint::spin_loop();
    }
}

#[cfg(any(bounded123, bounded321))]
/// We bound the loop to at most 3 iterations.
fn spin_until(value: u64) {
    for _ in 0..3 {
        if FLAG.load(Acquire) == value {
            return;
        }
    }
    unsafe { miri_genmc_assume(false) };
}

#[cfg(not(any(bounded123, bounded321)))]
/// For full replacement, we limit it to only 1 load.
fn spin_until(value: u64) {
    unsafe { miri_genmc_assume(FLAG.load(Acquire) == value) };
}

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let t0 = || {
            X = 42;
            FLAG.store(1, Release);

            spin_until(3);
            let c = X;
            if c != 44 {
                std::process::abort();
            }
        };
        let t1 = || {
            spin_until(1);
            let a = X;
            X = a + 1;
            FLAG.store(2, Release);
        };
        let t2 = || {
            spin_until(2);
            let b = X;
            X = b + 1;
            FLAG.store(3, Release);
        };
        // Reverse the order for the second test variant.
        #[cfg(any(bounded321, replaced321))]
        let (t0, t1, t2) = (t2, t1, t0);

        spawn_pthread_closure(t0);
        spawn_pthread_closure(t1);
        spawn_pthread_closure(t2);

        0
    }
}
