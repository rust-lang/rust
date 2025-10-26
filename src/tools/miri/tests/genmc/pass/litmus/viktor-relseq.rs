//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows -Zmiri-genmc-estimate

// Translated from GenMC's "litmus/viktor-relseq" test.
//
// This test also checks that we can run the GenMC estimation mode.

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;

static LOCK: AtomicU64 = AtomicU64::new(0);

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        spawn_pthread_closure(|| {
            LOCK.fetch_add(1, Acquire);
            LOCK.fetch_add(1, Relaxed);
        });
        spawn_pthread_closure(|| {
            LOCK.fetch_add(1, Relaxed);
            LOCK.fetch_add(1, Relaxed);
        });
        spawn_pthread_closure(|| {
            LOCK.fetch_add(1, Release);
        });
        spawn_pthread_closure(|| {
            LOCK.fetch_add(1, Relaxed);
        });
        0
    }
}
