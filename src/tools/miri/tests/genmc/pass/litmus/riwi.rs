//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Translated from GenMC's "litmus/riwi" test.

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
        spawn_pthread_closure(|| {
            Y.load(Relaxed);
            X.fetch_add(1, Relaxed);
        });
        spawn_pthread_closure(|| {
            X.fetch_add(1, Relaxed);
            Y.store(1, Release);
        });
        0
    }
}
