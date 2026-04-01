//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Translated from GenMC's "litmus/MP+incMP" test.

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;

static X: AtomicU64 = AtomicU64::new(0);
static Y: AtomicU64 = AtomicU64::new(0);
static Z: AtomicU64 = AtomicU64::new(0);

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        spawn_pthread_closure(|| {
            Y.load(Acquire);
            Z.fetch_add(1, AcqRel);
        });
        spawn_pthread_closure(|| {
            Z.fetch_add(1, AcqRel);
            X.load(Acquire);
        });
        spawn_pthread_closure(|| {
            X.store(1, Release);
            Y.store(1, Release);
        });
        0
    }
}
