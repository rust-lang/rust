//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Translated from GenMC's "litmus/Z6+acq" test.

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
            X.store(1, Relaxed);
            std::sync::atomic::fence(SeqCst);
            Y.store(1, Relaxed);
        });
        spawn_pthread_closure(|| {
            Y.load(Acquire);
            Z.store(1, Relaxed);
        });
        spawn_pthread_closure(|| {
            Z.store(2, Relaxed);
            std::sync::atomic::fence(SeqCst);
            X.load(Relaxed);
        });
        0
    }
}
