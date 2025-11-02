//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Translated from GenMC's test "litmus/casdep".

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
            let a = X.load(Relaxed);
            let _b = Y.compare_exchange(a, 1, Relaxed, Relaxed);
            Z.store(a, Relaxed);
        });
        spawn_pthread_closure(|| {
            Y.store(2, Relaxed);
        });
        0
    }
}
