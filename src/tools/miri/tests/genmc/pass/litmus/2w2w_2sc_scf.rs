//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Translated from GenMC's test "2+2W+2sc+scf".
// It tests correct handling of SeqCst fences combined with relaxed accesses.

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
            X.store(1, SeqCst);
            Y.store(2, SeqCst);
        });
        spawn_pthread_closure(|| {
            Y.store(1, Relaxed);
            std::sync::atomic::fence(SeqCst);
            X.store(2, Relaxed);
        });
        0
    }
}
