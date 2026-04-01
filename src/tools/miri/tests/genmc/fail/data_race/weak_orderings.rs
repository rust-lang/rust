//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows
//@revisions: rlx_rlx rlx_acq rel_rlx

// Translated from GenMC's test `wrong/racy/MP+rel+rlx`, `MP+rlx+acq` and `MP+rlx+rlx`.
// Test if Miri with GenMC can detect the data race on `X`.
// Relaxed orderings on an atomic store-load pair should not synchronize the non-atomic write to X, leading to a data race.

// FIXME(genmc): once Miri-GenMC error reporting is improved, ensure that it correctly points to the two spans involved in the data race.

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{self, *};

use genmc::spawn_pthread_closure;

static mut X: u64 = 0;
static Y: AtomicUsize = AtomicUsize::new(0);

const STORE_ORD: Ordering = if cfg!(rel_rlx) { Release } else { Relaxed };
const LOAD_ORD: Ordering = if cfg!(rlx_acq) { Acquire } else { Relaxed };

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        spawn_pthread_closure(|| {
            X = 1;
            Y.store(1, STORE_ORD);
        });
        spawn_pthread_closure(|| {
            if Y.load(LOAD_ORD) != 0 {
                X = 2; //~ ERROR: Undefined Behavior: Non-atomic race
            }
        });
    }
    0
}
