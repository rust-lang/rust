//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Translated from GenMC's test `wrong/racy/MPU2+rels+rlx`.
// Test if Miri with GenMC can detect the data race on `X`.
// The data race only occurs if thread 1 finishes, then threads 3 and 4 run, then thread 2.
//
// This data race is hard to detect for Miri without GenMC, requiring -Zmiri-many-seeds=0..1024 at the time this test was created.

// FIXME(genmc): once Miri-GenMC error reporting is improved, ensure that it correctly points to the two spans involved in the data race.

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::*;

use genmc::spawn_pthread_closure;
static mut X: u64 = 0;
static Y: AtomicUsize = AtomicUsize::new(0);

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let _t1 = spawn_pthread_closure(|| {
            X = 1;
            Y.store(0, Release);
            Y.store(1, Relaxed);
        });
        let _t2 = spawn_pthread_closure(|| {
            if Y.load(Relaxed) > 2 {
                X = 2; //~ ERROR: Undefined Behavior: Non-atomic race
            }
        });
        let _t3 = spawn_pthread_closure(|| {
            let _ = Y.compare_exchange(2, 3, Relaxed, Relaxed);
        });

        let _t4 = spawn_pthread_closure(|| {
            let _ = Y.compare_exchange(1, 2, Relaxed, Relaxed);
        });
    }
    0
}
