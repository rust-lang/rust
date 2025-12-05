//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows -Zmiri-genmc-verbose
//@normalize-stderr-test: "Verification took .*s" -> "Verification took [TIME]s"

// Test that we can detect a deadlock involving `std::sync::Mutex` in GenMC mode.
// FIXME(genmc): We cannot detect the deadlock currently. Instead, the deadlocked execution is treated like any other blocked execution.
// This behavior matches GenMC's on an equivalent program, and additional analysis is required to detect such deadlocks.
// This should become a `fail` test once this deadlock can be detected.
//
// FIXME(genmc): use `std::thread` once GenMC mode performance is better and produces fewer warnings for compare_exchange.

#![no_main]
#![feature(abort_unwind)]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::Mutex;

use crate::genmc::*;

static X: Mutex<u64> = Mutex::new(0);
static Y: Mutex<u64> = Mutex::new(0);

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let t0 = spawn_pthread_closure(|| {
            let mut x = X.lock().unwrap();
            let mut y = Y.lock().unwrap();
            *x += 1;
            *y += 1;
        });
        let t1 = spawn_pthread_closure(|| {
            let mut y = Y.lock().unwrap();
            let mut x = X.lock().unwrap();
            *x += 1;
            *y += 1;
        });
        join_pthreads([t0, t1]);
        0
    }
}
