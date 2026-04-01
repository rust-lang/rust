//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows -Zmiri-genmc-verbose
//@normalize-stderr-test: "Verification took .*s" -> "Verification took [TIME]s"

// Test various features of the `std::sync::Mutex` API with GenMC.
// Miri running with GenMC intercepts the Mutex functions `lock`, `try_lock` and `unlock`, instead of running their actual implementation.
// This interception should not break any functionality.
//
// FIXME(genmc): Once GenMC supports mixed size accesses, add stack/heap allocated Mutexes to the test.
// FIXME(genmc): Once the actual implementation of mutexes can be used in GenMC mode and there is a setting to disable Mutex interception: Add test revision without interception.
//
// Miri provides annotations to GenMC for the condition required to unblock a thread blocked on a Mutex lock call.
// This massively reduces the number of blocked executions we need to explore (in this test we require zero blocked execution).
// We use verbose output to check that this test always explores zero blocked executions.

#![no_main]
#![feature(abort_unwind)]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::Mutex;

use crate::genmc::*;

const REPS: u64 = 3;

static LOCK: Mutex<u64> = Mutex::new(0);
static OTHER_LOCK: Mutex<u64> = Mutex::new(1234);

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    std::panic::abort_unwind(main_);
    0
}

fn main_() {
    // Two mutexes should not interfere, holding this guard does not affect the other mutex.
    let other_guard = OTHER_LOCK.lock().unwrap();

    let guard = LOCK.lock().unwrap();
    // Trying to lock should fail if the mutex is already held.
    assert!(LOCK.try_lock().is_err());
    // Dropping the guard should unlock the mutex correctly.
    drop(guard);
    // Trying to lock now should succeed.
    assert!(LOCK.try_lock().is_ok());

    // Spawn multiple threads interacting with the same mutex.
    unsafe {
        let ids = [
            spawn_pthread_closure(|| {
                for _ in 0..REPS {
                    *LOCK.lock().unwrap() += 2;
                }
            }),
            spawn_pthread_closure(|| {
                for _ in 0..REPS {
                    *LOCK.lock().unwrap() += 4;
                }
            }),
        ];
        join_pthreads(ids);
    }
    // Due to the Mutex, all increments should be visible in every explored execution.
    assert!(*LOCK.lock().unwrap() == REPS * 6);

    drop(other_guard);
}
