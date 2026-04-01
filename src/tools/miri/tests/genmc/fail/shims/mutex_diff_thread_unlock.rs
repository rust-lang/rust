//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows
//@error-in-other-file: Undefined Behavior

// Test that GenMC throws an error if a `std::sync::Mutex` is unlocked from a different thread than the one that locked it.
//
// This test will cause an error on all targets, even mutexes on that targets allow for unlocking on a different thread.
// GenMC always assumes a `pthread`-like API.

#![no_main]

use std::sync::Mutex;

static MUTEX: Mutex<u64> = Mutex::new(0);

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);
unsafe impl<T> Send for EvilSend<T> {}

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    let guard = EvilSend(MUTEX.lock().unwrap());
    let handle = std::thread::spawn(move || {
        let guard = guard; // avoid field capturing
        drop(guard);
    });
    handle.join().unwrap();
    0
}
