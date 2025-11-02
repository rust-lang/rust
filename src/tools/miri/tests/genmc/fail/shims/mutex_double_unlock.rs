//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows
//@error-in-other-file: Undefined Behavior

// Test that GenMC can detect a double unlock of a mutex.
// This test will cause an error even if the program actually would work entirely fine despite the double-unlock
// because GenMC always assumes a `pthread`-like API.

#![no_main]

use std::sync::Mutex;

static MUTEX: Mutex<u64> = Mutex::new(0);

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    let mut guard = MUTEX.lock().unwrap();
    unsafe {
        std::ptr::drop_in_place(&raw mut guard);
    }
    drop(guard);
    0
}
