//@ignore-target: windows # No pthreads on Windows
// We are making scheduler assumptions here.
//@compile-flags: -Zmiri-deterministic-concurrency

// Joining itself is undefined behavior.

use std::{ptr, thread};

fn main() {
    let handle = thread::spawn(|| {
        unsafe {
            let native: libc::pthread_t = libc::pthread_self();
            assert_eq!(libc::pthread_join(native, ptr::null_mut()), 0); //~ ERROR: Undefined Behavior: trying to join itself
        }
    });
    thread::yield_now();
    handle.join().unwrap();
}
