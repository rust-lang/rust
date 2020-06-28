// ignore-windows: No libc on Windows

// Joining itself is undefined behavior.

#![feature(rustc_private)]

extern crate libc;

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
