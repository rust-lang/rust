// compile-flags: -O
// run-pass

#![allow(unused_must_use)]
// Makes sure that zero-initializing large types is reasonably fast,
// Doing it incorrectly causes massive slowdown in LLVM during
// optimisation.

// pretty-expanded FIXME #23616
// ignore-emscripten no threads support

#![feature(intrinsics)]

use std::{mem, thread};

const SIZE: usize = 1024 * 1024;

fn main() {
    // do the test in a new thread to avoid (spurious?) stack overflows
    thread::spawn(|| {
        let _memory: [u8; SIZE] = unsafe { mem::zeroed() };
    }).join();
}
