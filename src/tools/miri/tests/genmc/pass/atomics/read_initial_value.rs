//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Test that we can read the initial value of global, heap and stack allocations in GenMC mode.

#![no_main]

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::*;

static X: AtomicU64 = AtomicU64::new(1234);

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    // Read initial value of global allocation.
    if 1234 != unsafe { *X.as_ptr() } {
        std::process::abort();
    }
    if 1234 != X.load(SeqCst) {
        std::process::abort();
    }

    // Read initial value of stack allocation.
    let a = AtomicU64::new(0xBB);
    if 0xBB != unsafe { *a.as_ptr() } {
        std::process::abort();
    }
    if 0xBB != a.load(SeqCst) {
        std::process::abort();
    }

    // Read initial value of heap allocation.
    let b = Box::new(AtomicU64::new(0xCCC));
    if 0xCCC != unsafe { *b.as_ptr() } {
        std::process::abort();
    }
    if 0xCCC != b.load(SeqCst) {
        std::process::abort();
    }

    0
}
