//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Test the basic functionality of compare_exchange.

#![no_main]

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::*;

static VALUE: AtomicUsize = AtomicUsize::new(0);

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    VALUE.store(1, SeqCst);

    // Expect success:
    if VALUE.compare_exchange(1, 2, SeqCst, SeqCst) != Ok(1) {
        std::process::abort();
    }
    // New value should be written:
    if 2 != VALUE.load(SeqCst) {
        std::process::abort()
    }

    // Expect failure:
    if VALUE.compare_exchange(1234, 42, SeqCst, SeqCst) != Err(2) {
        std::process::abort();
    }
    // Value should be unchanged:
    if 2 != VALUE.load(SeqCst) {
        std::process::abort()
    }
    0
}
