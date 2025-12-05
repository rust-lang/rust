//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Test that we can read the value of a non-atomic store atomically and an of an atomic value non-atomically.

#![no_main]

use std::sync::atomic::Ordering::*;
use std::sync::atomic::{AtomicI8, AtomicU64};

static X: AtomicU64 = AtomicU64::new(1234);
static Y: AtomicI8 = AtomicI8::new(0xB);

fn assert_equals<T: Eq>(a: T, b: T) {
    if a != b {
        std::process::abort();
    }
}

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    // 8 byte unsigned integer:
    // Read initial value.
    assert_equals(1234, X.load(Relaxed));
    // Atomic store, non-atomic load.
    X.store(0xFFFF, Relaxed);
    assert_equals(0xFFFF, unsafe { *X.as_ptr() });
    // Non-atomic store, atomic load.
    unsafe { *X.as_ptr() = 0xAAAA };
    assert_equals(0xAAAA, X.load(Relaxed));

    // 1 byte signed integer:
    // Read initial value.
    assert_equals(0xB, Y.load(Relaxed));
    // Atomic store, non-atomic load.
    Y.store(42, Relaxed);
    assert_equals(42, unsafe { *Y.as_ptr() });
    // Non-atomic store, atomic load.
    unsafe { *Y.as_ptr() = -42 };
    assert_equals(-42, Y.load(Relaxed));
    0
}
