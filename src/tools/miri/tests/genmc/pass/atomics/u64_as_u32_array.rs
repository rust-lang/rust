//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Tests mixed-size non-atomic accesses.

#![no_main]

use std::sync::atomic::*;

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    let mut data = 0u64;
    // Treat this like an array of two AtomicI32.
    let atomics = unsafe { &*(&raw mut data as *mut u64 as *mut [AtomicI32; 2]) };

    atomics[0].load(Ordering::SeqCst);
    atomics[1].store(-1, Ordering::SeqCst);
    atomics[0].store(-1, Ordering::Relaxed);

    assert_eq!(data, u64::MAX);

    0
}
