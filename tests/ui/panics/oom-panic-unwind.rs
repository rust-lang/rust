//! Test that out-of-memory conditions trigger catchable panics with `-Z oom=panic`.

//@ compile-flags: -Z oom=panic
//@ run-pass
//@ no-prefer-dynamic
//@ needs-unwind
//@ only-linux
//@ ignore-backends: gcc

use std::hint::black_box;
use std::mem::forget;
use std::panic::catch_unwind;

fn main() {
    let panic = catch_unwind(|| {
        // This is guaranteed to exceed even the size of the address space
        for _ in 0..16 {
            // Truncates to a suitable value for both 32-bit and 64-bit targets.
            let alloc_size = 0x1000_0000_1000_0000u64 as usize;
            forget(black_box(vec![0u8; alloc_size]));
        }
    });
    assert!(panic.is_err());
}
