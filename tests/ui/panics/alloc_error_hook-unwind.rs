//! Test that out-of-memory conditions trigger catchable panics with `set_alloc_error_hook`.

//@ run-pass
//@ needs-unwind
//@ only-linux
//@ ignore-backends: gcc

#![feature(alloc_error_hook)]

use std::hint::black_box;
use std::mem::forget;
use std::panic::catch_unwind;

fn main() {
    std::alloc::set_alloc_error_hook(|_| panic!());

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
