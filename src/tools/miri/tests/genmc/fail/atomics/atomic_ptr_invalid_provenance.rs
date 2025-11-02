//@revisions: send make
//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Test that we can distinguish two pointers with the same address, but different provenance, after they are sent to GenMC and back.
// We have two variants, one where we send such a pointer to GenMC, and one where we make it on the GenMC side.

#![no_main]
#![feature(box_as_ptr)]

use std::sync::atomic::AtomicPtr;
use std::sync::atomic::Ordering::*;

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let atomic_ptr = AtomicPtr::new(std::ptr::null_mut());
        let mut a = Box::new(0u64);
        let mut b = Box::new(0u64);
        let a_ptr: *mut u64 = Box::as_mut_ptr(&mut a);
        let b_ptr: *mut u64 = Box::as_mut_ptr(&mut b);

        // Store valid pointer to `a`:
        atomic_ptr.store(a_ptr, Relaxed);
        let ptr = atomic_ptr.load(Relaxed);
        *ptr = 42;
        if *a != 42 {
            std::process::abort();
        }
        // Store valid pointer to `b`:
        atomic_ptr.store(b_ptr, Relaxed);
        let ptr = atomic_ptr.load(Relaxed);
        *ptr = 43;
        if *b != 43 {
            std::process::abort();
        }

        // Make `atomic_ptr` contain a pointer with the provenance of `b`, but the address of `a`.
        if cfg!(send) {
            // Variant 1: create the invalid pointer non-atomically, then send it to GenMC.
            let fake_a_ptr = b_ptr.with_addr(a_ptr.addr());
            if a_ptr.addr() != fake_a_ptr.addr() {
                std::process::abort();
            }
            atomic_ptr.store(fake_a_ptr, Relaxed);
        } else {
            // Variant 2: send `b_ptr` to GenMC, then create the invalid pointer to `a` using atomic operations.
            atomic_ptr.store(b_ptr, Relaxed);
            atomic_ptr.fetch_byte_add(a_ptr.addr(), Relaxed);
            atomic_ptr.fetch_byte_sub(b_ptr.addr(), Relaxed);
        }
        let ptr = atomic_ptr.load(Relaxed);
        if a_ptr.addr() != ptr.addr() {
            std::process::abort();
        }
        // This pointer has the same address as `a_ptr`, but not the same
        // provenance, so writing to it fails.
        *ptr = 44; //~ ERROR: points to before the beginning of the allocation

        0
    }
}
