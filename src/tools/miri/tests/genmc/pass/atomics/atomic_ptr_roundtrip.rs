//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Test that we can send pointers with any alignment to GenMC and back, even across threads.
// After a round-trip, the pointers should still work properly (no missing provenance).

#![no_main]
#![allow(static_mut_refs)]

#[path = "../../../utils/genmc.rs"]
mod genmc;
#[path = "../../../utils/mod.rs"]
mod utils;

use std::sync::atomic::AtomicPtr;
use std::sync::atomic::Ordering::*;

use genmc::*;
use utils::*;

static PTR: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());

static mut X: [u8; 16] = [0; 16];

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let ids = [
            spawn_pthread_closure(|| {
                for i in 0..X.len() {
                    X[i] = i.try_into().unwrap();
                    PTR.store(&raw mut X[i], SeqCst);
                    // Wait for the other thread to reset the AtomicPtr.
                    miri_genmc_assume(PTR.load(SeqCst).is_null());
                    // Check that we see the update the other thread did through the pointer.
                    if X[i] != (i + 1) as u8 {
                        std::process::abort();
                    }
                }
            }),
            spawn_pthread_closure(|| {
                for i in 0..X.len() {
                    let x = PTR.load(SeqCst);
                    // Wait for the other thread to store the next pointer.
                    miri_genmc_assume(!x.is_null());
                    // Check that we see the update when reading from the pointer.
                    if usize::from(*x) != i {
                        std::process::abort();
                    }
                    *x = (i + 1) as u8;
                    PTR.store(std::ptr::null_mut(), SeqCst);
                }
            }),
        ];
        join_pthreads(ids);
        0
    }
}
