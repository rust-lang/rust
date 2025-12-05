//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Test that we can detect a double-free bug across two threads, which only shows up if the second thread reads an atomic pointer at a very specific moment.
// GenMC can detect this error consistently, without having to run the buggy code with multiple RNG seeds or in a loop.

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::alloc::{Layout, alloc, dealloc};
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;

static X: AtomicPtr<u64> = AtomicPtr::new(std::ptr::null_mut());

unsafe fn free(ptr: *mut u64) {
    dealloc(ptr as *mut u8, Layout::new::<u64>()) //~ ERROR: Undefined Behavior
}

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let ids = [
            spawn_pthread_closure(|| {
                let a: *mut u64 = alloc(Layout::new::<u64>()) as *mut u64;
                X.store(a, SeqCst);
                // We have to yield to the other thread exactly here to reproduce the double-free.
                let b = X.swap(std::ptr::null_mut(), SeqCst);
                free(b);
            }),
            spawn_pthread_closure(|| {
                let b = X.load(SeqCst);
                if !b.is_null() {
                    free(b);
                }
            }),
        ];
        join_pthreads(ids);
        0
    }
}
