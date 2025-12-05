//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Test that use-after-free bugs involving atomic pointers are detected in GenMC mode.

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;
#[path = "../../../utils/mod.rs"]
mod utils;

use std::sync::atomic::AtomicPtr;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;
use crate::utils::*;

static X: AtomicPtr<u64> = AtomicPtr::new(std::ptr::null_mut());
static mut Y: u64 = 0;

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let ids = [
            spawn_pthread_closure(|| {
                let mut z: u64 = 1234;
                X.store(&raw mut z, SeqCst); // The other thread can read this value and then access `z` after it is deallocated.
                X.store(&raw mut Y, SeqCst);
            }),
            spawn_pthread_closure(|| {
                let ptr = X.load(SeqCst);
                miri_genmc_assume(!ptr.is_null());
                *ptr = 42; //~ ERROR: Undefined Behavior
            }),
        ];
        join_pthreads(ids);
        0
    }
}
