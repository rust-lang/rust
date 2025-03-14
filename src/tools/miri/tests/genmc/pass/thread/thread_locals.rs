//@compile-flags: -Zmiri-ignore-leaks -Zmiri-genmc -Zmiri-disable-stacked-borrows

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::alloc::{Layout, alloc};
use std::cell::Cell;
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;

static X: AtomicPtr<u64> = AtomicPtr::new(std::ptr::null_mut());

thread_local! {
    static R: Cell<*mut u64> = Cell::new(std::ptr::null_mut());
}

pub unsafe fn malloc() -> *mut u64 {
    alloc(Layout::new::<u64>()) as *mut u64
}

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    // FIXME(genmc,HACK): remove this initializing write once Miri-GenMC supports mixed atomic-non-atomic accesses.
    X.store(std::ptr::null_mut(), SeqCst);

    unsafe {
        spawn_pthread_closure(|| {
            R.set(malloc());
            let r_ptr = R.get();
            let _ = X.compare_exchange(std::ptr::null_mut(), r_ptr, SeqCst, SeqCst);
        });
        spawn_pthread_closure(|| {
            R.set(malloc());
        });
        spawn_pthread_closure(|| {
            R.set(malloc());
            let r_ptr = R.get();
            let _ = X.compare_exchange(std::ptr::null_mut(), r_ptr, SeqCst, SeqCst);
        });

        0
    }
}
