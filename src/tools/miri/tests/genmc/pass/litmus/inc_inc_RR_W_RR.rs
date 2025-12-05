//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::ffi::c_void;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;

static X: AtomicU64 = AtomicU64::new(0);

static mut A: u64 = 0;
static mut B: u64 = 0;
static mut C: u64 = 0;
static mut D: u64 = 0;

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    let thread_order = [thread_1, thread_2, thread_3, thread_4, thread_5];
    let ids = unsafe { spawn_pthreads_no_params(thread_order) };
    unsafe { join_pthreads(ids) };

    if unsafe { A == 42 && B == 2 && C == 1 && D == 42 } {
        std::process::abort();
    }

    0
}

pub extern "C" fn thread_1(_value: *mut c_void) -> *mut c_void {
    X.fetch_add(1, Relaxed);
    std::ptr::null_mut()
}

pub extern "C" fn thread_2(_value: *mut c_void) -> *mut c_void {
    X.fetch_add(1, Relaxed);
    std::ptr::null_mut()
}

pub extern "C" fn thread_3(_value: *mut c_void) -> *mut c_void {
    unsafe {
        A = X.load(Relaxed);
        B = X.load(Relaxed);
    }
    std::ptr::null_mut()
}

pub extern "C" fn thread_4(_value: *mut c_void) -> *mut c_void {
    X.store(42, Relaxed);
    std::ptr::null_mut()
}

pub extern "C" fn thread_5(_value: *mut c_void) -> *mut c_void {
    unsafe {
        C = X.load(Relaxed);
        D = X.load(Relaxed);
    }
    std::ptr::null_mut()
}
