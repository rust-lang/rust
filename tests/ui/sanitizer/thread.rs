// Verifies that ThreadSanitizer is able to detect a data race in heap allocated
// memory block.
//
// Test case minimizes the use of the standard library to avoid its ambiguous
// status with respect to instrumentation (it could vary depending on whatever
// a function call is inlined or not).
//
// The conflicting data access is de-facto synchronized with a special TSAN
// barrier, which does not introduce synchronization from TSAN perspective, but
// is necessary to make the test robust. Without the barrier data race detection
// would occasionally fail, making test flaky.
//
//@ needs-sanitizer-support
//@ needs-sanitizer-thread
//
//@ compile-flags: -Z sanitizer=thread -O -C unsafe-allow-abi-mismatch=sanitizer
//
//@ run-fail-or-crash
//@ error-pattern: WARNING: ThreadSanitizer: data race
//@ error-pattern: Location is heap block of size 4
//@ error-pattern: allocated by main thread

#![feature(rustc_private)]
extern crate libc;

use std::mem;
use std::ptr;

static mut BARRIER: u64 = 0;

extern "C" {
    fn __tsan_testonly_barrier_init(barrier: *mut u64, count: u32);
    fn __tsan_testonly_barrier_wait(barrier: *mut u64);
}

extern "C" fn start(c: *mut libc::c_void) -> *mut libc::c_void {
    unsafe {
        let c: *mut u32 = c.cast();
        *c += 1;
        __tsan_testonly_barrier_wait(&raw mut BARRIER);
        ptr::null_mut()
    }
}

fn main() {
    unsafe {
        __tsan_testonly_barrier_init(&raw mut BARRIER, 2);
        let c: *mut u32 = Box::into_raw(Box::new(1));
        let mut t: libc::pthread_t = mem::zeroed();
        libc::pthread_create(&mut t, ptr::null(), start, c.cast());
        __tsan_testonly_barrier_wait(&raw mut BARRIER);
        *c += 1;
        libc::pthread_join(t, ptr::null_mut());
        Box::from_raw(c);
    }
}
