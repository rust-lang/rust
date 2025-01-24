//@ needs-sanitizer-support
//@ needs-sanitizer-memory
//
//@ revisions: unoptimized optimized
//
//@ [optimized]compile-flags: -Z sanitizer=memory -Zsanitizer-memory-track-origins -O
//@ [unoptimized]compile-flags: -Z sanitizer=memory -Zsanitizer-memory-track-origins
//
//@ run-fail
//@ error-pattern: MemorySanitizer: use-of-uninitialized-value
//@ error-pattern: Uninitialized value was created by an allocation
//@ error-pattern: in the stack frame
//
// This test case intentionally limits the usage of the std,
// since it will be linked with an uninstrumented version of it.

#![feature(core_intrinsics)]
#![no_main]

use std::hint::black_box;
use std::mem::MaybeUninit;

#[inline(never)]
#[no_mangle]
#[allow(invalid_value)]
fn random() -> char {
    let r = unsafe { MaybeUninit::uninit().assume_init() };
    // Avoid optimizing everything out.
    black_box(r)
}

#[no_mangle]
extern "C" fn main(_argc: std::ffi::c_int, _argv: *const *const u8) -> std::ffi::c_int {
    random();
    0
}
