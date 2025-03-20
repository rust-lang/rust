//@ needs-sanitizer-support
//@ needs-sanitizer-memory
//
//@ compile-flags: -C unsafe-allow-abi-mismatch=sanitizer
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
#![allow(invalid_value)]
#![no_main]

use std::hint::black_box;
use std::mem::MaybeUninit;

#[inline(never)]
#[no_mangle]
fn random() -> [isize; 32] {
    let r = MaybeUninit::uninit();
    // Avoid optimizing everything out.
    unsafe { std::intrinsics::volatile_load(r.as_ptr()) }
}

#[inline(never)]
#[no_mangle]
fn xor(a: &[isize]) -> isize {
    let mut s = 0;
    for i in 0..a.len() {
        s = s ^ a[i];
    }
    s
}

#[no_mangle]
extern "C" fn main(_argc: std::ffi::c_int, _argv: *const *const u8) -> std::ffi::c_int {
    let r = black_box(random as fn() -> [isize; 32])();
    xor(&r) as std::ffi::c_int
}
