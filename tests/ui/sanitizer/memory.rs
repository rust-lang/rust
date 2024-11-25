//@ needs-sanitizer-support
//@ needs-sanitizer-memory
//
//@ revisions: unoptimized optimized
//
//@ [optimized]compile-flags: -Z sanitizer=memory -Zsanitizer-memory-track-origins -O
//@ [unoptimized]compile-flags: -Z sanitizer=memory -Zsanitizer-memory-track-origins
//
//@ run-fail
//@ check-run-results: MemorySanitizer: use-of-uninitialized-value
//@ check-run-results: Uninitialized value was created by an allocation
//@ check-run-results: in the stack frame
//
// This test case intentionally limits the usage of the std,
// since it will be linked with an uninstrumented version of it.

#![feature(core_intrinsics)]
#![feature(start)]
#![allow(invalid_value)]

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

#[start]
fn main(_: isize, _: *const *const u8) -> isize {
    let r = black_box(random as fn() -> [isize; 32])();
    xor(&r)
}
