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

#[start]
fn main(_: isize, _: *const *const u8) -> isize {
    random();
    0
}
