// needs-sanitizer-support
// needs-sanitizer-memory
// min-llvm-version: 14.0.0
//
// revisions: unoptimized optimized
//
// [optimized]compile-flags: -Z sanitizer=memory -Zsanitizer-memory-track-origins -O
// [unoptimized]compile-flags: -Z sanitizer=memory -Zsanitizer-memory-track-origins
//
// run-fail
// error-pattern: MemorySanitizer: use-of-uninitialized-value
// error-pattern: Uninitialized value was created by an allocation
// error-pattern: in the stack frame
//
// This test case intentionally limits the usage of the std,
// since it will be linked with an uninstrumented version of it.

#![feature(core_intrinsics)]
#![feature(start)]
#![feature(bench_black_box)]

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
