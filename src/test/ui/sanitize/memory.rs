// needs-sanitizer-support
// needs-sanitizer-memory
//
// compile-flags: -Z sanitizer=memory -Zsanitizer-memory-track-origins -O
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
#![allow(invalid_value)]

use std::hint::black_box;
use std::mem::MaybeUninit;

#[inline(never)]
#[no_mangle]
fn random() -> [isize; 32] {
    let r = unsafe { MaybeUninit::uninit().assume_init() };
    // Avoid optimizing everything out.
    black_box(r)
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
    let r = random();
    xor(&r)
}
