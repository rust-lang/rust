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
//@ [optimized]error-pattern: Uninitialized value was created by an allocation
//@ [optimized]error-pattern: in the stack frame
//
// FIXME the unoptimized case actually has that text in the output too, per
// <https://github.com/rust-lang/rust/pull/138759#issuecomment-3037186707>
// but doesn't seem to be getting picked up for some reason. For now we don't
// check for that part, since it's still testing that memory sanitizer reported
// a use of an uninitialized value, which is the critical part.
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
