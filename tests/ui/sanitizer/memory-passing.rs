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
//@ run-pass
//
// This test case intentionally limits the usage of the std,
// since it will be linked with an uninstrumented version of it.

#![feature(core_intrinsics)]
#![allow(invalid_value)]
#![no_main]

use std::hint::black_box;

fn calling_black_box_on_zst_ok() {
    // It's OK to call black_box on a value of a zero-sized type, even if its
    // underlying the memory location is uninitialized. For non-zero-sized types,
    // this would be an MSAN error.
    let zst = ();
    black_box(zst);
}

#[no_mangle]
extern "C" fn main(_argc: std::ffi::c_int, _argv: *const *const u8) -> std::ffi::c_int {
    calling_black_box_on_zst_ok();
    0
}
