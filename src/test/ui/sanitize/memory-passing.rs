// needs-sanitizer-support
// needs-sanitizer-memory
//
// revisions: unoptimized optimized
//
// [optimized]compile-flags: -Z sanitizer=memory -Zsanitizer-memory-track-origins -O
// [unoptimized]compile-flags: -Z sanitizer=memory -Zsanitizer-memory-track-origins
//
// run-pass
//
// This test case intentionally limits the usage of the std,
// since it will be linked with an uninstrumented version of it.

#![feature(core_intrinsics)]
#![feature(start)]
#![allow(invalid_value)]

use std::hint::black_box;

fn calling_black_box_on_zst_ok() {
    // It's OK to call black_box on a value of a zero-sized type, even if its
    // underlying the memory location is uninitialized. For non-zero-sized types,
    // this would be an MSAN error.
    let zst = ();
    black_box(zst);
}

#[start]
fn main(_: isize, _: *const *const u8) -> isize {
    calling_black_box_on_zst_ok();
    0
}
