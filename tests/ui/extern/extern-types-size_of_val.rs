// run-fail
// check-run-results
// revisions: size align
// normalize-stderr-test "panicking\.rs:\d+:\d+:" -> "panicking.rs:"
#![feature(extern_types)]

use std::mem::{align_of_val, size_of_val};

extern "C" {
    type A;
}

fn main() {
    let x: &A = unsafe { &*(1usize as *const A) };

    // These don't have a dynamic size, so this should panic.
    if cfg!(size) {
        assert_eq!(size_of_val(x), 0);
    } else {
        assert_eq!(align_of_val(x), 1);
    }
}
