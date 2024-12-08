//@ run-fail
//@ check-run-results
//@ exec-env:RUST_BACKTRACE=0
//@ normalize-stderr-test: "(core/src/panicking\.rs):[0-9]+:[0-9]+" -> "$1:$$LINE:$$COL"
//@ revisions: size align
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
