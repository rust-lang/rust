//! Validates the correct behavior of writing a `bool` value using `std::ptr::write`.
//!
//! This test addresses historical concerns regarding the internal representation of `bool`
//! (e.g., as `i1` in LLVM versus its byte-aligned memory layout) and checks that
//! `ptr::write` correctly handles this type without issues, confirming its memory
//! behavior is as expected.

//@ run-pass

use std::ptr;

pub fn main() {
    unsafe {
        let mut x: bool = false;
        // this line breaks it
        ptr::write(&mut x, false);
    }
}
