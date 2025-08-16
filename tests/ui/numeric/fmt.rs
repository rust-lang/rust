// Test to ensure that if you use a `NumBuffer` with a too small integer, it
// will fail at compilation time.

//@ build-fail
//@ ignore-pass

#![feature(int_format_into)]

use std::fmt::NumBuffer;

fn main() {
    let x = 0u32;
    let mut buf = NumBuffer::<u8>::new();
    x.format_into(&mut buf);
}

//~? ERROR evaluation panicked
