// Test to ensure that you cannot use `cast_into` to convert a `NumBuffer` to
// a bigger buffer.

//@ build-fail

#![feature(fmt_internals)]

extern crate core;

use core::fmt::NumBuffer;

fn main() {
    let mut x = NumBuffer::<u32>::new();
    let mut y = x.cast_into::<u64>();
    //~? ERROR: target `NumBuffer` size must be smaller or equal to source `NumBuffer` size
}
