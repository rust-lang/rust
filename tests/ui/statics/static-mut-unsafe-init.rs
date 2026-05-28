//! regression test for https://github.com/rust-lang/rust/issues/17450
//@ build-pass
#![allow(dead_code)]

static mut X: isize = 3;
static mut Y: isize = unsafe { X };

fn main() {}
