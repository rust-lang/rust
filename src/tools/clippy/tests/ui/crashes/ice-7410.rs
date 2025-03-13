//@ check-pass
//@compile-flags: -Clink-arg=-nostartfiles
//@ignore-target: apple windows

#![crate_type = "lib"]
#![no_std]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::redundant_pattern_matching)]
#![allow(clippy::needless_else)]

use core::panic::PanicInfo;

struct S;

impl Drop for S {
    fn drop(&mut self) {}
}

pub fn main(argc: isize, argv: *const *const u8) -> isize {
    if let Some(_) = Some(S) {
    } else {
    }
    0
}
