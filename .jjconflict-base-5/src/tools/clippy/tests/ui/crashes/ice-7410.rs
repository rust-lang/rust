//@ check-pass
//@compile-flags: -Clink-arg=-nostartfiles
//@ignore-target: windows

#![crate_type = "lib"]
#![no_std]
#![expect(clippy::needless_else, clippy::redundant_pattern_matching)]

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
