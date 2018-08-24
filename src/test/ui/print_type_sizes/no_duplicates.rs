// compile-flags: -Z print-type-sizes
// compile-pass

// This file illustrates that when the same type occurs repeatedly
// (even if multiple functions), it is only printed once in the
// print-type-sizes output.

#![feature(start)]

pub struct SevenBytes([u8; 7]);

pub fn f1() {
    let _s: SevenBytes = SevenBytes([0; 7]);
}

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    let _s: SevenBytes = SevenBytes([0; 7]);
    0
}
