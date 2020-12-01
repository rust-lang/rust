// compile-flags: -Z print-type-sizes
// build-pass
// ignore-pass
// ^-- needed because `--pass check` does not emit the output needed.
//     FIXME: consider using an attribute instead of side-effects.

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
