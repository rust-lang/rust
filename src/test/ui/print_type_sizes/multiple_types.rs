// compile-flags: -Z print-type-sizes
// build-pass (FIXME(62277): could be check-pass?)

// This file illustrates that when multiple structural types occur in
// a function, every one of them is included in the output.

#![feature(start)]

pub struct SevenBytes([u8;  7]);
pub struct FiftyBytes([u8; 50]);

pub enum Enum {
    Small(SevenBytes),
    Large(FiftyBytes),
}

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    let _e: Enum;
    let _f: FiftyBytes;
    let _s: SevenBytes;
    0
}
