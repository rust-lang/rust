// compile-flags: -Z print-type-sizes
// build-pass

// This file illustrates two things:
//
// 1. Only types that appear in a monomorphized function appear in the
//    print-type-sizes output, and
//
// 2. For an enum, the print-type-sizes output will also include the
//    size of each variant.

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
    0
}
