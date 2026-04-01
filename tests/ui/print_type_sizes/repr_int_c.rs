//@ compile-flags: -Z print-type-sizes --crate-type=lib
//@ build-pass

// This test makes sure that the tag is not grown for `repr(C)` or `repr(u8)`
// variants (see https://github.com/rust-lang/rust/issues/50098 for the original bug).

#![allow(dead_code)]

#[repr(C, u8)]
enum ReprCu8 {
    A(u16),
    B,
}

#[repr(u8)]
enum Repru8 {
    A(u16),
    B,
}
