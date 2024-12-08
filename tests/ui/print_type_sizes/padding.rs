//@ compile-flags: -Z print-type-sizes --crate-type=lib
//@ build-pass

// This file illustrates how padding is handled: alignment
// requirements can lead to the introduction of padding, either before
// fields or at the end of the structure as a whole.
//
// It avoids using u64/i64 because on some targets that is only 4-byte
// aligned (while on most it is 8-byte aligned) and so the resulting
// padding and overall computed sizes can be quite different.

#![allow(dead_code)]

struct S {
    a: bool,
    b: bool,
    g: i32,
}

enum E1 {
    A(i32, i8),
    B(S),
}

enum E2 {
    A(i8, i32),
    B(S),
}
