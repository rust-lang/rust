// compile-flags: -Z print-type-sizes
// build-pass (FIXME(62277): could be check-pass?)
// ignore-pass
// ^-- needed because `--pass check` does not emit the output needed.
//     FIXME: consider using an attribute instead of side-effects.

// This file illustrates how padding is handled: alignment
// requirements can lead to the introduction of padding, either before
// fields or at the end of the structure as a whole.
//
// It avoids using u64/i64 because on some targets that is only 4-byte
// aligned (while on most it is 8-byte aligned) and so the resulting
// padding and overall computed sizes can be quite different.
#![feature(start)]
#![allow(dead_code)]

#[repr(align(16))]
#[derive(Default)]
struct A(i32);

enum E {
    A(i32),
    B(A)
}

#[derive(Default)]
struct S {
    a: i32,
    b: i32,
    c: A,
    d: i8,
}

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    let _s: S = Default::default();
    0
}
