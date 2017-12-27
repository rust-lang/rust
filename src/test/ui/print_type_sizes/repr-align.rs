// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z print-type-sizes
// must-compile-successfully

// This file illustrates how padding is handled: alignment
// requirements can lead to the introduction of padding, either before
// fields or at the end of the structure as a whole.
//
// It avoids using u64/i64 because on some targets that is only 4-byte
// aligned (while on most it is 8-byte aligned) and so the resulting
// padding and overall computed sizes can be quite different.
#![feature(attr_literals)]
#![feature(repr_align)]
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
