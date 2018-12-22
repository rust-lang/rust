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
// compile-pass

// This file illustrates how packing is handled; it should cause
// the elimination of padding that would normally be introduced
// to satisfy alignment desirata.
//
// It avoids using u64/i64 because on some targets that is only 4-byte
// aligned (while on most it is 8-byte aligned) and so the resulting
// padding and overall computed sizes can be quite different.

#![allow(dead_code)]
#![feature(start)]

#[derive(Default)]
#[repr(packed)]
struct Packed1 {
    a: u8,
    b: u8,
    g: i32,
    c: u8,
    h: i16,
    d: u8,
}

#[derive(Default)]
#[repr(packed(2))]
struct Packed2 {
    a: u8,
    b: u8,
    g: i32,
    c: u8,
    h: i16,
    d: u8,
}

#[derive(Default)]
#[repr(packed(2))]
#[repr(C)]
struct Packed2C {
    a: u8,
    b: u8,
    g: i32,
    c: u8,
    h: i16,
    d: u8,
}

#[derive(Default)]
struct Padded {
    a: u8,
    b: u8,
    g: i32,
    c: u8,
    h: i16,
    d: u8,
}

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    let _c: Packed1 = Default::default();
    let _d: Packed2 = Default::default();
    let _e: Packed2C = Default::default();
    let _f: Padded = Default::default();
    0
}
