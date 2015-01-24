// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]
#![allow(unused_unsafe)]

use std::marker::Sync;

struct Foo {
    a: usize,
    b: *const ()
}

unsafe impl Sync for Foo {}

fn foo<T>(a: T) -> T {
    a
}

static BLOCK_INTEGRAL: usize = { 1 };
static BLOCK_EXPLICIT_UNIT: () = { () };
static BLOCK_IMPLICIT_UNIT: () = { };
static BLOCK_FLOAT: f64 = { 1.0 };
static BLOCK_ENUM: Option<usize> = { Some(100) };
static BLOCK_STRUCT: Foo = { Foo { a: 12, b: 0 as *const () } };
static BLOCK_UNSAFE: usize = unsafe { 1000 };

static BLOCK_FN_INFERRED: fn(usize) -> usize = { foo };

static BLOCK_FN: fn(usize) -> usize = { foo::<usize> };

static BLOCK_ENUM_CONSTRUCTOR: fn(usize) -> Option<usize> = { Some };

// FIXME #13972
// static BLOCK_UNSAFE_SAFE_PTR: &'static isize = unsafe { &*(0xdeadbeef as *const isize) };
// static BLOCK_UNSAFE_SAFE_PTR_2: &'static isize = unsafe {
//     const X: *const isize = 0xdeadbeef as *const isize;
//     &*X
// };

pub fn main() {
    assert_eq!(BLOCK_INTEGRAL, 1);
    assert_eq!(BLOCK_EXPLICIT_UNIT, ());
    assert_eq!(BLOCK_IMPLICIT_UNIT, ());
    assert_eq!(BLOCK_FLOAT, 1.0_f64);
    assert_eq!(BLOCK_STRUCT.a, 12);
    assert_eq!(BLOCK_STRUCT.b, 0 as *const ());
    assert_eq!(BLOCK_ENUM, Some(100));
    assert_eq!(BLOCK_UNSAFE, 1000);
    assert_eq!(BLOCK_FN_INFERRED(300), 300);
    assert_eq!(BLOCK_FN(300), 300);
    assert_eq!(BLOCK_ENUM_CONSTRUCTOR(200), Some(200));
    // FIXME #13972
    // assert_eq!(BLOCK_UNSAFE_SAFE_PTR as *const isize as usize, 0xdeadbeef_us);
    // assert_eq!(BLOCK_UNSAFE_SAFE_PTR_2 as *const isize as usize, 0xdeadbeef_us);
}
