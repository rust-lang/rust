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

struct Foo {
    a: uint,
    b: *()
}

fn foo<T>(a: T) -> T {
    a
}

static BLOCK_INTEGRAL: uint = { 1 };
static BLOCK_EXPLICIT_UNIT: () = { () };
static BLOCK_IMPLICIT_UNIT: () = { };
static BLOCK_FLOAT: f64 = { 1.0 };
static BLOCK_ENUM: Option<uint> = { Some(100) };
static BLOCK_STRUCT: Foo = { Foo { a: 12, b: 0 as *() } };
static BLOCK_UNSAFE: uint = unsafe { 1000 };

// FIXME: #13970
// static BLOCK_FN_INFERRED: fn(uint) -> uint = { foo };

// FIXME: #13971
// static BLOCK_FN: fn(uint) -> uint = { foo::<uint> };

// FIXME: #13972
// static BLOCK_ENUM_CONSTRUCTOR: fn(uint) -> Option<uint> = { Some };

// FIXME: #13973
// static BLOCK_UNSAFE_SAFE_PTR: &'static int = unsafe { &*(0xdeadbeef as *int) };
// static BLOCK_UNSAFE_SAFE_PTR_2: &'static int = unsafe {
//     static X: *int = 0xdeadbeef as *int;
//     &*X
// };

pub fn main() {
    assert_eq!(BLOCK_INTEGRAL, 1);
    assert_eq!(BLOCK_EXPLICIT_UNIT, ());
    assert_eq!(BLOCK_IMPLICIT_UNIT, ());
    assert_eq!(BLOCK_FLOAT, 1.0_f64);
    assert_eq!(BLOCK_STRUCT.a, 12);
    assert_eq!(BLOCK_STRUCT.b, 0 as *());
    assert_eq!(BLOCK_ENUM, Some(100));
    assert_eq!(BLOCK_UNSAFE, 1000);

    // FIXME: #13970
    // assert_eq!(BLOCK_FN_INFERRED(300), 300);

    // FIXME: #13971
    // assert_eq!(BLOCK_FN(300), 300);

    // FIXME: #13972
    // assert_eq!(BLOCK_ENUM_CONSTRUCTOR(200), Some(200));

    // FIXME: #13973
    // assert_eq!(BLOCK_UNSAFE_SAFE_PTR as *int as uint, 0xdeadbeef_u);
    // assert_eq!(BLOCK_UNSAFE_SAFE_PTR_2 as *int as uint, 0xdeadbeef_u);
}
