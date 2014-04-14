// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]

use std::mem;

#[packed]
struct S4 {
    a: u8,
    b: [u8, .. 3],
}

#[packed]
struct S5 {
    a: u8,
    b: u32
}

#[packed]
struct S13_str {
    a: i64,
    b: f32,
    c: u8,
    d: ~str
}

enum Foo {
    Bar = 1,
    Baz = 2
}

#[packed]
struct S3_Foo {
    a: u8,
    b: u16,
    c: Foo
}

#[packed]
struct S7_Option {
    a: f32,
    b: u8,
    c: u16,
    d: Option<@f64>
}

// Placing packed structs in statics should work
static TEST_S4: S4 = S4 { a: 1, b: [2, 3, 4] };
static TEST_S5: S5 = S5 { a: 3, b: 67 };
static TEST_S3_Foo: S3_Foo = S3_Foo { a: 1, b: 2, c: Baz };


pub fn main() {
    assert_eq!(mem::size_of::<S4>(), 4);
    assert_eq!(mem::size_of::<S5>(), 5);
    assert_eq!(mem::size_of::<S13_str>(), 13 + mem::size_of::<~str>());
    assert_eq!(mem::size_of::<S3_Foo>(), 3 + mem::size_of::<Foo>());
    assert_eq!(mem::size_of::<S7_Option>(), 7 + mem::size_of::<Option<@f64>>());
}
