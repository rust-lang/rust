// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(match_default_bindings)]

const CONST_REF: &[u8; 3] = b"foo";

trait Foo {
    const CONST_REF_DEFAULT: &'static [u8; 3] = b"bar";
    const CONST_REF: &'static [u8; 3];
}

impl Foo for i32 {
    const CONST_REF: &'static [u8; 3] = b"jjj";
}

impl Foo for i64 {
    const CONST_REF_DEFAULT: &'static [u8; 3] = b"ggg";
    const CONST_REF: &'static [u8; 3] = b"fff";
}

// Check that (associated and free) const references are not mistaken for a
// non-reference pattern (in which case they would be auto-dereferenced, making
// the types mismatched).

fn const_ref() -> bool {
    let f = b"foo";
    match f {
        CONST_REF => true,
        _ => false,
    }
}

fn associated_const_ref() -> bool {
    match (b"bar", b"jjj", b"ggg", b"fff") {
        (i32::CONST_REF_DEFAULT, i32::CONST_REF, i64::CONST_REF_DEFAULT, i64::CONST_REF) => true,
        _ => false,
    }
}

pub fn main() {
    assert!(const_ref());
    assert!(associated_const_ref());
}
