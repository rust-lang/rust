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

// This file illustrates how enums with a non-null field are handled,
// modelled after cases like `Option<&u32>` and such.
//
// It uses NonZero directly, rather than `&_` or `Unique<_>`, because
// the test is not set up to deal with target-dependent pointer width.
//
// It avoids using u64/i64 because on some targets that is only 4-byte
// aligned (while on most it is 8-byte aligned) and so the resulting
// padding and overall computed sizes can be quite different.

#![feature(nonzero)]
#![allow(dead_code)]

extern crate core;
use core::nonzero::{NonZero, Zeroable};

pub enum MyOption<T> { None, Some(T) }

impl<T> Default for MyOption<T> {
    fn default() -> Self { MyOption::None }
}

pub enum EmbeddedDiscr {
    None,
    Record { pre: u8, val: NonZero<u32>, post: u16 },
}

impl Default for EmbeddedDiscr {
    fn default() -> Self { EmbeddedDiscr::None }
}

#[derive(Default)]
pub struct IndirectNonZero<T: Zeroable> {
    pre: u8,
    nested: NestedNonZero<T>,
    post: u16,
}

pub struct NestedNonZero<T: Zeroable> {
    pre: u8,
    val: NonZero<T>,
    post: u16,
}

impl<T: Zeroable+Default> Default for NestedNonZero<T> {
    fn default() -> Self {
        unsafe {
            NestedNonZero { pre: 0, val: NonZero::new(Default::default()), post: 0 }
        }
    }
}

pub fn main() {
    let _x: MyOption<NonZero<u32>> = Default::default();
    let _y: EmbeddedDiscr = Default::default();
    let _z: MyOption<IndirectNonZero<u32>> = Default::default();
}
