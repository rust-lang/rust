// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:lang-item-public.rs
// ignore-android

#![feature(start, no_std)]
#![no_std]

extern crate lang_item_public as lang_lib;

#[start]
fn main(_: isize, _: *const *const u8) -> isize {
    1_isize % 1_isize
}
