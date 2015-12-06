// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(lang_items, start, collections)]
#![no_std]

extern crate std as other;

#[macro_use] extern crate collections;

#[start]
fn start(_argc: isize, _argv: *const *const u8) -> isize {
    for _ in [1,2,3].iter() { }
    0
}
