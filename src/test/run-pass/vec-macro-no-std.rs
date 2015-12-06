// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(lang_items, start, libc, collections)]
#![no_std]

extern crate std as other;

extern crate libc;

#[macro_use]
extern crate collections;

use collections::vec::Vec;

// Issue #16806

#[start]
fn start(_argc: isize, _argv: *const *const u8) -> isize {
    let x: Vec<u8> = vec![0, 1, 2];
    match x.last() {
        Some(&2) => (),
        _ => panic!(),
    }
    0
}
