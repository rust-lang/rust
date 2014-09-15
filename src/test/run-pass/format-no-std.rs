// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(phase)]
#![deny(warnings)]
#![no_std]

#[phase(plugin, link)]
extern crate core;

#[phase(plugin, link)]
extern crate collections;

extern crate native;

use core::str::Str;

pub fn main() {
    let s = format!("{}", 1i);
    assert_eq!(s.as_slice(), "1");

    let s = format!("test");
    assert_eq!(s.as_slice(), "test");

    let s = format!("{test}", test=3i);
    assert_eq!(s.as_slice(), "3");

    let s = format!("hello {}", "world");
    assert_eq!(s.as_slice(), "hello world");
}
