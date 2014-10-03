// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:roman_numerals.rs
// ignore-stage1
// ignore-android

#![feature(phase)]

#[phase(plugin, link)]
extern crate roman_numerals;

pub fn main() {
    assert_eq!(rn!(MMXV), 2015);
    assert_eq!(rn!(MCMXCIX), 1999);
    assert_eq!(rn!(XXV), 25);
    assert_eq!(rn!(MDCLXVI), 1666);
    assert_eq!(rn!(MMMDCCCLXXXVIII), 3888);
    assert_eq!(rn!(MMXIV), 2014);
}
