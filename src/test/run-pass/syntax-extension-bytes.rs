// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static static_vec: &'static [u8] = bytes!("abc", 0xFF, '!');

pub fn main() {
    let vec = bytes!("abc");
    let expected: &[u8] = &[97_u8, 98_u8, 99_u8];
    assert_eq!(vec, expected);

    let vec = bytes!("null", 0);
    let expected: &[u8] = &[110_u8, 117_u8, 108_u8, 108_u8, 0_u8];
    assert_eq!(vec, expected);

    let vec = bytes!(' ', " ", 32, 32u8);
    let expected: &[u8] = &[32_u8, 32_u8, 32_u8, 32_u8];
    assert_eq!(vec, expected);

    let expected: &[u8] = &[97_u8, 98_u8, 99_u8, 255_u8, 33_u8];
    assert_eq!(static_vec, expected);
}
