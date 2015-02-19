// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15679

use std::str;

pub fn main() {
    // Chars of 1, 2, 3, and 4 bytes
    let chs: Vec<char> = vec!('e', 'é', '€', '\U00010000');
    let s: String = chs.iter().cloned().collect();
    let schs: Vec<char> = s.chars().collect();

    assert!(s.len() == 10_usize);
    assert!(s.chars().count() == 4_usize);
    assert!(schs.len() == 4_usize);
    assert!(schs.iter().cloned().collect::<String>() == s);
    assert!(s.char_at(0_usize) == 'e');
    assert!(s.char_at(1_usize) == 'é');

    assert!((str::from_utf8(s.as_bytes()).is_ok()));
    // invalid prefix
    assert!((!str::from_utf8(&[0x80_u8]).is_ok()));
    // invalid 2 byte prefix
    assert!((!str::from_utf8(&[0xc0_u8]).is_ok()));
    assert!((!str::from_utf8(&[0xc0_u8, 0x10_u8]).is_ok()));
    // invalid 3 byte prefix
    assert!((!str::from_utf8(&[0xe0_u8]).is_ok()));
    assert!((!str::from_utf8(&[0xe0_u8, 0x10_u8]).is_ok()));
    assert!((!str::from_utf8(&[0xe0_u8, 0xff_u8, 0x10_u8]).is_ok()));
    // invalid 4 byte prefix
    assert!((!str::from_utf8(&[0xf0_u8]).is_ok()));
    assert!((!str::from_utf8(&[0xf0_u8, 0x10_u8]).is_ok()));
    assert!((!str::from_utf8(&[0xf0_u8, 0xff_u8, 0x10_u8]).is_ok()));
    assert!((!str::from_utf8(&[0xf0_u8, 0xff_u8, 0xff_u8, 0x10_u8]).is_ok()));
}
