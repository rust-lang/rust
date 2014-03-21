// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::str;

pub fn main() {
    // Chars of 1, 2, 3, and 4 bytes
    let chs: Vec<char> = vec!('e', 'é', '€', '\U00010000');
    let s: ~str = str::from_chars(chs.as_slice());
    let schs: Vec<char> = s.chars().collect();

    assert!(s.len() == 10u);
    assert!(s.char_len() == 4u);
    assert!(schs.len() == 4u);
    assert!(str::from_chars(schs.as_slice()) == s);
    assert!(s.char_at(0u) == 'e');
    assert!(s.char_at(1u) == 'é');

    assert!((str::is_utf8(s.as_bytes())));
    // invalid prefix
    assert!((!str::is_utf8([0x80_u8])));
    // invalid 2 byte prefix
    assert!((!str::is_utf8([0xc0_u8])));
    assert!((!str::is_utf8([0xc0_u8, 0x10_u8])));
    // invalid 3 byte prefix
    assert!((!str::is_utf8([0xe0_u8])));
    assert!((!str::is_utf8([0xe0_u8, 0x10_u8])));
    assert!((!str::is_utf8([0xe0_u8, 0xff_u8, 0x10_u8])));
    // invalid 4 byte prefix
    assert!((!str::is_utf8([0xf0_u8])));
    assert!((!str::is_utf8([0xf0_u8, 0x10_u8])));
    assert!((!str::is_utf8([0xf0_u8, 0xff_u8, 0x10_u8])));
    assert!((!str::is_utf8([0xf0_u8, 0xff_u8, 0xff_u8, 0x10_u8])));

    let mut stack = ~"a×c€";
    assert_eq!(stack.pop_char(), Some('€'));
    assert_eq!(stack.pop_char(), Some('c'));
    stack.push_char('u');
    assert!(stack == ~"a×u");
    assert_eq!(stack.shift_char(), Some('a'));
    assert_eq!(stack.shift_char(), Some('×'));
    stack.unshift_char('ß');
    assert!(stack == ~"ßu");
}
