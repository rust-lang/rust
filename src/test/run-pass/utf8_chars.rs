// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate extra;

use std::str;

pub fn main() {
    // Chars of 1, 2, 3, and 4 bytes
    let chs: ~[char] = ~['e', 'é', '€', '\U00010000'];
    let s: ~str = str::from_chars(chs);
    let schs: ~[char] = s.chars().collect();

    fail_unless!(s.len() == 10u);
    fail_unless!(s.char_len() == 4u);
    fail_unless!(schs.len() == 4u);
    fail_unless!(str::from_chars(schs) == s);
    fail_unless!(s.char_at(0u) == 'e');
    fail_unless!(s.char_at(1u) == 'é');

    fail_unless!((str::is_utf8(s.as_bytes())));
    // invalid prefix
    fail_unless!((!str::is_utf8([0x80_u8])));
    // invalid 2 byte prefix
    fail_unless!((!str::is_utf8([0xc0_u8])));
    fail_unless!((!str::is_utf8([0xc0_u8, 0x10_u8])));
    // invalid 3 byte prefix
    fail_unless!((!str::is_utf8([0xe0_u8])));
    fail_unless!((!str::is_utf8([0xe0_u8, 0x10_u8])));
    fail_unless!((!str::is_utf8([0xe0_u8, 0xff_u8, 0x10_u8])));
    // invalid 4 byte prefix
    fail_unless!((!str::is_utf8([0xf0_u8])));
    fail_unless!((!str::is_utf8([0xf0_u8, 0x10_u8])));
    fail_unless!((!str::is_utf8([0xf0_u8, 0xff_u8, 0x10_u8])));
    fail_unless!((!str::is_utf8([0xf0_u8, 0xff_u8, 0xff_u8, 0x10_u8])));

    let mut stack = ~"a×c€";
    fail_unless_eq!(stack.pop_char(), '€');
    fail_unless_eq!(stack.pop_char(), 'c');
    stack.push_char('u');
    fail_unless!(stack == ~"a×u");
    fail_unless_eq!(stack.shift_char(), 'a');
    fail_unless_eq!(stack.shift_char(), '×');
    stack.unshift_char('ß');
    fail_unless!(stack == ~"ßu");
}
