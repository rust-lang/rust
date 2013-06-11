// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;

use std::iterator::IteratorUtil;
use std::str;
use std::vec;

pub fn main() {
    // Chars of 1, 2, 3, and 4 bytes
    let chs: ~[char] = ~['e', 'é', '€', 0x10000 as char];
    let s: ~str = str::from_chars(chs);
    let schs: ~[char] = s.iter().collect();

    assert!(s.len() == 10u);
    assert!(s.char_len() == 4u);
    assert!(schs.len() == 4u);
    assert!(str::from_chars(schs) == s);
    assert!(s.char_at(0u) == 'e');
    assert!(s.char_at(1u) == 'é');

    assert!((str::is_utf8(s.as_bytes())));
    assert!((!str::is_utf8(~[0x80_u8])));
    assert!((!str::is_utf8(~[0xc0_u8])));
    assert!((!str::is_utf8(~[0xc0_u8, 0x10_u8])));

    let mut stack = ~"a×c€";
    assert_eq!(stack.pop_char(), '€');
    assert_eq!(stack.pop_char(), 'c');
    stack.push_char('u');
    assert!(stack == ~"a×u");
    assert_eq!(stack.shift_char(), 'a');
    assert_eq!(stack.shift_char(), '×');
    stack.unshift_char('ß');
    assert!(stack == ~"ßu");
}
