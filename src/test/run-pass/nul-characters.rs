// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main()
{
    let all_nuls1 = "\0\x00\u0000\U00000000";
    let all_nuls2 = "\U00000000\u0000\x00\0";
    let all_nuls3 = "\u0000\U00000000\x00\0";
    let all_nuls4 = "\x00\u0000\0\U00000000";

    // sizes for two should suffice
    fail_unless_eq!(all_nuls1.len(), 4);
    fail_unless_eq!(all_nuls2.len(), 4);

    // string equality should pass between the strings
    fail_unless_eq!(all_nuls1, all_nuls2);
    fail_unless_eq!(all_nuls2, all_nuls3);
    fail_unless_eq!(all_nuls3, all_nuls4);

    // all extracted characters in all_nuls are equivalent to each other
    for c1 in all_nuls1.chars()
    {
        for c2 in all_nuls1.chars()
        {
            fail_unless_eq!(c1,c2);
        }
    }

    // testing equality between explicit character literals
    fail_unless_eq!('\0', '\x00');
    fail_unless_eq!('\u0000', '\x00');
    fail_unless_eq!('\u0000', '\U00000000');

    // NUL characters should make a difference
    fail_unless!("Hello World" != "Hello \0World");
    fail_unless!("Hello World" != "Hello World\0");
}
