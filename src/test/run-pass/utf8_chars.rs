// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod std;

pub fn main() {
    // Chars of 1, 2, 3, and 4 bytes
    let chs: ~[char] = ~['e', 'é', '€', 0x10000 as char];
    let s: ~str = str::from_chars(chs);

    fail_unless!((str::len(s) == 10u));
    fail_unless!((str::char_len(s) == 4u));
    fail_unless!((vec::len(str::chars(s)) == 4u));
    fail_unless!((str::from_chars(str::chars(s)) == s));
    fail_unless!((str::char_at(s, 0u) == 'e'));
    fail_unless!((str::char_at(s, 1u) == 'é'));

    fail_unless!((str::is_utf8(str::to_bytes(s))));
    fail_unless!((!str::is_utf8(~[0x80_u8])));
    fail_unless!((!str::is_utf8(~[0xc0_u8])));
    fail_unless!((!str::is_utf8(~[0xc0_u8, 0x10_u8])));

    let mut stack = ~"a×c€";
    fail_unless!((str::pop_char(&mut stack) == '€'));
    fail_unless!((str::pop_char(&mut stack) == 'c'));
    str::push_char(&mut stack, 'u');
    fail_unless!((stack == ~"a×u"));
    fail_unless!((str::shift_char(&mut stack) == 'a'));
    fail_unless!((str::shift_char(&mut stack) == '×'));
    str::unshift_char(&mut stack, 'ß');
    fail_unless!((stack == ~"ßu"));
}
