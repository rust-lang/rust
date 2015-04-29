// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: This was copied from core/str/mod.rs because it is currently unstable.
pub fn char_at(s: &str, byte: usize) -> char {
    s[byte..].chars().next().unwrap()
}

// FIXME: This was copied from core/str/mod.rs because it is currently unstable.
#[inline]
pub fn slice_shift_char(s: &str) -> Option<(char, &str)> {
    if s.is_empty() {
        None
    } else {
        let ch = char_at(s, 0);
        Some((ch, &s[ch.len_utf8()..]))
    }
}
