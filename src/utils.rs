// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#[inline]
pub fn prev_char(s: &str, mut i: usize) -> usize {
    if i == 0 { return 0; }

    i -= 1;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[inline]
pub fn next_char(s: &str, mut i: usize) -> usize {
    if i >= s.len() { return s.len(); }

    while !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

#[inline]
pub fn make_indent(width: usize) -> String {
    let mut indent = String::with_capacity(width);
    for _ in 0..width {
        indent.push(' ')
    }
    indent
}
