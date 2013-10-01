// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;

pub struct Escape<'self>(&'self str);

impl<'self> fmt::Default for Escape<'self> {
    fn fmt(s: &Escape<'self>, fmt: &mut fmt::Formatter) {
        // Because the internet is always right, turns out there's not that many
        // characters to escape: http://stackoverflow.com/questions/7381974
        let pile_o_bits = s.as_slice();
        let mut last = 0;
        for (i, ch) in s.byte_iter().enumerate() {
            match ch as char {
                '<' | '>' | '&' | '\'' | '"' => {
                    fmt.buf.write(pile_o_bits.slice(last, i).as_bytes());
                    let s = match ch as char {
                        '>' => "&gt;",
                        '<' => "&lt;",
                        '&' => "&amp;",
                        '\'' => "&#39;",
                        '"' => "&quot;",
                        _ => unreachable!()
                    };
                    fmt.buf.write(s.as_bytes());
                    last = i + 1;
                }
                _ => {}
            }
        }

        if last < s.len() {
            fmt.buf.write(pile_o_bits.slice_from(last).as_bytes());
        }
    }
}
