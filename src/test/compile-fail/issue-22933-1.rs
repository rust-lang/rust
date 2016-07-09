// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]
#![allow(warnings)]

struct CNFParser {
    token: char,
}

impl CNFParser {
    fn is_whitespace(c: char) -> bool {
        c == ' ' || c == '\n'
    }

    fn consume_whitespace(&mut self) {
        self.consume_while(&(CNFParser::is_whitespace))
    }

    fn consume_while(&mut self, p: &Fn(char) -> bool) {
        while p(self.token) {
            return
        }
    }
}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
