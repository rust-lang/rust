// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_at_most_once_rep)]

macro_rules! foo {
    ($(a)?) => {}
}

macro_rules! baz {
    ($(a),?) => {} // comma separator is meaningless for `?`
}

macro_rules! bar {
    ($(a)?+) => {}
}

pub fn main() {
    foo!(a?a?a); //~ ERROR no rules expected the token `?`
    foo!(a?a); //~ ERROR no rules expected the token `?`
    foo!(a?); //~ ERROR no rules expected the token `?`
    baz!(a?a?a); //~ ERROR no rules expected the token `?`
    baz!(a?a); //~ ERROR no rules expected the token `?`
    baz!(a?); //~ ERROR no rules expected the token `?`
    baz!(a,); //~ ERROR unexpected end of macro invocation
    baz!(a?a?a,); //~ ERROR no rules expected the token `?`
    baz!(a?a,); //~ ERROR no rules expected the token `?`
    baz!(a?,); //~ ERROR no rules expected the token `?`
    bar!(); //~ ERROR unexpected end of macro invocation
    bar!(a?); //~ ERROR unexpected end of macro invocation
}
