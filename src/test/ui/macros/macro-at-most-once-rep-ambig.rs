// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The logic for parsing Kleene operators in macros has a special case to disambiguate `?`.
// Specifically, `$(pat)?` is the ZeroOrOne operator whereas `$(pat)?+` or `$(pat)?*` are the
// ZeroOrMore and OneOrMore operators using `?` as a separator. These tests are intended to
// exercise that logic in the macro parser.
//
// Moreover, we also throw in some tests for using a separator with `?`, which is meaningless but
// included for consistency with `+` and `*`.
//
// This test focuses on error cases.

#![feature(macro_at_most_once_rep)]

macro_rules! foo {
    ($(a)?) => {}
}

macro_rules! baz {
    ($(a),?) => {} // comma separator is meaningless for `?`
}

macro_rules! barplus {
    ($(a)?+) => {}
}

macro_rules! barstar {
    ($(a)?*) => {}
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
    barplus!(); //~ ERROR unexpected end of macro invocation
    barplus!(a?); //~ ERROR unexpected end of macro invocation
    barstar!(a?); //~ ERROR unexpected end of macro invocation
}
