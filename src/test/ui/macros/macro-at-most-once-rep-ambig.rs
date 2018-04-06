// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests the behavior of various Kleene operators in macros with respect to `?` terminals. In
// particular, `?` in the position of a separator and of a Kleene operator is tested.

#![feature(macro_at_most_once_rep)]

// should match `` and `a`
macro_rules! foo {
    ($(a)?) => {}
}

macro_rules! baz {
    ($(a),?) => {} //~ ERROR `?` macro repetition does not allow a separator
}

// should match `+` and `a+`
macro_rules! barplus {
    ($(a)?+) => {}
}

// should match `*` and `a*`
macro_rules! barstar {
    ($(a)?*) => {}
}

pub fn main() {
    foo!(a?a?a); //~ ERROR no rules expected the token `?`
    foo!(a?a); //~ ERROR no rules expected the token `?`
    foo!(a?); //~ ERROR no rules expected the token `?`
    barplus!(); //~ ERROR unexpected end of macro invocation
    barstar!(); //~ ERROR unexpected end of macro invocation
    barplus!(a?); //~ ERROR no rules expected the token `?`
    barplus!(a); //~ ERROR unexpected end of macro invocation
    barstar!(a?); //~ ERROR no rules expected the token `?`
    barstar!(a); //~ ERROR unexpected end of macro invocation
    barplus!(+); // ok
    barstar!(*); // ok
    barplus!(a+); // ok
    barstar!(a*); // ok
}
