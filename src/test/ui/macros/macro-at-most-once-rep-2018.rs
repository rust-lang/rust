// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that `?` is a Kleene op and not a macro separator in the 2018 edition.

// edition:2018

macro_rules! foo {
    ($(a)?) => {};
}

macro_rules! baz {
    ($(a),?) => {}; //~ERROR the `?` macro repetition operator
}

macro_rules! barplus {
    ($(a)?+) => {}; // ok. matches "a+" and "+"
}

macro_rules! barstar {
    ($(a)?*) => {}; // ok. matches "a*" and "*"
}

pub fn main() {
    foo!();
    foo!(a);
    foo!(a?); //~ ERROR no rules expected the token `?`
    foo!(a?a); //~ ERROR no rules expected the token `?`
    foo!(a?a?a); //~ ERROR no rules expected the token `?`

    barplus!(); //~ERROR unexpected end of macro invocation
    barplus!(a); //~ERROR unexpected end of macro invocation
    barplus!(a?); //~ ERROR no rules expected the token `?`
    barplus!(a?a); //~ ERROR no rules expected the token `?`
    barplus!(a);
    barplus!(+);

    barstar!(); //~ERROR unexpected end of macro invocation
    barstar!(a); //~ERROR unexpected end of macro invocation
    barstar!(a?); //~ ERROR no rules expected the token `?`
    barstar!(a?a); //~ ERROR no rules expected the token `?`
    barstar!(a*);
    barstar!(*);
}
