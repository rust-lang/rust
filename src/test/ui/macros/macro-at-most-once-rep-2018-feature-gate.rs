// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Feature gate test for macro_at_most_once_rep under 2018 edition.

// gate-test-macro_at_most_once_rep
// compile-flags: --edition=2018

macro_rules! foo {
    ($(a)?) => {}
    //~^ERROR using the `?` macro Kleene operator for
    //~|ERROR expected `*` or `+`
}

macro_rules! baz {
    ($(a),?) => {} //~ERROR expected `*` or `+`
}

macro_rules! barplus {
    ($(a)?+) => {}
    //~^ERROR using the `?` macro Kleene operator for
    //~|ERROR expected `*` or `+`
}

macro_rules! barstar {
    ($(a)?*) => {}
    //~^ERROR using the `?` macro Kleene operator for
    //~|ERROR expected `*` or `+`
}

pub fn main() {
    foo!();
    foo!(a);
    foo!(a?); //~ ERROR no rules expected the token `?`
    foo!(a?a); //~ ERROR no rules expected the token `?`
    foo!(a?a?a); //~ ERROR no rules expected the token `?`
}

