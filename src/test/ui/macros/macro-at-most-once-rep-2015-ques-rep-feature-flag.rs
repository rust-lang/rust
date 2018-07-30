// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test behavior of `?` macro _kleene op_ under the 2015 edition. Namely, it doesn't exist, even
// with the feature flag.

// gate-test-macro_at_most_once_rep
// compile-flags: --edition=2015

#![feature(macro_at_most_once_rep)]

macro_rules! bar {
    ($(a)?) => {} //~ERROR expected `*` or `+`
}

macro_rules! baz {
    ($(a),?) => {} //~ERROR expected `*` or `+`
}

fn main() {}

