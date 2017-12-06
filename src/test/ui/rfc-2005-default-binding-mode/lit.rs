// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(match_default_bindings)]

// FIXME(tschottdorf): we want these to compile, but they don't.

fn with_str() {
    let s: &'static str = "abc";

    match &s {
            "abc" => true, //~ ERROR mismatched types
            _ => panic!(),
    };
}

fn with_bytes() {
    let s: &'static [u8] = b"abc";

    match &s {
        b"abc" => true, //~ ERROR mismatched types
        _ => panic!(),
    };
}

pub fn main() {
    with_str();
    with_bytes();
}
