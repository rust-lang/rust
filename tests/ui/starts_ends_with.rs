// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

fn main() {}

#[allow(clippy::unnecessary_operation)]
fn starts_with() {
    "".chars().next() == Some(' ');
    Some(' ') != "".chars().next();
}

fn chars_cmp_with_unwrap() {
    let s = String::from("foo");
    if s.chars().next().unwrap() == 'f' {
        // s.starts_with('f')
        // Nothing here
    }
    if s.chars().next_back().unwrap() == 'o' {
        // s.ends_with('o')
        // Nothing here
    }
    if s.chars().last().unwrap() == 'o' {
        // s.ends_with('o')
        // Nothing here
    }
    if s.chars().next().unwrap() != 'f' {
        // !s.starts_with('f')
        // Nothing here
    }
    if s.chars().next_back().unwrap() != 'o' {
        // !s.ends_with('o')
        // Nothing here
    }
    if s.chars().last().unwrap() != 'o' {
        // !s.ends_with('o')
        // Nothing here
    }
}

#[allow(clippy::unnecessary_operation)]
fn ends_with() {
    "".chars().last() == Some(' ');
    Some(' ') != "".chars().last();
    "".chars().next_back() == Some(' ');
    Some(' ') != "".chars().next_back();
}
