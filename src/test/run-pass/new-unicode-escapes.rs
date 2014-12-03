// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    let s = "\u{2603}";
    assert_eq!(s, "☃");

    let s = "\u{2a10}\u{2A01}\u{2Aa0}";
    assert_eq!(s, "⨐⨁⪠");

    let s = "\\{20}";
    let mut correct_s = String::from_str("\\");
    correct_s.push_str("{20}");
    assert_eq!(s, correct_s.as_slice());
}
