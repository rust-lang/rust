// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that generating drop glue for Box<str> doesn't ICE

fn f(s: Box<str>) -> Box<str> {
    s
}

fn main() {
    // There is currently no safe way to construct a `Box<str>`, so improvise
    let box_arr: Box<[u8]> = box ['h' as u8, 'e' as u8, 'l' as u8, 'l' as u8, 'o' as u8];
    let box_str: Box<str> = unsafe { std::mem::transmute(box_arr) };
    assert_eq!(box_str.as_slice(), "hello");
    f(box_str);
}
