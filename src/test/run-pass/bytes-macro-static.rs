// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static FOO: &'static [u8] = bytes!("hello, world");

pub fn main() {
    let b = match true {
        true => bytes!("test"),
        false => unreachable!()
    };

    assert_eq!(b, "test".as_bytes());
    assert_eq!(FOO, "hello, world".as_bytes());
}
