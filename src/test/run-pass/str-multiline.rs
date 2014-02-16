// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



extern crate extra;

pub fn main() {
    let a: ~str = ~"this \
is a test";
    let b: ~str =
        ~"this \
               is \
               another \
               test";
    assert_eq!(a, ~"this is a test");
    assert_eq!(b, ~"this is another test");
}
