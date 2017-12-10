// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(slice_patterns)]

// NB: this test was introduced in #23121 and will have to change when default match binding modes
// stabilizes.

fn slice_pat(x: &[u8]) {
    // OLD!
    match x {
        [a, b..] => {},
        //~^ ERROR non-reference pattern used to match a reference
        _ => panic!(),
    }
}

fn main() {
    slice_pat("foo".as_bytes());
}
