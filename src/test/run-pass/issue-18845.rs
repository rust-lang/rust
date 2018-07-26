// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This used to generate invalid IR in that even if we took the
// `false` branch we'd still try to free the Box from the other
// arm. This was due to treating `*Box::new(9)` as an rvalue datum
// instead of as a place.

fn test(foo: bool) -> u8 {
    match foo {
        true => *Box::new(9),
        false => 0
    }
}

fn main() {
    assert_eq!(9, test(true));
}
