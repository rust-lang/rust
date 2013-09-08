// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::iterator::*;

// Unfold had a bug with 'self that mean it didn't work
// cross-crate

fn main() {
    fn count(st: &mut uint) -> Option<uint> {
        if *st < 10 {
            let ret = Some(*st);
            *st += 1;
            ret
        } else {
            None
        }
    }

    let mut it = Unfold::new(0, count);
    let mut i = 0;
    for counted in it {
        assert_eq!(counted, i);
        i += 1;
    }
    assert_eq!(i, 10);
}
