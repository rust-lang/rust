// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate collections;

use std::collections::HashMap;

// This is a fancy one: it uses an external iterator established
// outside the loop, breaks, then _picks back up_ and continues
// iterating with it.

pub fn main() {
    let mut h = HashMap::new();
    let kvs = [(1i, 10i), (2i, 20i), (3i, 30i)];
    for &(k,v) in kvs.iter() {
        h.insert(k,v);
    }
    let mut x = 0;
    let mut y = 0;

    let mut i = h.iter();

    for (&k,&v) in i {
        x += k;
        y += v;
        break;
    }

    for (&k,&v) in i {
        x += k;
        y += v;
    }

    assert_eq!(x, 6);
    assert_eq!(y, 60);
}
