// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::hashmap::HashMap;

fn main() {
    let mut h = HashMap::new();
    let kvs = [(1, 10), (2, 20), (3, 30)];
    foreach &(k,v) in kvs.iter() {
        h.insert(k,v);
    }
    let mut x = 0;
    let mut y = 0;
    foreach (&k,&v) in h.iter() {
        x += k;
        y += v;
    }
    assert_eq!(x, 6);
    assert_eq!(y, 60);
}