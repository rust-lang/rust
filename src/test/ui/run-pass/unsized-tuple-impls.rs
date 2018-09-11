// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unsized_tuple_coercion)]

use std::collections::HashSet;

fn main() {
    let x : &(i32, i32, [i32]) = &(0, 1, [2, 3]);
    let y : &(i32, i32, [i32]) = &(0, 1, [2, 3, 4]);
    let mut a = [y, x];
    a.sort();
    assert_eq!(a, [x, y]);

    assert_eq!(&format!("{:?}", a), "[(0, 1, [2, 3]), (0, 1, [2, 3, 4])]");

    let mut h = HashSet::new();
    h.insert(x);
    h.insert(y);
    assert!(h.contains(x));
    assert!(h.contains(y));
}
