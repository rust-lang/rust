// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::vec;

pub fn main() {
    let a = vec![1, 2];
    let b = vec![3, 4];
    let v = a + b;
    assert_eq!(v.len(), 4);
    assert_eq!(v[0], 1);
    assert_eq!(v[2], 3);

    let a = vec![1, 2];
    let b = vec![3, 4];
    let b_slice: &[u32] = &b;
    let v = a + b_slice;
    assert_eq!(v.len(), 4);
    assert_eq!(v[1], 2);
    assert_eq!(v[3], 4);
}
