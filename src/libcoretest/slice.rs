// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::slice::{Found, NotFound};

#[test]
fn binary_search_not_found() {
    let b = [1i, 2, 4, 6, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&6)) == Found(3));
    let b = [1i, 2, 4, 6, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&5)) == NotFound(3));
    let b = [1i, 2, 4, 6, 7, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&6)) == Found(3));
    let b = [1i, 2, 4, 6, 7, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&5)) == NotFound(3));
    let b = [1i, 2, 4, 6, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&8)) == Found(4));
    let b = [1i, 2, 4, 6, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&7)) == NotFound(4));
    let b = [1i, 2, 4, 6, 7, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&8)) == Found(5));
    let b = [1i, 2, 4, 5, 6, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&7)) == NotFound(5));
    let b = [1i, 2, 4, 5, 6, 8, 9];
    assert!(b.binary_search(|v| v.cmp(&0)) == NotFound(0));
    let b = [1i, 2, 4, 5, 6, 8];
    assert!(b.binary_search(|v| v.cmp(&9)) == NotFound(6));
}
