// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test references to `Self::Item` in the trait. Issue #20220.

use std::vec;

trait IntoIteratorX {
    type Item;
    type IntoIter: Iterator<Item=Self::Item>;

    fn into_iter_x(self) -> Self::IntoIter;
}

impl<T> IntoIteratorX for Vec<T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter_x(self) -> vec::IntoIter<T> {
        self.into_iter()
    }
}

fn main() {
    let vec = vec![1, 2, 3];
    for (i, e) in vec.into_iter().enumerate() {
        assert_eq!(i+1, e);
    }
}
