// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(repeat_generic_slice)]

fn main() {
    assert_eq!([1, 2].repeat(2), vec![1, 2, 1, 2]);
    assert_eq!([1, 2, 3, 4].repeat(0), vec![]);
    assert_eq!([1, 2, 3, 4].repeat(1), vec![1, 2, 3, 4]);
    assert_eq!([1, 2, 3, 4].repeat(3),
               vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);
}
