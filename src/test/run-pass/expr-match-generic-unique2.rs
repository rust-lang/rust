// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![allow(unknown_features)]
#![feature(box_syntax)]

fn test_generic<T: Clone, F>(expected: T, eq: F) where F: FnOnce(T, T) -> bool {
    let actual: T = match true {
        true => expected.clone(),
        _ => panic!("wat")
    };
    assert!(eq(expected, actual));
}

fn test_vec() {
    fn compare_box(v1: Box<isize>, v2: Box<isize>) -> bool { return v1 == v2; }
    test_generic::<Box<isize>, _>(box 1, compare_box);
}

pub fn main() { test_vec(); }
