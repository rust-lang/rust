// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn test_generic<T: Clone, F>(expected: Box<T>, eq: F) where F: FnOnce(Box<T>, Box<T>) -> bool {
    let actual: Box<T> = match true {
        true => { expected.clone() },
        _ => panic!("wat")
    };
    assert!(eq(expected, actual));
}

fn test_box() {
    fn compare_box(b1: Box<bool>, b2: Box<bool>) -> bool {
        return *b1 == *b2;
    }
    test_generic::<bool, _>(box true, compare_box);
}

pub fn main() { test_box(); }
