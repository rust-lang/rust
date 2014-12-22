// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::slice::BinarySearchResult::{Found, NotFound};

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

#[test]
fn iterator_to_slice() {
    macro_rules! test {
        ($data: expr) => {{
            let data: &mut [_] = &mut $data;
            let other_data: &mut [_] = &mut $data;

            {
                let mut iter = data.iter();
                assert_eq!(iter[], other_data[]);

                iter.next();
                assert_eq!(iter[], other_data[1..]);

                iter.next_back();
                assert_eq!(iter[], other_data[1..2]);

                let s = iter.as_slice();
                iter.next();
                assert_eq!(s, other_data[1..2]);
            }
            {
                let mut iter = data.iter_mut();
                assert_eq!(iter[], other_data[]);
                // mutability:
                assert!(iter[mut] == other_data);

                iter.next();
                assert_eq!(iter[], other_data[1..]);
                assert!(iter[mut] == other_data[mut 1..]);

                iter.next_back();

                assert_eq!(iter[], other_data[1..2]);
                assert!(iter[mut] == other_data[mut 1..2]);

                let s = iter.into_slice();
                assert!(s == other_data[mut 1..2]);
            }
        }}
    }

    // try types of a variety of sizes
    test!([(1u64, 1u64, 1u8), (2, 2, 2), (3, 3, 3)]);
    test!([1u64,2,3]);
    test!([1u8,2,3]);
    test!([(),(),()]);
}
