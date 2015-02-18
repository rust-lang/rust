// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::result::Result::{Ok, Err};

#[test]
fn binary_search_not_found() {
    let b = [1, 2, 4, 6, 8, 9];
    assert!(b.binary_search_by(|v| v.cmp(&6)) == Ok(3));
    let b = [1, 2, 4, 6, 8, 9];
    assert!(b.binary_search_by(|v| v.cmp(&5)) == Err(3));
    let b = [1, 2, 4, 6, 7, 8, 9];
    assert!(b.binary_search_by(|v| v.cmp(&6)) == Ok(3));
    let b = [1, 2, 4, 6, 7, 8, 9];
    assert!(b.binary_search_by(|v| v.cmp(&5)) == Err(3));
    let b = [1, 2, 4, 6, 8, 9];
    assert!(b.binary_search_by(|v| v.cmp(&8)) == Ok(4));
    let b = [1, 2, 4, 6, 8, 9];
    assert!(b.binary_search_by(|v| v.cmp(&7)) == Err(4));
    let b = [1, 2, 4, 6, 7, 8, 9];
    assert!(b.binary_search_by(|v| v.cmp(&8)) == Ok(5));
    let b = [1, 2, 4, 5, 6, 8, 9];
    assert!(b.binary_search_by(|v| v.cmp(&7)) == Err(5));
    let b = [1, 2, 4, 5, 6, 8, 9];
    assert!(b.binary_search_by(|v| v.cmp(&0)) == Err(0));
    let b = [1, 2, 4, 5, 6, 8];
    assert!(b.binary_search_by(|v| v.cmp(&9)) == Err(6));
}

#[test]
fn iterator_to_slice() {
    macro_rules! test {
        ($data: expr) => {{
            let data: &mut [_] = &mut $data;
            let other_data: &mut [_] = &mut $data;

            {
                let mut iter = data.iter();
                assert_eq!(&iter[..], &other_data[..]);

                iter.next();
                assert_eq!(&iter[..], &other_data[1..]);

                iter.next_back();
                assert_eq!(&iter[..], &other_data[1..2]);

                let s = iter.as_slice();
                iter.next();
                assert_eq!(s, &other_data[1..2]);
            }
            {
                let mut iter = data.iter_mut();
                assert_eq!(&iter[..], &other_data[..]);
                // mutability:
                assert!(&mut iter[] == other_data);

                iter.next();
                assert_eq!(&iter[..], &other_data[1..]);
                assert!(&mut iter[] == &mut other_data[1..]);

                iter.next_back();

                assert_eq!(&iter[..], &other_data[1..2]);
                assert!(&mut iter[] == &mut other_data[1..2]);

                let s = iter.into_slice();
                assert!(s == &mut other_data[1..2]);
            }
        }}
    }

    // try types of a variety of sizes
    test!([(1u64, 1u64, 1u8), (2, 2, 2), (3, 3, 3)]);
    test!([1u64,2,3]);
    test!([1u8,2,3]);
    test!([(),(),()]);
}
