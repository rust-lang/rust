// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use core::array::FixedSizeArray;
use core::convert::TryFrom;

#[test]
fn fixed_size_array() {
    let mut array = [0; 64];
    let mut zero_sized = [(); 64];
    let mut empty_array = [0; 0];
    let mut empty_zero_sized = [(); 0];

    assert_eq!(FixedSizeArray::as_slice(&array).len(), 64);
    assert_eq!(FixedSizeArray::as_slice(&zero_sized).len(), 64);
    assert_eq!(FixedSizeArray::as_slice(&empty_array).len(), 0);
    assert_eq!(FixedSizeArray::as_slice(&empty_zero_sized).len(), 0);

    assert_eq!(FixedSizeArray::as_mut_slice(&mut array).len(), 64);
    assert_eq!(FixedSizeArray::as_mut_slice(&mut zero_sized).len(), 64);
    assert_eq!(FixedSizeArray::as_mut_slice(&mut empty_array).len(), 0);
    assert_eq!(FixedSizeArray::as_mut_slice(&mut empty_zero_sized).len(), 0);
}

#[test]
fn array_try_from() {
    macro_rules! test {
        ($($N:expr),+) => {
            $({
                type Array = [u8; $N];
                let array: Array = [0; $N];
                let slice: &[u8] = &array[..];

                let result = <&Array>::try_from(slice);
                assert_eq!(&array, result.unwrap());
            })+
        }
    }
    test! {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32
    }
}
