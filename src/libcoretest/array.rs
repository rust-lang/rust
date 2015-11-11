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
use core::iter::{empty, Iterator, repeat};

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
fn test_from_iter() {
    assert_eq!((0..3).collect::<[u8; 3]>(), [0, 1, 2]);
    assert_eq!((0..5).map(|x| x * x).collect::<[i32; 5]>(), [0, 1, 4, 9, 16]);
    assert_eq!(repeat([0, 1, 2]).take(32).collect::<[[u64; 3]; 32]>(), [[0, 1, 2]; 32]);
    assert_eq!(empty().collect::<[u8; 0]>(), []);
}

#[test]
#[should_panic(expected = "iterator too short")]
fn test_from_iter_too_short() {
    (0..2).collect::<[i32; 3]>();
}

#[test]
#[should_panic(expected = "iterator too long")]
fn test_from_iter_too_long() {
    (0..3).collect::<[i32; 2]>();
}

#[test]
#[should_panic(expected = "iterator too long")]
fn test_from_iter_too_long_empty() {
    repeat(0).collect::<[i32; 0]>();
}
