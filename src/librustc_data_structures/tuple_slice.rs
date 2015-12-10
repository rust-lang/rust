// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::slice;

/// Allows to view uniform tuples as slices
pub trait TupleSlice<T> {
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];
}

macro_rules! impl_tuple_slice {
    ($tuple_type:ty, $size:expr) => {
        impl<T> TupleSlice<T> for $tuple_type {
            fn as_slice(&self) -> &[T] {
                unsafe {
                    let ptr = &self.0 as *const T;
                    slice::from_raw_parts(ptr, $size)
                }
            }

            fn as_mut_slice(&mut self) -> &mut [T] {
                unsafe {
                    let ptr = &mut self.0 as *mut T;
                    slice::from_raw_parts_mut(ptr, $size)
                }
            }
        }
    }
}

impl_tuple_slice!((T,T), 2);
impl_tuple_slice!((T,T,T), 3);
impl_tuple_slice!((T,T,T,T), 4);
impl_tuple_slice!((T,T,T,T,T), 5);
impl_tuple_slice!((T,T,T,T,T,T), 6);
impl_tuple_slice!((T,T,T,T,T,T,T), 7);
impl_tuple_slice!((T,T,T,T,T,T,T,T), 8);

#[test]
fn test_sliced_tuples() {
    let t2 = (100i32, 101i32);
    assert_eq!(t2.as_slice(), &[100i32, 101i32]);

    let t3 = (102i32, 103i32, 104i32);
    assert_eq!(t3.as_slice(), &[102i32, 103i32, 104i32]);

    let t4 = (105i32, 106i32, 107i32, 108i32);
    assert_eq!(t4.as_slice(), &[105i32, 106i32, 107i32, 108i32]);

    let t5 = (109i32, 110i32, 111i32, 112i32, 113i32);
    assert_eq!(t5.as_slice(), &[109i32, 110i32, 111i32, 112i32, 113i32]);
}
