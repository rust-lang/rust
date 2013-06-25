// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Borrowed pointer utilities

#[cfg(not(test))]
use prelude::*;

/// Cast a region pointer - &T - to a uint.
#[inline]
pub fn to_uint<T>(thing: &T) -> uint {
    thing as *T as uint
}

/// Determine if two borrowed pointers point to the same thing.
#[inline]
pub fn ref_eq<'a, 'b, T>(thing: &'a T, other: &'b T) -> bool {
    to_uint(thing) == to_uint(other)
}

// Equality for region pointers
#[cfg(not(test))]
impl<'self, T: Eq> Eq for &'self T {
    #[inline]
    fn eq(&self, other: & &'self T) -> bool {
        *(*self) == *(*other)
    }
    #[inline]
    fn ne(&self, other: & &'self T) -> bool {
        *(*self) != *(*other)
    }
}

// Comparison for region pointers
#[cfg(not(test))]
impl<'self, T: Ord> Ord for &'self T {
    #[inline]
    fn lt(&self, other: & &'self T) -> bool {
        *(*self) < *(*other)
    }
    #[inline]
    fn le(&self, other: & &'self T) -> bool {
        *(*self) <= *(*other)
    }
    #[inline]
    fn ge(&self, other: & &'self T) -> bool {
        *(*self) >= *(*other)
    }
    #[inline]
    fn gt(&self, other: & &'self T) -> bool {
        *(*self) > *(*other)
    }
}
