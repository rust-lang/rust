// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Exposes the NonZero lang item which provides optimization hints.

use cmp::Eq;
use intrinsics;
use kinds::Copy;
use ops::Deref;
use option::Option;
use option::Option::Some;
use ptr::{null, null_mut, RawPtr, RawMutPtr};

/// A wrapper type for raw pointers and integers that will never be
/// NULL or 0 that might allow certain optimizations.
#[lang="non_zero"]
#[deriving(Clone, PartialEq, Eq, PartialOrd)]
#[experimental]
pub struct NonZero<T>(T);

impl<T> NonZero<T> {
    /// Create an instance of NonZero with the provided value.
    /// You must indeed ensure that the value is actually "non-zero".
    #[inline(always)]
    pub unsafe fn new(inner: T) -> NonZero<T> {
        NonZero(inner)
    }
}

impl<T: Copy> Copy for NonZero<T> {}

impl<T> Deref<T> for NonZero<T> {
    #[inline]
    fn deref<'a>(&'a self) -> &'a T {
        let NonZero(ref inner) = *self;
        inner
    }
}

impl<T> RawPtr<T> for NonZero<*const T> {
    #[inline]
    fn null() -> NonZero<*const T> { NonZero(null()) }

    #[inline]
    fn is_null(&self) -> bool { false }

    #[inline]
    fn to_uint(&self) -> uint {
        **self as uint
    }

    #[inline]
    unsafe fn offset(self, count: int) -> NonZero<*const T> {
        NonZero(intrinsics::offset(*self, count))
    }

    #[inline]
    unsafe fn as_ref<'a>(&self) -> Option<&'a T> {
        Some(&***self)
    }
}

impl<T> RawPtr<T> for NonZero<*mut T> {
    #[inline]
    fn null() -> NonZero<*mut T> { NonZero(null_mut()) }

    #[inline]
    fn is_null(&self) -> bool { false }

    #[inline]
    fn to_uint(&self) -> uint {
        **self as uint
    }

    #[inline]
    unsafe fn offset(self, count: int) -> NonZero<*mut T> {
        NonZero(intrinsics::offset(*self as *const T, count) as *mut T)
    }

    #[inline]
    unsafe fn as_ref<'a>(&self) -> Option<&'a T> {
        Some(&***self)
    }
}

impl<T> RawMutPtr<T> for NonZero<*mut T> {
    #[inline]
    unsafe fn as_mut<'a>(&self) -> Option<&'a mut T> {
        Some(&mut ***self)
    }
}
