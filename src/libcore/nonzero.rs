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

use intrinsics;
use marker::Copy;
use ops::Deref;
use option::Option;
use ptr::PtrExt;

/// Unsafe trait to indicate what types are usable with the NonZero struct
pub unsafe trait Zeroable {
    /// Returns `true` if the Zeroable item is zero.
    fn is_zero(&self) -> bool;
}

unsafe impl<T> Zeroable for *const T {
    #[inline(always)]
    fn is_zero(&self) -> bool { self.is_null() }
}

unsafe impl<T> Zeroable for *mut T {
    #[inline(always)]
    fn is_zero(&self) -> bool { self.is_null() }
}

unsafe impl Zeroable for int {
    #[inline(always)]
    fn is_zero(&self) -> bool { *self == 0 }
}
unsafe impl Zeroable for uint {
    #[inline(always)]
    fn is_zero(&self) -> bool { *self == 0 }
}

unsafe impl Zeroable for i8 {
    #[inline(always)]
    fn is_zero(&self) -> bool { *self == 0 }
}

unsafe impl Zeroable for u8 {
    #[inline(always)]
    fn is_zero(&self) -> bool { *self == 0 }
}

unsafe impl Zeroable for i16 {
    #[inline(always)]
    fn is_zero(&self) -> bool { *self == 0 }
}

unsafe impl Zeroable for u16 {
    #[inline(always)]
    fn is_zero(&self) -> bool { *self == 0 }
}

unsafe impl Zeroable for i32 {
    #[inline(always)]
    fn is_zero(&self) -> bool { *self == 0 }
}

unsafe impl Zeroable for u32 {
    #[inline(always)]
    fn is_zero(&self) -> bool { *self == 0 }
}

unsafe impl Zeroable for i64 {
    #[inline(always)]
    fn is_zero(&self) -> bool { *self == 0 }
}

unsafe impl Zeroable for u64 {
    #[inline(always)]
    fn is_zero(&self) -> bool { *self == 0 }
}

/// A wrapper type for raw pointers and integers that will never be
/// NULL or 0 that might allow certain optimizations.
#[lang="non_zero"]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[unstable(feature = "core")]
pub struct NonZero<T: Zeroable>(T);

impl<T: Zeroable> NonZero<T> {
    /// Create an instance of NonZero with the provided value.
    #[inline(always)]
    pub fn new(inner: T) -> Option<NonZero<T>> {
        if !inner.is_zero() {
            Option::Some(NonZero(inner))
        } else {
            Option::None
        }
    }

    /// Returns a reference to the inner value.
    #[inline(always)]
    pub fn get_ref(&self) -> &T {
        unsafe {
            let ret = &self.0;
            //intrinsics::assume(!ret.is_zero());
            ret
        }
    }
}

impl<T: Zeroable + Copy> NonZero<T> {
    /// Returns a copy of the inner value.
    #[inline(always)]
    pub fn get(self) -> T {
        unsafe {
            let ret = self.0;
            //intrinsics::assume(!ret.is_zero());
            ret
        }
    }
}
