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

use marker::Copy;
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

macro_rules! zeroable_impl {
    ($($t:ty)*) => ($(
        unsafe impl Zeroable for $t {
            #[inline(always)]
            fn is_zero(&self) -> bool { *self == 0 }
        }
    )*)
}

zeroable_impl! { isize usize i8 u8 i16 u16 i32 u32 i64 u64 }

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

    /// Crates an instance of NonZero with the provided value. No zero-checking
    /// will occur, making this function unsafe. Prefer `new`.
    #[inline(always)]
    pub unsafe fn new_unchecked(inner: T) -> NonZero<T> {
        NonZero(inner)
    }

    /// Returns a reference to the inner value.
    #[inline]
    pub fn get_ref(&self) -> &T {
        let ret = &self.0;
        /* FIXME: Triggers an assert in LLVM.
        unsafe {
            let is_nonzero = !ret.is_zero();
            ::intrinsics::assume(is_nonzero);
        }
        */
        ret
    }
}

impl<T: Zeroable + Copy> NonZero<T> {
    /// Returns a copy of the inner value.
    #[inline]
    pub fn get(self) -> T {
        let ret = self.0;
        /* FIXME: Triggers an assert in LLVM.
        unsafe {
            let is_nonzero = !ret.is_zero();
            ::intrinsics::assume(is_nonzero);
        }
        */
        ret
    }
}
