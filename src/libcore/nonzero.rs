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
#![unstable(feature = "nonzero",
            reason = "needs an RFC to flesh out the design",
            issue = "27730")]

use marker::Sized;
use ops::{CoerceUnsized, Deref};

/// Unsafe trait to indicate what types are usable with the NonZero struct
pub unsafe trait Zeroable {}

unsafe impl<T:?Sized> Zeroable for *const T {}
unsafe impl<T:?Sized> Zeroable for *mut T {}
unsafe impl Zeroable for isize {}
unsafe impl Zeroable for usize {}
unsafe impl Zeroable for i8 {}
unsafe impl Zeroable for u8 {}
unsafe impl Zeroable for i16 {}
unsafe impl Zeroable for u16 {}
unsafe impl Zeroable for i32 {}
unsafe impl Zeroable for u32 {}
unsafe impl Zeroable for i64 {}
unsafe impl Zeroable for u64 {}

/// A wrapper type for raw pointers and integers that will never be
/// NULL or 0 that might allow certain optimizations.
#[lang = "non_zero"]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct NonZero<T: Zeroable>(T);

#[cfg(stage0)]
macro_rules! nonzero_new {
    () => (
        /// Creates an instance of NonZero with the provided value.
        /// You must indeed ensure that the value is actually "non-zero".
        #[inline(always)]
        pub unsafe fn new(inner: T) -> NonZero<T> {
            NonZero(inner)
        }
    )
}
#[cfg(not(stage0))]
macro_rules! nonzero_new {
    () => (
        /// Creates an instance of NonZero with the provided value.
        /// You must indeed ensure that the value is actually "non-zero".
        #[inline(always)]
        pub const unsafe fn new(inner: T) -> NonZero<T> {
            NonZero(inner)
        }
    )
}

impl<T: Zeroable> NonZero<T> {
    nonzero_new!{}
}

impl<T: Zeroable> Deref for NonZero<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        let NonZero(ref inner) = *self;
        inner
    }
}

impl<T: Zeroable+CoerceUnsized<U>, U: Zeroable> CoerceUnsized<NonZero<U>> for NonZero<T> {}
