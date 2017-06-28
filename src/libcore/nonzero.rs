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

use ops::CoerceUnsized;

/// Unsafe trait to indicate what types are usable with the NonZero struct
pub unsafe trait Zeroable {
    /// Whether this value is zero
    fn is_zero(&self) -> bool;
}

macro_rules! impl_zeroable_for_pointer_types {
    ( $( $Ptr: ty )+ ) => {
        $(
            /// For fat pointers to be considered "zero", only the "data" part needs to be null.
            unsafe impl<T: ?Sized> Zeroable for $Ptr {
                #[inline]
                fn is_zero(&self) -> bool {
                    // Cast because `is_null` is only available on thin pointers
                    (*self as *mut u8).is_null()
                }
            }
        )+
    }
}

macro_rules! impl_zeroable_for_integer_types {
    ( $( $Int: ty )+ ) => {
        $(
            unsafe impl Zeroable for $Int {
                #[inline]
                fn is_zero(&self) -> bool {
                    *self == 0
                }
            }
        )+
    }
}

impl_zeroable_for_pointer_types! {
    *const T
    *mut T
}

impl_zeroable_for_integer_types! {
    usize u8 u16 u32 u64 u128
    isize i8 i16 i32 i64 i128
}

/// A wrapper type for raw pointers and integers that will never be
/// NULL or 0 that might allow certain optimizations.
#[lang = "non_zero"]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct NonZero<T: Zeroable>(T);

impl<T: Zeroable> NonZero<T> {
    /// Creates an instance of NonZero with the provided value.
    /// You must indeed ensure that the value is actually "non-zero".
    #[inline]
    pub const unsafe fn new_unchecked(inner: T) -> Self {
        NonZero(inner)
    }

    /// Creates an instance of NonZero with the provided value.
    #[inline]
    pub fn new(inner: T) -> Option<Self> {
        if inner.is_zero() {
            None
        } else {
            Some(NonZero(inner))
        }
    }

    /// Gets the inner value.
    pub fn get(self) -> T {
        self.0
    }
}

impl<T: Zeroable+CoerceUnsized<U>, U: Zeroable> CoerceUnsized<NonZero<U>> for NonZero<T> {}
