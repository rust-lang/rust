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

use ops::Deref;

/// A wrapper type for raw pointers and integers that will never be
/// NULL or 0 that might allow certain optimizations.
#[lang="non_zero"]
#[deriving(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Show)]
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

impl<T> Deref<T> for NonZero<T> {
    #[inline]
    fn deref<'a>(&'a self) -> &'a T {
        let NonZero(ref inner) = *self;
        inner
    }
}
