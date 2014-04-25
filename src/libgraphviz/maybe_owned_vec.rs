// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::container::Container;
use std::iter::FromIterator;
use std::slice;

// Note: Once Dynamically Sized Types (DST) lands, this should be
// replaced with something like `enum Owned<'a, Sized? U>{ Owned(~U),
// Borrowed(&'a U) }`; and then `U` could be instantiated with `[T]`
// or `str`, etc.

/// MaybeOwnedVector<'a,T> abstracts over `Vec<T>` and `&'a [T]`.
///
/// Some clients will have a pre-allocated vector ready to hand off in
/// a slice; others will want to create the set on the fly and hand
/// off ownership.
#[deriving(Eq)]
pub enum MaybeOwnedVector<'a,T> {
    Growable(Vec<T>),
    Borrowed(&'a [T]),
}

impl<'a,T> MaybeOwnedVector<'a,T> {
    pub fn iter(&'a self) -> slice::Items<'a,T> {
        match self {
            &Growable(ref v) => v.iter(),
            &Borrowed(ref v) => v.iter(),
        }
    }
}

impl<'a,T> Container for MaybeOwnedVector<'a,T> {
    fn len(&self) -> uint {
        match self {
            &Growable(ref v) => v.len(),
            &Borrowed(ref v) => v.len(),
        }
    }
}

// The `Vector` trait is provided in the prelude and is implemented on
// both `&'a [T]` and `Vec<T>`, so it makes sense to try to support it
// seamlessly.  The other vector related traits from the prelude do
// not appear to be implemented on both `&'a [T]` and `Vec<T>`.  (It
// is possible that this is an oversight in some cases.)
//
// In any case, with `Vector` in place, the client can just use
// `as_slice` if they prefer that over `match`.

impl<'b,T> slice::Vector<T> for MaybeOwnedVector<'b,T> {
    fn as_slice<'a>(&'a self) -> &'a [T] {
        match self {
            &Growable(ref v) => v.as_slice(),
            &Borrowed(ref v) => v.as_slice(),
        }
    }
}

impl<'a,T> FromIterator<T> for MaybeOwnedVector<'a,T> {
    fn from_iter<I:Iterator<T>>(iterator: I) -> MaybeOwnedVector<T> {
        Growable(FromIterator::from_iter(iterator))
    }
}
