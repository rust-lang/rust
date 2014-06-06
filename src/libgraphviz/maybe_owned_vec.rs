// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;
use std::iter::FromIterator;
use std::slice;

// Note 1: It is not clear whether the flexibility of providing both
// the `Growable` and `FixedLen` variants is sufficiently useful.
// Consider restricting to just a two variant enum.

// Note 2: Once Dynamically Sized Types (DST) lands, it might be
// reasonable to replace this with something like `enum MaybeOwned<'a,
// Sized? U>{ Owned(Box<U>), Borrowed(&'a U) }`; and then `U` could be
// instantiated with `[T]` or `str`, etc.  Of course, that would imply
// removing the `Growable` variant, which relates to note 1 above.
// Alternatively, we might add `MaybeOwned` for the general case but
// keep some form of `MaybeOwnedVector` to avoid unnecessary copying
// of the contents of `Vec<T>`, since we anticipate that to be a
// frequent way to dynamically construct a vector.

/// MaybeOwnedVector<'a,T> abstracts over `Vec<T>`, `&'a [T]`.
///
/// Some clients will have a pre-allocated vector ready to hand off in
/// a slice; others will want to create the set on the fly and hand
/// off ownership, via `Growable`.
pub enum MaybeOwnedVector<'a,T> {
    Growable(Vec<T>),
    Borrowed(&'a [T]),
}

/// Trait for moving into a `MaybeOwnedVector`
pub trait IntoMaybeOwnedVector<'a,T> {
    /// Moves self into a `MaybeOwnedVector`
    fn into_maybe_owned(self) -> MaybeOwnedVector<'a,T>;
}

impl<'a,T> IntoMaybeOwnedVector<'a,T> for Vec<T> {
    #[inline]
    fn into_maybe_owned(self) -> MaybeOwnedVector<'a,T> { Growable(self) }
}

impl<'a,T> IntoMaybeOwnedVector<'a,T> for &'a [T] {
    #[inline]
    fn into_maybe_owned(self) -> MaybeOwnedVector<'a,T> { Borrowed(self) }
}

impl<'a,T> MaybeOwnedVector<'a,T> {
    pub fn iter(&'a self) -> slice::Items<'a,T> {
        match self {
            &Growable(ref v) => v.iter(),
            &Borrowed(ref v) => v.iter(),
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
        // If we are building from scratch, might as well build the
        // most flexible variant.
        Growable(FromIterator::from_iter(iterator))
    }
}

impl<'a,T:fmt::Show> fmt::Show for MaybeOwnedVector<'a,T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl<'a,T:Clone> CloneableVector<T> for MaybeOwnedVector<'a,T> {
    /// Returns a copy of `self`.
    fn to_owned(&self) -> Vec<T> {
        self.as_slice().to_owned()
    }

    /// Convert `self` into an owned slice, not making a copy if possible.
    fn into_owned(self) -> Vec<T> {
        match self {
            Growable(v) => v.as_slice().to_owned(),
            Borrowed(v) => v.to_owned(),
        }
    }
}

impl<'a,T:Clone> MaybeOwnedVector<'a,T> {
    /// Convert `self` into a growable `Vec`, not making a copy if possible.
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Growable(v) => v,
            Borrowed(v) => Vec::from_slice(v),
        }
    }
}
