// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "collections_range",
            reason = "waiting for dust to settle on inclusive ranges",
            issue = "30877")]

//! Range syntax.

use core::ops::{RangeFull, Range, RangeTo, RangeFrom, RangeToInclusive, RangeInclusive};
use ::Bound;

/// **RangeArgument** is implemented by Rust's built-in range types, produced
/// by range syntax like `..`, `a..`, `..b` or `c..d`.
pub trait RangeArgument<T> {
    /// Lower bound of the range
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(collections)]
    /// #![feature(collections_range)]
    /// #![feature(collections_bound)]
    ///
    /// extern crate collections;
    ///
    /// # fn main() {
    /// use collections::range::RangeArgument;
    /// use collections::Bound::*;
    ///
    /// assert_eq!((..10).lower(), Unbounded);
    /// assert_eq!((3..10).lower(), Included(&3));
    /// # }
    /// ```
    fn lower(&self) -> Bound<&T> {
        Bound::Unbounded
    }

    /// Upper bound of the range
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(collections)]
    /// #![feature(collections_range)]
    /// #![feature(collections_bound)]
    ///
    /// extern crate collections;
    ///
    /// # fn main() {
    /// use collections::range::RangeArgument;
    /// use collections::Bound::*;
    ///
    /// assert_eq!((3..).upper(), Unbounded);
    /// assert_eq!((3..10).upper(), Excluded(&10));
    /// # }
    /// ```
    fn upper(&self) -> Bound<&T> {
        Bound::Unbounded
    }
}

impl<T> RangeArgument<T> for RangeFull {}

impl<T> RangeArgument<T> for RangeFrom<T> {
    fn lower(&self) -> Bound<&T> {
        Bound::Included(&self.start)
    }
}

impl<T> RangeArgument<T> for RangeTo<T> {
    fn upper(&self) -> Bound<&T> {
        Bound::Excluded(&self.end)
    }
}

impl<T> RangeArgument<T> for RangeToInclusive<T> {
    fn upper(&self) -> Bound<&T> {
        Bound::Included(&self.end)
    }
}

impl<T> RangeArgument<T> for Range<T> {
    fn lower(&self) -> Bound<&T> {
        Bound::Included(&self.start)
    }
    fn upper(&self) -> Bound<&T> {
        Bound::Excluded(&self.end)
    }
}

impl<T> RangeArgument<T> for RangeInclusive<T> {
    fn lower(&self) -> Bound<&T> {
        match *self {
            RangeInclusive::NonEmpty { ref start, .. } => Bound::Included(start),
            RangeInclusive::Empty { ref at } => Bound::Included(at),
        }
    }
    fn upper(&self) -> Bound<&T> {
        match *self {
            RangeInclusive::NonEmpty { ref end, .. } => Bound::Included(end),
            RangeInclusive::Empty { ref at } => Bound::Excluded(at),
        }
    }
}
