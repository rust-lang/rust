// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fmt;

/// An unbounded range (`..`).
///
/// `RangeFull` is primarily used as a [slicing index], its shorthand is `..`.
/// It cannot serve as an [`Iterator`] because it doesn't have a starting point.
///
/// # Examples
///
/// The `..` syntax is a `RangeFull`:
///
/// ```
/// assert_eq!((..), std::ops::RangeFull);
/// ```
///
/// It does not have an [`IntoIterator`] implementation, so you can't use it in
/// a `for` loop directly. This won't compile:
///
/// ```compile_fail,E0277
/// for i in .. {
///    // ...
/// }
/// ```
///
/// Used as a [slicing index], `RangeFull` produces the full array as a slice.
///
/// ```
/// let arr = [0, 1, 2, 3];
/// assert_eq!(arr[ .. ], [0,1,2,3]);  // RangeFull
/// assert_eq!(arr[ ..3], [0,1,2  ]);
/// assert_eq!(arr[1.. ], [  1,2,3]);
/// assert_eq!(arr[1..3], [  1,2  ]);
/// ```
///
/// [`IntoIterator`]: ../iter/trait.Iterator.html
/// [`Iterator`]: ../iter/trait.IntoIterator.html
/// [slicing index]: ../slice/trait.SliceIndex.html
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RangeFull;

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for RangeFull {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "..")
    }
}

/// A (half-open) range bounded inclusively below and exclusively above
/// (`start..end`).
///
/// The `Range` `start..end` contains all values with `x >= start` and
/// `x < end`.
///
/// # Examples
///
/// ```
/// assert_eq!((3..5), std::ops::Range { start: 3, end: 5 });
/// assert_eq!(3 + 4 + 5, (3..6).sum());
///
/// let arr = [0, 1, 2, 3];
/// assert_eq!(arr[ .. ], [0,1,2,3]);
/// assert_eq!(arr[ ..3], [0,1,2  ]);
/// assert_eq!(arr[1.. ], [  1,2,3]);
/// assert_eq!(arr[1..3], [  1,2  ]);  // Range
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]  // not Copy -- see #27186
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Range<Idx> {
    /// The lower bound of the range (inclusive).
    #[stable(feature = "rust1", since = "1.0.0")]
    pub start: Idx,
    /// The upper bound of the range (exclusive).
    #[stable(feature = "rust1", since = "1.0.0")]
    pub end: Idx,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<Idx: fmt::Debug> fmt::Debug for Range<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}..{:?}", self.start, self.end)
    }
}

#[unstable(feature = "range_contains", reason = "recently added as per RFC", issue = "32311")]
impl<Idx: PartialOrd<Idx>> Range<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(range_contains)]
    ///
    /// assert!(!(3..5).contains(2));
    /// assert!( (3..5).contains(3));
    /// assert!( (3..5).contains(4));
    /// assert!(!(3..5).contains(5));
    ///
    /// assert!(!(3..3).contains(3));
    /// assert!(!(3..2).contains(3));
    /// ```
    pub fn contains(&self, item: Idx) -> bool {
        (self.start <= item) && (item < self.end)
    }
}

/// A range only bounded inclusively below (`start..`).
///
/// The `RangeFrom` `start..` contains all values with `x >= start`.
///
/// *Note*: Currently, no overflow checking is done for the [`Iterator`]
/// implementation; if you use an integer range and the integer overflows, it
/// might panic in debug mode or create an endless loop in release mode. **This
/// overflow behavior might change in the future.**
///
/// # Examples
///
/// ```
/// assert_eq!((2..), std::ops::RangeFrom { start: 2 });
/// assert_eq!(2 + 3 + 4, (2..).take(3).sum());
///
/// let arr = [0, 1, 2, 3];
/// assert_eq!(arr[ .. ], [0,1,2,3]);
/// assert_eq!(arr[ ..3], [0,1,2  ]);
/// assert_eq!(arr[1.. ], [  1,2,3]);  // RangeFrom
/// assert_eq!(arr[1..3], [  1,2  ]);
/// ```
///
/// [`Iterator`]: ../iter/trait.IntoIterator.html
#[derive(Clone, PartialEq, Eq, Hash)]  // not Copy -- see #27186
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RangeFrom<Idx> {
    /// The lower bound of the range (inclusive).
    #[stable(feature = "rust1", since = "1.0.0")]
    pub start: Idx,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<Idx: fmt::Debug> fmt::Debug for RangeFrom<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}..", self.start)
    }
}

#[unstable(feature = "range_contains", reason = "recently added as per RFC", issue = "32311")]
impl<Idx: PartialOrd<Idx>> RangeFrom<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(range_contains)]
    ///
    /// assert!(!(3..).contains(2));
    /// assert!( (3..).contains(3));
    /// assert!( (3..).contains(1_000_000_000));
    /// ```
    pub fn contains(&self, item: Idx) -> bool {
        (self.start <= item)
    }
}

/// A range only bounded exclusively above (`..end`).
///
/// The `RangeTo` `..end` contains all values with `x < end`.
/// It cannot serve as an [`Iterator`] because it doesn't have a starting point.
///
/// # Examples
///
/// The `..end` syntax is a `RangeTo`:
///
/// ```
/// assert_eq!((..5), std::ops::RangeTo { end: 5 });
/// ```
///
/// It does not have an [`IntoIterator`] implementation, so you can't use it in
/// a `for` loop directly. This won't compile:
///
/// ```compile_fail,E0277
/// // error[E0277]: the trait bound `std::ops::RangeTo<{integer}>:
/// // std::iter::Iterator` is not satisfied
/// for i in ..5 {
///     // ...
/// }
/// ```
///
/// When used as a [slicing index], `RangeTo` produces a slice of all array
/// elements before the index indicated by `end`.
///
/// ```
/// let arr = [0, 1, 2, 3];
/// assert_eq!(arr[ .. ], [0,1,2,3]);
/// assert_eq!(arr[ ..3], [0,1,2  ]);  // RangeTo
/// assert_eq!(arr[1.. ], [  1,2,3]);
/// assert_eq!(arr[1..3], [  1,2  ]);
/// ```
///
/// [`IntoIterator`]: ../iter/trait.Iterator.html
/// [`Iterator`]: ../iter/trait.IntoIterator.html
/// [slicing index]: ../slice/trait.SliceIndex.html
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RangeTo<Idx> {
    /// The upper bound of the range (exclusive).
    #[stable(feature = "rust1", since = "1.0.0")]
    pub end: Idx,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<Idx: fmt::Debug> fmt::Debug for RangeTo<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "..{:?}", self.end)
    }
}

#[unstable(feature = "range_contains", reason = "recently added as per RFC", issue = "32311")]
impl<Idx: PartialOrd<Idx>> RangeTo<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(range_contains)]
    ///
    /// assert!( (..5).contains(-1_000_000_000));
    /// assert!( (..5).contains(4));
    /// assert!(!(..5).contains(5));
    /// ```
    pub fn contains(&self, item: Idx) -> bool {
        (item < self.end)
    }
}

/// An range bounded inclusively below and above (`start..=end`).
///
/// The `RangeInclusive` `start..=end` contains all values with `x >= start`
/// and `x <= end`.
///
/// # Examples
///
/// ```
/// #![feature(inclusive_range,inclusive_range_syntax)]
///
/// assert_eq!((3..=5), std::ops::RangeInclusive { start: 3, end: 5 });
/// assert_eq!(3 + 4 + 5, (3..=5).sum());
///
/// let arr = [0, 1, 2, 3];
/// assert_eq!(arr[ ..=2], [0,1,2  ]);
/// assert_eq!(arr[1..=2], [  1,2  ]);  // RangeInclusive
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]  // not Copy -- see #27186
#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
pub struct RangeInclusive<Idx> {
    /// The lower bound of the range (inclusive).
    #[unstable(feature = "inclusive_range",
               reason = "recently added, follows RFC",
               issue = "28237")]
    pub start: Idx,
    /// The upper bound of the range (inclusive).
    #[unstable(feature = "inclusive_range",
               reason = "recently added, follows RFC",
               issue = "28237")]
    pub end: Idx,
}

#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
impl<Idx: fmt::Debug> fmt::Debug for RangeInclusive<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}..={:?}", self.start, self.end)
    }
}

#[unstable(feature = "range_contains", reason = "recently added as per RFC", issue = "32311")]
impl<Idx: PartialOrd<Idx>> RangeInclusive<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(range_contains,inclusive_range_syntax)]
    ///
    /// assert!(!(3..=5).contains(2));
    /// assert!( (3..=5).contains(3));
    /// assert!( (3..=5).contains(4));
    /// assert!( (3..=5).contains(5));
    /// assert!(!(3..=5).contains(6));
    ///
    /// assert!( (3..=3).contains(3));
    /// assert!(!(3..=2).contains(3));
    /// ```
    pub fn contains(&self, item: Idx) -> bool {
        self.start <= item && item <= self.end
    }
}

/// A range only bounded inclusively above (`..=end`).
///
/// The `RangeToInclusive` `..=end` contains all values with `x <= end`.
/// It cannot serve as an [`Iterator`] because it doesn't have a starting point.
///
/// # Examples
///
/// The `..=end` syntax is a `RangeToInclusive`:
///
/// ```
/// #![feature(inclusive_range,inclusive_range_syntax)]
/// assert_eq!((..=5), std::ops::RangeToInclusive{ end: 5 });
/// ```
///
/// It does not have an [`IntoIterator`] implementation, so you can't use it in a
/// `for` loop directly. This won't compile:
///
/// ```compile_fail,E0277
/// #![feature(inclusive_range_syntax)]
///
/// // error[E0277]: the trait bound `std::ops::RangeToInclusive<{integer}>:
/// // std::iter::Iterator` is not satisfied
/// for i in ..=5 {
///     // ...
/// }
/// ```
///
/// When used as a [slicing index], `RangeToInclusive` produces a slice of all
/// array elements up to and including the index indicated by `end`.
///
/// ```
/// #![feature(inclusive_range_syntax)]
///
/// let arr = [0, 1, 2, 3];
/// assert_eq!(arr[ ..=2], [0,1,2  ]);  // RangeToInclusive
/// assert_eq!(arr[1..=2], [  1,2  ]);
/// ```
///
/// [`IntoIterator`]: ../iter/trait.Iterator.html
/// [`Iterator`]: ../iter/trait.IntoIterator.html
/// [slicing index]: ../slice/trait.SliceIndex.html
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
pub struct RangeToInclusive<Idx> {
    /// The upper bound of the range (inclusive)
    #[unstable(feature = "inclusive_range",
               reason = "recently added, follows RFC",
               issue = "28237")]
    pub end: Idx,
}

#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
impl<Idx: fmt::Debug> fmt::Debug for RangeToInclusive<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "..={:?}", self.end)
    }
}

#[unstable(feature = "range_contains", reason = "recently added as per RFC", issue = "32311")]
impl<Idx: PartialOrd<Idx>> RangeToInclusive<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(range_contains,inclusive_range_syntax)]
    ///
    /// assert!( (..=5).contains(-1_000_000_000));
    /// assert!( (..=5).contains(5));
    /// assert!(!(..=5).contains(6));
    /// ```
    pub fn contains(&self, item: Idx) -> bool {
        (item <= self.end)
    }
}

// RangeToInclusive<Idx> cannot impl From<RangeTo<Idx>>
// because underflow would be possible with (..0).into()
