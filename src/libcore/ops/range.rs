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
/// `x < end`.  It is empty unless `start < end`.
///
/// # Examples
///
/// ```
/// assert_eq!((3..5), std::ops::Range { start: 3, end: 5 });
/// assert_eq!(3 + 4 + 5, (3..6).sum());
///
/// let arr = ['a', 'b', 'c', 'd'];
/// assert_eq!(arr[ .. ], ['a', 'b', 'c', 'd']);
/// assert_eq!(arr[ ..3], ['a', 'b', 'c',    ]);
/// assert_eq!(arr[1.. ], [     'b', 'c', 'd']);
/// assert_eq!(arr[1..3], [     'b', 'c'     ]);  // Range
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

impl<Idx: PartialOrd<Idx>> Range<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(range_contains)]
    ///
    /// use std::f32;
    ///
    /// assert!(!(3..5).contains(&2));
    /// assert!( (3..5).contains(&3));
    /// assert!( (3..5).contains(&4));
    /// assert!(!(3..5).contains(&5));
    ///
    /// assert!(!(3..3).contains(&3));
    /// assert!(!(3..2).contains(&3));
    ///
    /// assert!( (0.0..1.0).contains(&0.5));
    /// assert!(!(0.0..1.0).contains(&f32::NAN));
    /// assert!(!(0.0..f32::NAN).contains(&0.5));
    /// assert!(!(f32::NAN..1.0).contains(&0.5));
    /// ```
    #[unstable(feature = "range_contains", reason = "recently added as per RFC", issue = "32311")]
    pub fn contains<U>(&self, item: &U) -> bool
    where
        Idx: PartialOrd<U>,
        U: ?Sized + PartialOrd<Idx>,
    {
        <Self as RangeBounds<Idx>>::contains(self, item)
    }

    /// Returns `true` if the range contains no items.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(range_is_empty)]
    ///
    /// assert!(!(3..5).is_empty());
    /// assert!( (3..3).is_empty());
    /// assert!( (3..2).is_empty());
    /// ```
    ///
    /// The range is empty if either side is incomparable:
    ///
    /// ```
    /// #![feature(range_is_empty)]
    ///
    /// use std::f32::NAN;
    /// assert!(!(3.0..5.0).is_empty());
    /// assert!( (3.0..NAN).is_empty());
    /// assert!( (NAN..5.0).is_empty());
    /// ```
    #[unstable(feature = "range_is_empty", reason = "recently added", issue = "48111")]
    pub fn is_empty(&self) -> bool {
        !(self.start < self.end)
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

impl<Idx: PartialOrd<Idx>> RangeFrom<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(range_contains)]
    ///
    /// use std::f32;
    ///
    /// assert!(!(3..).contains(&2));
    /// assert!( (3..).contains(&3));
    /// assert!( (3..).contains(&1_000_000_000));
    ///
    /// assert!( (0.0..).contains(&0.5));
    /// assert!(!(0.0..).contains(&f32::NAN));
    /// assert!(!(f32::NAN..).contains(&0.5));
    /// ```
    #[unstable(feature = "range_contains", reason = "recently added as per RFC", issue = "32311")]
    pub fn contains<U>(&self, item: &U) -> bool
    where
        Idx: PartialOrd<U>,
        U: ?Sized + PartialOrd<Idx>,
    {
        <Self as RangeBounds<Idx>>::contains(self, item)
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

impl<Idx: PartialOrd<Idx>> RangeTo<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(range_contains)]
    ///
    /// use std::f32;
    ///
    /// assert!( (..5).contains(&-1_000_000_000));
    /// assert!( (..5).contains(&4));
    /// assert!(!(..5).contains(&5));
    ///
    /// assert!( (..1.0).contains(&0.5));
    /// assert!(!(..1.0).contains(&f32::NAN));
    /// assert!(!(..f32::NAN).contains(&0.5));
    /// ```
    #[unstable(feature = "range_contains", reason = "recently added as per RFC", issue = "32311")]
    pub fn contains<U>(&self, item: &U) -> bool
    where
        Idx: PartialOrd<U>,
        U: ?Sized + PartialOrd<Idx>,
    {
        <Self as RangeBounds<Idx>>::contains(self, item)
    }
}

/// An range bounded inclusively below and above (`start..=end`).
///
/// The `RangeInclusive` `start..=end` contains all values with `x >= start`
/// and `x <= end`.  It is empty unless `start <= end`.
///
/// This iterator is [fused], but the specific values of `start` and `end` after
/// iteration has finished are **unspecified** other than that [`.is_empty()`]
/// will return `true` once no more values will be produced.
///
/// [fused]: ../iter/trait.FusedIterator.html
/// [`.is_empty()`]: #method.is_empty
///
/// # Examples
///
/// ```
/// #![feature(inclusive_range_fields)]
///
/// assert_eq!((3..=5), std::ops::RangeInclusive { start: 3, end: 5 });
/// assert_eq!(3 + 4 + 5, (3..=5).sum());
///
/// let arr = [0, 1, 2, 3];
/// assert_eq!(arr[ ..=2], [0,1,2  ]);
/// assert_eq!(arr[1..=2], [  1,2  ]);  // RangeInclusive
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]  // not Copy -- see #27186
#[stable(feature = "inclusive_range", since = "1.26.0")]
pub struct RangeInclusive<Idx> {
    /// The lower bound of the range (inclusive).
    #[unstable(feature = "inclusive_range_fields", issue = "49022")]
    pub start: Idx,
    /// The upper bound of the range (inclusive).
    #[unstable(feature = "inclusive_range_fields", issue = "49022")]
    pub end: Idx,
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<Idx: fmt::Debug> fmt::Debug for RangeInclusive<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}..={:?}", self.start, self.end)
    }
}

impl<Idx: PartialOrd<Idx>> RangeInclusive<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(range_contains)]
    ///
    /// use std::f32;
    ///
    /// assert!(!(3..=5).contains(&2));
    /// assert!( (3..=5).contains(&3));
    /// assert!( (3..=5).contains(&4));
    /// assert!( (3..=5).contains(&5));
    /// assert!(!(3..=5).contains(&6));
    ///
    /// assert!( (3..=3).contains(&3));
    /// assert!(!(3..=2).contains(&3));
    ///
    /// assert!( (0.0..=1.0).contains(&1.0));
    /// assert!(!(0.0..=1.0).contains(&f32::NAN));
    /// assert!(!(0.0..=f32::NAN).contains(&0.0));
    /// assert!(!(f32::NAN..=1.0).contains(&1.0));
    /// ```
    #[unstable(feature = "range_contains", reason = "recently added as per RFC", issue = "32311")]
    pub fn contains<U>(&self, item: &U) -> bool
    where
        Idx: PartialOrd<U>,
        U: ?Sized + PartialOrd<Idx>,
    {
        <Self as RangeBounds<Idx>>::contains(self, item)
    }

    /// Returns `true` if the range contains no items.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(range_is_empty)]
    ///
    /// assert!(!(3..=5).is_empty());
    /// assert!(!(3..=3).is_empty());
    /// assert!( (3..=2).is_empty());
    /// ```
    ///
    /// The range is empty if either side is incomparable:
    ///
    /// ```
    /// #![feature(range_is_empty)]
    ///
    /// use std::f32::NAN;
    /// assert!(!(3.0..=5.0).is_empty());
    /// assert!( (3.0..=NAN).is_empty());
    /// assert!( (NAN..=5.0).is_empty());
    /// ```
    ///
    /// This method returns `true` after iteration has finished:
    ///
    /// ```
    /// #![feature(range_is_empty)]
    ///
    /// let mut r = 3..=5;
    /// for _ in r.by_ref() {}
    /// // Precise field values are unspecified here
    /// assert!(r.is_empty());
    /// ```
    #[unstable(feature = "range_is_empty", reason = "recently added", issue = "48111")]
    pub fn is_empty(&self) -> bool {
        !(self.start <= self.end)
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
/// assert_eq!((..=5), std::ops::RangeToInclusive{ end: 5 });
/// ```
///
/// It does not have an [`IntoIterator`] implementation, so you can't use it in a
/// `for` loop directly. This won't compile:
///
/// ```compile_fail,E0277
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
/// let arr = [0, 1, 2, 3];
/// assert_eq!(arr[ ..=2], [0,1,2  ]);  // RangeToInclusive
/// assert_eq!(arr[1..=2], [  1,2  ]);
/// ```
///
/// [`IntoIterator`]: ../iter/trait.Iterator.html
/// [`Iterator`]: ../iter/trait.IntoIterator.html
/// [slicing index]: ../slice/trait.SliceIndex.html
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[stable(feature = "inclusive_range", since = "1.26.0")]
pub struct RangeToInclusive<Idx> {
    /// The upper bound of the range (inclusive)
    #[stable(feature = "inclusive_range", since = "1.26.0")]
    pub end: Idx,
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
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
    /// #![feature(range_contains)]
    ///
    /// use std::f32;
    ///
    /// assert!( (..=5).contains(&-1_000_000_000));
    /// assert!( (..=5).contains(&5));
    /// assert!(!(..=5).contains(&6));
    ///
    /// assert!( (..=1.0).contains(&1.0));
    /// assert!(!(..=1.0).contains(&f32::NAN));
    /// assert!(!(..=f32::NAN).contains(&0.5));
    /// ```
    #[unstable(feature = "range_contains", reason = "recently added as per RFC", issue = "32311")]
    pub fn contains<U>(&self, item: &U) -> bool
    where
        Idx: PartialOrd<U>,
        U: ?Sized + PartialOrd<Idx>,
    {
        <Self as RangeBounds<Idx>>::contains(self, item)
    }
}

// RangeToInclusive<Idx> cannot impl From<RangeTo<Idx>>
// because underflow would be possible with (..0).into()

/// An endpoint of a range of keys.
///
/// # Examples
///
/// `Bound`s are range endpoints:
///
/// ```
/// #![feature(collections_range)]
///
/// use std::ops::Bound::*;
/// use std::ops::RangeBounds;
///
/// assert_eq!((..100).start(), Unbounded);
/// assert_eq!((1..12).start(), Included(&1));
/// assert_eq!((1..12).end(), Excluded(&12));
/// ```
///
/// Using a tuple of `Bound`s as an argument to [`BTreeMap::range`].
/// Note that in most cases, it's better to use range syntax (`1..5`) instead.
///
/// ```
/// use std::collections::BTreeMap;
/// use std::ops::Bound::{Excluded, Included, Unbounded};
///
/// let mut map = BTreeMap::new();
/// map.insert(3, "a");
/// map.insert(5, "b");
/// map.insert(8, "c");
///
/// for (key, value) in map.range((Excluded(3), Included(8))) {
///     println!("{}: {}", key, value);
/// }
///
/// assert_eq!(Some((&3, &"a")), map.range((Unbounded, Included(5))).next());
/// ```
///
/// [`BTreeMap::range`]: ../../std/collections/btree_map/struct.BTreeMap.html#method.range
#[stable(feature = "collections_bound", since = "1.17.0")]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Bound<T> {
    /// An inclusive bound.
    #[stable(feature = "collections_bound", since = "1.17.0")]
    Included(#[stable(feature = "collections_bound", since = "1.17.0")] T),
    /// An exclusive bound.
    #[stable(feature = "collections_bound", since = "1.17.0")]
    Excluded(#[stable(feature = "collections_bound", since = "1.17.0")] T),
    /// An infinite endpoint. Indicates that there is no bound in this direction.
    #[stable(feature = "collections_bound", since = "1.17.0")]
    Unbounded,
}

#[unstable(feature = "collections_range",
           reason = "might be replaced with `Into<_>` and a type containing two `Bound` values",
           issue = "30877")]
/// `RangeBounds` is implemented by Rust's built-in range types, produced
/// by range syntax like `..`, `a..`, `..b` or `c..d`.
pub trait RangeBounds<T: ?Sized> {
    /// Start index bound.
    ///
    /// Returns the start value as a `Bound`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(collections_range)]
    ///
    /// # fn main() {
    /// use std::ops::Bound::*;
    /// use std::ops::RangeBounds;
    ///
    /// assert_eq!((..10).start(), Unbounded);
    /// assert_eq!((3..10).start(), Included(&3));
    /// # }
    /// ```
    fn start(&self) -> Bound<&T>;

    /// End index bound.
    ///
    /// Returns the end value as a `Bound`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(collections_range)]
    ///
    /// # fn main() {
    /// use std::ops::Bound::*;
    /// use std::ops::RangeBounds;
    ///
    /// assert_eq!((3..).end(), Unbounded);
    /// assert_eq!((3..10).end(), Excluded(&10));
    /// # }
    /// ```
    fn end(&self) -> Bound<&T>;


    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(range_contains)]
    ///
    /// use std::f32;
    ///
    /// assert!( (3..5).contains(&4));
    /// assert!(!(3..5).contains(&2));
    ///
    /// assert!( (0.0..1.0).contains(&0.5));
    /// assert!(!(0.0..1.0).contains(&f32::NAN));
    /// assert!(!(0.0..f32::NAN).contains(&0.5));
    /// assert!(!(f32::NAN..1.0).contains(&0.5));
    #[unstable(feature = "range_contains", reason = "recently added as per RFC", issue = "32311")]
    fn contains<U>(&self, item: &U) -> bool
    where
        T: PartialOrd<U>,
        U: ?Sized + PartialOrd<T>,
    {
        (match self.start() {
            Included(ref start) => *start <= item,
            Excluded(ref start) => *start < item,
            Unbounded => true,
        })
        &&
        (match self.end() {
            Included(ref end) => item <= *end,
            Excluded(ref end) => item < *end,
            Unbounded => true,
        })
    }
}

use self::Bound::{Excluded, Included, Unbounded};

#[unstable(feature = "collections_range",
           reason = "might be replaced with `Into<_>` and a type containing two `Bound` values",
           issue = "30877")]
impl<T: ?Sized> RangeBounds<T> for RangeFull {
    fn start(&self) -> Bound<&T> {
        Unbounded
    }
    fn end(&self) -> Bound<&T> {
        Unbounded
    }
}

#[unstable(feature = "collections_range",
           reason = "might be replaced with `Into<_>` and a type containing two `Bound` values",
           issue = "30877")]
impl<T> RangeBounds<T> for RangeFrom<T> {
    fn start(&self) -> Bound<&T> {
        Included(&self.start)
    }
    fn end(&self) -> Bound<&T> {
        Unbounded
    }
}

#[unstable(feature = "collections_range",
           reason = "might be replaced with `Into<_>` and a type containing two `Bound` values",
           issue = "30877")]
impl<T> RangeBounds<T> for RangeTo<T> {
    fn start(&self) -> Bound<&T> {
        Unbounded
    }
    fn end(&self) -> Bound<&T> {
        Excluded(&self.end)
    }
}

#[unstable(feature = "collections_range",
           reason = "might be replaced with `Into<_>` and a type containing two `Bound` values",
           issue = "30877")]
impl<T> RangeBounds<T> for Range<T> {
    fn start(&self) -> Bound<&T> {
        Included(&self.start)
    }
    fn end(&self) -> Bound<&T> {
        Excluded(&self.end)
    }
}

#[unstable(feature = "collections_range",
           reason = "might be replaced with `Into<_>` and a type containing two `Bound` values",
           issue = "30877")]
impl<T> RangeBounds<T> for RangeInclusive<T> {
    fn start(&self) -> Bound<&T> {
        Included(&self.start)
    }
    fn end(&self) -> Bound<&T> {
        Included(&self.end)
    }
}

#[unstable(feature = "collections_range",
           reason = "might be replaced with `Into<_>` and a type containing two `Bound` values",
           issue = "30877")]
impl<T> RangeBounds<T> for RangeToInclusive<T> {
    fn start(&self) -> Bound<&T> {
        Unbounded
    }
    fn end(&self) -> Bound<&T> {
        Included(&self.end)
    }
}

#[unstable(feature = "collections_range",
           reason = "might be replaced with `Into<_>` and a type containing two `Bound` values",
           issue = "30877")]
impl<T> RangeBounds<T> for (Bound<T>, Bound<T>) {
    fn start(&self) -> Bound<&T> {
        match *self {
            (Included(ref start), _) => Included(start),
            (Excluded(ref start), _) => Excluded(start),
            (Unbounded, _)           => Unbounded,
        }
    }

    fn end(&self) -> Bound<&T> {
        match *self {
            (_, Included(ref end)) => Included(end),
            (_, Excluded(ref end)) => Excluded(end),
            (_, Unbounded)         => Unbounded,
        }
    }
}

#[unstable(feature = "collections_range",
           reason = "might be replaced with `Into<_>` and a type containing two `Bound` values",
           issue = "30877")]
impl<'a, T: ?Sized + 'a> RangeBounds<T> for (Bound<&'a T>, Bound<&'a T>) {
    fn start(&self) -> Bound<&T> {
        self.0
    }

    fn end(&self) -> Bound<&T> {
        self.1
    }
}

#[unstable(feature = "collections_range",
           reason = "might be replaced with `Into<_>` and a type containing two `Bound` values",
           issue = "30877")]
impl<'a, T> RangeBounds<T> for RangeFrom<&'a T> {
    fn start(&self) -> Bound<&T> {
        Included(self.start)
    }
    fn end(&self) -> Bound<&T> {
        Unbounded
    }
}

#[unstable(feature = "collections_range",
           reason = "might be replaced with `Into<_>` and a type containing two `Bound` values",
           issue = "30877")]
impl<'a, T> RangeBounds<T> for RangeTo<&'a T> {
    fn start(&self) -> Bound<&T> {
        Unbounded
    }
    fn end(&self) -> Bound<&T> {
        Excluded(self.end)
    }
}

#[unstable(feature = "collections_range",
           reason = "might be replaced with `Into<_>` and a type containing two `Bound` values",
           issue = "30877")]
impl<'a, T> RangeBounds<T> for Range<&'a T> {
    fn start(&self) -> Bound<&T> {
        Included(self.start)
    }
    fn end(&self) -> Bound<&T> {
        Excluded(self.end)
    }
}

#[unstable(feature = "collections_range",
           reason = "might be replaced with `Into<_>` and a type containing two `Bound` values",
           issue = "30877")]
impl<'a, T> RangeBounds<T> for RangeInclusive<&'a T> {
    fn start(&self) -> Bound<&T> {
        Included(self.start)
    }
    fn end(&self) -> Bound<&T> {
        Included(self.end)
    }
}

#[unstable(feature = "collections_range",
           reason = "might be replaced with `Into<_>` and a type containing two `Bound` values",
           issue = "30877")]
impl<'a, T> RangeBounds<T> for RangeToInclusive<&'a T> {
    fn start(&self) -> Bound<&T> {
        Unbounded
    }
    fn end(&self) -> Bound<&T> {
        Included(self.end)
    }
}
