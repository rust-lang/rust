//! # Experimental replacement range types
//!
//! The types within this module are meant to replace the existing
//! `Range`, `RangeInclusive`, and `RangeFrom` types in a future edition.
//!
//! ```
//! #![feature(new_range_api)]
//! use core::range::{Range, RangeFrom, RangeInclusive};
//!
//! let arr = [0, 1, 2, 3, 4];
//! assert_eq!(arr[                      ..   ], [0, 1, 2, 3, 4]);
//! assert_eq!(arr[                      .. 3 ], [0, 1, 2      ]);
//! assert_eq!(arr[                      ..=3 ], [0, 1, 2, 3   ]);
//! assert_eq!(arr[     RangeFrom::from(1..  )], [   1, 2, 3, 4]);
//! assert_eq!(arr[         Range::from(1..3 )], [   1, 2      ]);
//! assert_eq!(arr[RangeInclusive::from(1..=3)], [   1, 2, 3   ]);
//! ```

use crate::fmt;
use crate::hash::Hash;

mod iter;

#[unstable(feature = "new_range_api", issue = "125687")]
pub mod legacy;

use Bound::{Excluded, Included, Unbounded};
#[doc(inline)]
pub use iter::{IterRange, IterRangeFrom, IterRangeInclusive};

#[doc(inline)]
pub use crate::iter::Step;
#[doc(inline)]
pub use crate::ops::{Bound, IntoBounds, OneSidedRange, RangeBounds, RangeFull, RangeTo};

/// A (half-open) range bounded inclusively below and exclusively above
/// (`start..end` in a future edition).
///
/// The range `start..end` contains all values with `start <= x < end`.
/// It is empty if `start >= end`.
///
/// # Examples
///
/// ```
/// #![feature(new_range_api)]
/// use core::range::Range;
///
/// assert_eq!(Range::from(3..5), Range { start: 3, end: 5 });
/// assert_eq!(3 + 4 + 5, Range::from(3..6).into_iter().sum());
/// ```
#[lang = "RangeCopy"]
#[derive(Copy, Hash)]
#[derive_const(Clone, Default, PartialEq, Eq)]
#[unstable(feature = "new_range_api", issue = "125687")]
pub struct Range<Idx> {
    /// The lower bound of the range (inclusive).
    #[unstable(feature = "new_range_api", issue = "125687")]
    pub start: Idx,
    /// The upper bound of the range (exclusive).
    #[unstable(feature = "new_range_api", issue = "125687")]
    pub end: Idx,
}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<Idx: fmt::Debug> fmt::Debug for Range<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.start.fmt(fmt)?;
        write!(fmt, "..")?;
        self.end.fmt(fmt)?;
        Ok(())
    }
}

impl<Idx: Step> Range<Idx> {
    /// Creates an iterator over the elements within this range.
    ///
    /// Shorthand for `.clone().into_iter()`
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(new_range_api)]
    /// use core::range::Range;
    ///
    /// let mut i = Range::from(3..9).iter().map(|n| n*n);
    /// assert_eq!(i.next(), Some(9));
    /// assert_eq!(i.next(), Some(16));
    /// assert_eq!(i.next(), Some(25));
    /// ```
    #[unstable(feature = "new_range_api", issue = "125687")]
    #[inline]
    pub fn iter(&self) -> IterRange<Idx> {
        self.clone().into_iter()
    }
}

impl<Idx: PartialOrd<Idx>> Range<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(new_range_api)]
    /// use core::range::Range;
    ///
    /// assert!(!Range::from(3..5).contains(&2));
    /// assert!( Range::from(3..5).contains(&3));
    /// assert!( Range::from(3..5).contains(&4));
    /// assert!(!Range::from(3..5).contains(&5));
    ///
    /// assert!(!Range::from(3..3).contains(&3));
    /// assert!(!Range::from(3..2).contains(&3));
    ///
    /// assert!( Range::from(0.0..1.0).contains(&0.5));
    /// assert!(!Range::from(0.0..1.0).contains(&f32::NAN));
    /// assert!(!Range::from(0.0..f32::NAN).contains(&0.5));
    /// assert!(!Range::from(f32::NAN..1.0).contains(&0.5));
    /// ```
    #[inline]
    #[unstable(feature = "new_range_api", issue = "125687")]
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
    /// #![feature(new_range_api)]
    /// use core::range::Range;
    ///
    /// assert!(!Range::from(3..5).is_empty());
    /// assert!( Range::from(3..3).is_empty());
    /// assert!( Range::from(3..2).is_empty());
    /// ```
    ///
    /// The range is empty if either side is incomparable:
    ///
    /// ```
    /// #![feature(new_range_api)]
    /// use core::range::Range;
    ///
    /// assert!(!Range::from(3.0..5.0).is_empty());
    /// assert!( Range::from(3.0..f32::NAN).is_empty());
    /// assert!( Range::from(f32::NAN..5.0).is_empty());
    /// ```
    #[inline]
    #[unstable(feature = "new_range_api", issue = "125687")]
    pub fn is_empty(&self) -> bool {
        !(self.start < self.end)
    }
}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> RangeBounds<T> for Range<T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(&self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        Excluded(&self.end)
    }
}

// This impl intentionally does not have `T: ?Sized`;
// see https://github.com/rust-lang/rust/pull/61584 for discussion of why.
//
/// If you need to use this implementation where `T` is unsized,
/// consider using the `RangeBounds` impl for a 2-tuple of [`Bound<&T>`][Bound],
/// i.e. replace `start..end` with `(Bound::Included(start), Bound::Excluded(end))`.
#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> RangeBounds<T> for Range<&T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        Excluded(self.end)
    }
}

// #[unstable(feature = "range_into_bounds", issue = "136903")]
#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> IntoBounds<T> for Range<T> {
    fn into_bounds(self) -> (Bound<T>, Bound<T>) {
        (Included(self.start), Excluded(self.end))
    }
}

#[unstable(feature = "new_range_api", issue = "125687")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T> const From<Range<T>> for legacy::Range<T> {
    #[inline]
    fn from(value: Range<T>) -> Self {
        Self { start: value.start, end: value.end }
    }
}

#[unstable(feature = "new_range_api", issue = "125687")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T> const From<legacy::Range<T>> for Range<T> {
    #[inline]
    fn from(value: legacy::Range<T>) -> Self {
        Self { start: value.start, end: value.end }
    }
}

/// A range bounded inclusively below and above (`start..=last`).
///
/// The `RangeInclusive` `start..=last` contains all values with `x >= start`
/// and `x <= last`. It is empty unless `start <= last`.
///
/// # Examples
///
/// The `start..=last` syntax is a `RangeInclusive`:
///
/// ```
/// #![feature(new_range_api)]
/// use core::range::RangeInclusive;
///
/// assert_eq!(RangeInclusive::from(3..=5), RangeInclusive { start: 3, last: 5 });
/// assert_eq!(3 + 4 + 5, RangeInclusive::from(3..=5).into_iter().sum());
/// ```
#[lang = "RangeInclusiveCopy"]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[unstable(feature = "new_range_api", issue = "125687")]
pub struct RangeInclusive<Idx> {
    /// The lower bound of the range (inclusive).
    #[unstable(feature = "new_range_api", issue = "125687")]
    pub start: Idx,
    /// The upper bound of the range (inclusive).
    #[unstable(feature = "new_range_api", issue = "125687")]
    pub last: Idx,
}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<Idx: fmt::Debug> fmt::Debug for RangeInclusive<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.start.fmt(fmt)?;
        write!(fmt, "..=")?;
        self.last.fmt(fmt)?;
        Ok(())
    }
}

impl<Idx: PartialOrd<Idx>> RangeInclusive<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(new_range_api)]
    /// use core::range::RangeInclusive;
    ///
    /// assert!(!RangeInclusive::from(3..=5).contains(&2));
    /// assert!( RangeInclusive::from(3..=5).contains(&3));
    /// assert!( RangeInclusive::from(3..=5).contains(&4));
    /// assert!( RangeInclusive::from(3..=5).contains(&5));
    /// assert!(!RangeInclusive::from(3..=5).contains(&6));
    ///
    /// assert!( RangeInclusive::from(3..=3).contains(&3));
    /// assert!(!RangeInclusive::from(3..=2).contains(&3));
    ///
    /// assert!( RangeInclusive::from(0.0..=1.0).contains(&1.0));
    /// assert!(!RangeInclusive::from(0.0..=1.0).contains(&f32::NAN));
    /// assert!(!RangeInclusive::from(0.0..=f32::NAN).contains(&0.0));
    /// assert!(!RangeInclusive::from(f32::NAN..=1.0).contains(&1.0));
    /// ```
    #[inline]
    #[unstable(feature = "new_range_api", issue = "125687")]
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
    /// #![feature(new_range_api)]
    /// use core::range::RangeInclusive;
    ///
    /// assert!(!RangeInclusive::from(3..=5).is_empty());
    /// assert!(!RangeInclusive::from(3..=3).is_empty());
    /// assert!( RangeInclusive::from(3..=2).is_empty());
    /// ```
    ///
    /// The range is empty if either side is incomparable:
    ///
    /// ```
    /// #![feature(new_range_api)]
    /// use core::range::RangeInclusive;
    ///
    /// assert!(!RangeInclusive::from(3.0..=5.0).is_empty());
    /// assert!( RangeInclusive::from(3.0..=f32::NAN).is_empty());
    /// assert!( RangeInclusive::from(f32::NAN..=5.0).is_empty());
    /// ```
    #[unstable(feature = "new_range_api", issue = "125687")]
    #[inline]
    pub fn is_empty(&self) -> bool {
        !(self.start <= self.last)
    }
}

impl<Idx: Step> RangeInclusive<Idx> {
    /// Creates an iterator over the elements within this range.
    ///
    /// Shorthand for `.clone().into_iter()`
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(new_range_api)]
    /// use core::range::RangeInclusive;
    ///
    /// let mut i = RangeInclusive::from(3..=8).iter().map(|n| n*n);
    /// assert_eq!(i.next(), Some(9));
    /// assert_eq!(i.next(), Some(16));
    /// assert_eq!(i.next(), Some(25));
    /// ```
    #[unstable(feature = "new_range_api", issue = "125687")]
    #[inline]
    pub fn iter(&self) -> IterRangeInclusive<Idx> {
        self.clone().into_iter()
    }
}

impl RangeInclusive<usize> {
    /// Converts to an exclusive `Range` for `SliceIndex` implementations.
    /// The caller is responsible for dealing with `last == usize::MAX`.
    #[inline]
    pub(crate) const fn into_slice_range(self) -> Range<usize> {
        Range { start: self.start, end: self.last + 1 }
    }
}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> RangeBounds<T> for RangeInclusive<T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(&self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        Included(&self.last)
    }
}

// This impl intentionally does not have `T: ?Sized`;
// see https://github.com/rust-lang/rust/pull/61584 for discussion of why.
//
/// If you need to use this implementation where `T` is unsized,
/// consider using the `RangeBounds` impl for a 2-tuple of [`Bound<&T>`][Bound],
/// i.e. replace `start..=end` with `(Bound::Included(start), Bound::Included(end))`.
#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> RangeBounds<T> for RangeInclusive<&T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        Included(self.last)
    }
}

// #[unstable(feature = "range_into_bounds", issue = "136903")]
#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> IntoBounds<T> for RangeInclusive<T> {
    fn into_bounds(self) -> (Bound<T>, Bound<T>) {
        (Included(self.start), Included(self.last))
    }
}

#[unstable(feature = "new_range_api", issue = "125687")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T> const From<RangeInclusive<T>> for legacy::RangeInclusive<T> {
    #[inline]
    fn from(value: RangeInclusive<T>) -> Self {
        Self::new(value.start, value.last)
    }
}
#[unstable(feature = "new_range_api", issue = "125687")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T> const From<legacy::RangeInclusive<T>> for RangeInclusive<T> {
    #[inline]
    fn from(value: legacy::RangeInclusive<T>) -> Self {
        assert!(
            !value.exhausted,
            "attempted to convert from an exhausted `legacy::RangeInclusive` (unspecified behavior)"
        );

        let (start, last) = value.into_inner();
        RangeInclusive { start, last }
    }
}

/// A range only bounded inclusively below (`start..`).
///
/// The `RangeFrom` `start..` contains all values with `x >= start`.
///
/// *Note*: Overflow in the [`Iterator`] implementation (when the contained
/// data type reaches its numerical limit) is allowed to panic, wrap, or
/// saturate. This behavior is defined by the implementation of the [`Step`]
/// trait. For primitive integers, this follows the normal rules, and respects
/// the overflow checks profile (panic in debug, wrap in release). Note also
/// that overflow happens earlier than you might assume: the overflow happens
/// in the call to `next` that yields the maximum value, as the range must be
/// set to a state to yield the next value.
///
/// [`Step`]: crate::iter::Step
///
/// # Examples
///
/// The `start..` syntax is a `RangeFrom`:
///
/// ```
/// #![feature(new_range_api)]
/// use core::range::RangeFrom;
///
/// assert_eq!(RangeFrom::from(2..), core::range::RangeFrom { start: 2 });
/// assert_eq!(2 + 3 + 4, RangeFrom::from(2..).into_iter().take(3).sum());
/// ```
#[lang = "RangeFromCopy"]
#[derive(Copy, Hash)]
#[derive_const(Clone, PartialEq, Eq)]
#[unstable(feature = "new_range_api", issue = "125687")]
pub struct RangeFrom<Idx> {
    /// The lower bound of the range (inclusive).
    #[unstable(feature = "new_range_api", issue = "125687")]
    pub start: Idx,
}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<Idx: fmt::Debug> fmt::Debug for RangeFrom<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.start.fmt(fmt)?;
        write!(fmt, "..")?;
        Ok(())
    }
}

impl<Idx: Step> RangeFrom<Idx> {
    /// Creates an iterator over the elements within this range.
    ///
    /// Shorthand for `.clone().into_iter()`
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(new_range_api)]
    /// use core::range::RangeFrom;
    ///
    /// let mut i = RangeFrom::from(3..).iter().map(|n| n*n);
    /// assert_eq!(i.next(), Some(9));
    /// assert_eq!(i.next(), Some(16));
    /// assert_eq!(i.next(), Some(25));
    /// ```
    #[unstable(feature = "new_range_api", issue = "125687")]
    #[inline]
    pub fn iter(&self) -> IterRangeFrom<Idx> {
        self.clone().into_iter()
    }
}

impl<Idx: PartialOrd<Idx>> RangeFrom<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(new_range_api)]
    /// use core::range::RangeFrom;
    ///
    /// assert!(!RangeFrom::from(3..).contains(&2));
    /// assert!( RangeFrom::from(3..).contains(&3));
    /// assert!( RangeFrom::from(3..).contains(&1_000_000_000));
    ///
    /// assert!( RangeFrom::from(0.0..).contains(&0.5));
    /// assert!(!RangeFrom::from(0.0..).contains(&f32::NAN));
    /// assert!(!RangeFrom::from(f32::NAN..).contains(&0.5));
    /// ```
    #[inline]
    #[unstable(feature = "new_range_api", issue = "125687")]
    pub fn contains<U>(&self, item: &U) -> bool
    where
        Idx: PartialOrd<U>,
        U: ?Sized + PartialOrd<Idx>,
    {
        <Self as RangeBounds<Idx>>::contains(self, item)
    }
}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> RangeBounds<T> for RangeFrom<T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(&self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        Unbounded
    }
}

// This impl intentionally does not have `T: ?Sized`;
// see https://github.com/rust-lang/rust/pull/61584 for discussion of why.
//
/// If you need to use this implementation where `T` is unsized,
/// consider using the `RangeBounds` impl for a 2-tuple of [`Bound<&T>`][Bound],
/// i.e. replace `start..` with `(Bound::Included(start), Bound::Unbounded)`.
#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> RangeBounds<T> for RangeFrom<&T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        Unbounded
    }
}

// #[unstable(feature = "range_into_bounds", issue = "136903")]
#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> IntoBounds<T> for RangeFrom<T> {
    fn into_bounds(self) -> (Bound<T>, Bound<T>) {
        (Included(self.start), Unbounded)
    }
}

#[unstable(feature = "new_range_api", issue = "125687")]
#[rustc_const_unstable(feature = "const_index", issue = "143775")]
impl<T> const From<RangeFrom<T>> for legacy::RangeFrom<T> {
    #[inline]
    fn from(value: RangeFrom<T>) -> Self {
        Self { start: value.start }
    }
}
#[unstable(feature = "new_range_api", issue = "125687")]
#[rustc_const_unstable(feature = "const_index", issue = "143775")]
impl<T> const From<legacy::RangeFrom<T>> for RangeFrom<T> {
    #[inline]
    fn from(value: legacy::RangeFrom<T>) -> Self {
        Self { start: value.start }
    }
}

/// A range only bounded inclusively above (`..=last`).
///
/// The `RangeToInclusive` `..=last` contains all values with `x <= last`.
/// It cannot serve as an [`Iterator`] because it doesn't have a starting point.
///
/// # Examples
///
/// The `..=last` syntax is a `RangeToInclusive`:
///
/// ```
/// #![feature(new_range_api)]
/// #![feature(new_range)]
/// assert_eq!((..=5), std::range::RangeToInclusive{ last: 5 });
/// ```
///
/// It does not have an [`IntoIterator`] implementation, so you can't use it in a
/// `for` loop directly. This won't compile:
///
/// ```compile_fail,E0277
/// // error[E0277]: the trait bound `std::range::RangeToInclusive<{integer}>:
/// // std::iter::Iterator` is not satisfied
/// for i in ..=5 {
///     // ...
/// }
/// ```
///
/// When used as a [slicing index], `RangeToInclusive` produces a slice of all
/// array elements up to and including the index indicated by `last`.
///
/// ```
/// let arr = [0, 1, 2, 3, 4];
/// assert_eq!(arr[ ..  ], [0, 1, 2, 3, 4]);
/// assert_eq!(arr[ .. 3], [0, 1, 2      ]);
/// assert_eq!(arr[ ..=3], [0, 1, 2, 3   ]); // This is a `RangeToInclusive`
/// assert_eq!(arr[1..  ], [   1, 2, 3, 4]);
/// assert_eq!(arr[1.. 3], [   1, 2      ]);
/// assert_eq!(arr[1..=3], [   1, 2, 3   ]);
/// ```
///
/// [slicing index]: crate::slice::SliceIndex
#[lang = "RangeToInclusiveCopy"]
#[doc(alias = "..=")]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[unstable(feature = "new_range_api", issue = "125687")]
pub struct RangeToInclusive<Idx> {
    /// The upper bound of the range (inclusive)
    #[unstable(feature = "new_range_api", issue = "125687")]
    pub last: Idx,
}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<Idx: fmt::Debug> fmt::Debug for RangeToInclusive<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "..=")?;
        self.last.fmt(fmt)?;
        Ok(())
    }
}

impl<Idx: PartialOrd<Idx>> RangeToInclusive<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!( (..=5).contains(&-1_000_000_000));
    /// assert!( (..=5).contains(&5));
    /// assert!(!(..=5).contains(&6));
    ///
    /// assert!( (..=1.0).contains(&1.0));
    /// assert!(!(..=1.0).contains(&f32::NAN));
    /// assert!(!(..=f32::NAN).contains(&0.5));
    /// ```
    #[inline]
    #[unstable(feature = "new_range_api", issue = "125687")]
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

#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> RangeBounds<T> for RangeToInclusive<T> {
    fn start_bound(&self) -> Bound<&T> {
        Unbounded
    }
    fn end_bound(&self) -> Bound<&T> {
        Included(&self.last)
    }
}

#[unstable(feature = "range_into_bounds", issue = "136903")]
impl<T> IntoBounds<T> for RangeToInclusive<T> {
    fn into_bounds(self) -> (Bound<T>, Bound<T>) {
        (Unbounded, Included(self.last))
    }
}
