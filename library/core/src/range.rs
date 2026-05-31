//! # Replacement range types
//!
//! The types within this module are meant to replace the legacy `Range`,
//! `RangeInclusive`, `RangeToInclusive` and `RangeFrom` types in a future edition.
//!
//! ```
//! use core::range::{Range, RangeFrom, RangeInclusive, RangeToInclusive};
//!
//! let arr = [0, 1, 2, 3, 4];
//! assert_eq!(arr[                        ..   ], [0, 1, 2, 3, 4]);
//! assert_eq!(arr[                        .. 3 ], [0, 1, 2      ]);
//! assert_eq!(arr[RangeToInclusive::from( ..=3)], [0, 1, 2, 3   ]);
//! assert_eq!(arr[       RangeFrom::from(1..  )], [   1, 2, 3, 4]);
//! assert_eq!(arr[           Range::from(1..3 )], [   1, 2      ]);
//! assert_eq!(arr[  RangeInclusive::from(1..=3)], [   1, 2, 3   ]);
//! ```

use crate::fmt;
use crate::hash::Hash;

mod iter;

#[unstable(feature = "new_range_api_legacy", issue = "125687")]
pub mod legacy;

#[doc(inline)]
#[stable(feature = "new_range_from_api", since = "1.96.0")]
pub use iter::RangeFromIter;
#[doc(inline)]
#[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
pub use iter::RangeInclusiveIter;
#[doc(inline)]
#[stable(feature = "new_range_api", since = "1.96.0")]
pub use iter::RangeIter;

// FIXME(#125687): re-exports temporarily removed
// Because re-exports of stable items (Bound, RangeBounds, RangeFull, RangeTo)
// can't be made unstable.
//
// #[doc(inline)]
// #[unstable(feature = "new_range_api", issue = "125687")]
// pub use crate::iter::Step;
// #[doc(inline)]
// #[unstable(feature = "new_range_api", issue = "125687")]
// pub use crate::ops::{Bound, IntoBounds, OneSidedRange, RangeBounds, RangeFull, RangeTo};
use crate::iter::Step;
use crate::ops::Bound::{self, Excluded, Included, Unbounded};
use crate::ops::{IntoBounds, OneSidedRange, OneSidedRangeBound, RangeBounds};

/// A (half-open) range bounded inclusively below and exclusively above.
///
/// The `Range` contains all values with `start <= x < end`.
/// It is empty if `start >= end`.
///
/// # Examples
///
/// ```
/// use core::range::Range;
///
/// assert_eq!(Range::from(3..5), Range { start: 3, end: 5 });
/// assert_eq!(3 + 4 + 5, Range::from(3..6).into_iter().sum());
/// ```
///
/// # Edition notes
///
/// It is planned that the syntax `start..end` will construct this
/// type in a future edition, but it does not do so today.
#[lang = "RangeCopy"]
#[derive(Copy, Hash)]
#[derive_const(Clone, Default, PartialEq, Eq)]
#[stable(feature = "new_range_api", since = "1.96.0")]
pub struct Range<Start, #[unstable(feature = "new_range_end_bound", issue = "155456")] End = Start>
{
    /// The lower bound of the range (inclusive).
    #[stable(feature = "new_range_api", since = "1.96.0")]
    pub start: Start,
    /// The upper bound of the range (exclusive).
    #[stable(feature = "new_range_api", since = "1.96.0")]
    pub end: End,
}

#[stable(feature = "new_range_api", since = "1.96.0")]
impl<Start: fmt::Debug, End: fmt::Debug> fmt::Debug for Range<Start, End> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.start.fmt(fmt)?;
        write!(fmt, "..")?;
        self.end.fmt(fmt)?;
        Ok(())
    }
}

impl<T: Step> Range<T> {
    /// Creates an iterator over the elements within this range.
    ///
    /// Shorthand for `.clone().into_iter()`
    ///
    /// # Examples
    ///
    /// ```
    /// use core::range::Range;
    ///
    /// let mut i = Range::from(3..9).iter().map(|n| n*n);
    /// assert_eq!(i.next(), Some(9));
    /// assert_eq!(i.next(), Some(16));
    /// assert_eq!(i.next(), Some(25));
    /// ```
    #[stable(feature = "new_range_api", since = "1.96.0")]
    #[inline]
    pub fn iter(&self) -> RangeIter<T> {
        self.clone().into_iter()
    }
}

impl<Start: PartialOrd, End: PartialOrd> Range<Start, End> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
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
    #[stable(feature = "new_range_api", since = "1.96.0")]
    #[rustc_const_unstable(feature = "const_range", issue = "none")]
    pub const fn contains<U>(&self, item: &U) -> bool
    where
        Start: [const] PartialOrd<U>,
        End: [const] PartialOrd<U>,
        U: ?Sized + [const] PartialOrd<Start> + [const] PartialOrd<End>,
    {
        <Self as RangeBounds<Start, End>>::contains(self, item)
    }

    /// Returns `true` if the range contains no items.
    ///
    /// # Examples
    ///
    /// ```
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
    /// use core::range::Range;
    ///
    /// assert!(!Range::from(3.0..5.0).is_empty());
    /// assert!( Range::from(3.0..f32::NAN).is_empty());
    /// assert!( Range::from(f32::NAN..5.0).is_empty());
    /// ```
    #[inline]
    #[stable(feature = "new_range_api", since = "1.96.0")]
    #[rustc_const_unstable(feature = "const_range", issue = "none")]
    pub const fn is_empty(&self) -> bool
    where
        Start: [const] PartialOrd<End>,
    {
        !(self.start < self.end)
    }
}

#[stable(feature = "new_range_api", since = "1.96.0")]
#[rustc_const_unstable(feature = "const_range", issue = "none")]
impl<Start, End> const RangeBounds<Start, End> for Range<Start, End> {
    fn start_bound(&self) -> Bound<&Start> {
        Included(&self.start)
    }
    fn end_bound(&self) -> Bound<&End> {
        Excluded(&self.end)
    }
}

// This impl intentionally does not have `T: ?Sized`;
// see https://github.com/rust-lang/rust/pull/61584 for discussion of why.
//
/// If you need to use this implementation where `T` is unsized,
/// consider using the `RangeBounds` impl for a 2-tuple of [`Bound<&T>`][Bound],
/// i.e. replace `start..end` with `(Bound::Included(start), Bound::Excluded(end))`.
#[stable(feature = "new_range_api", since = "1.96.0")]
#[rustc_const_unstable(feature = "const_range", issue = "none")]
impl<Start, End> const RangeBounds<Start, End> for Range<&Start, &End> {
    fn start_bound(&self) -> Bound<&Start> {
        Included(self.start)
    }
    fn end_bound(&self) -> Bound<&End> {
        Excluded(self.end)
    }
}

#[unstable(feature = "range_into_bounds", issue = "136903")]
#[rustc_const_unstable(feature = "const_range", issue = "none")]
impl<Start, End> const IntoBounds<Start, End> for Range<Start, End> {
    fn into_bounds(self) -> (Bound<Start>, Bound<End>) {
        (Included(self.start), Excluded(self.end))
    }
}

#[stable(feature = "new_range_api", since = "1.96.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T> const From<Range<T, T>> for legacy::Range<T> {
    #[inline]
    fn from(value: Range<T>) -> Self {
        Self { start: value.start, end: value.end }
    }
}
#[stable(feature = "new_range_api", since = "1.96.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T> const From<legacy::Range<T>> for Range<T, T> {
    #[inline]
    fn from(value: legacy::Range<T>) -> Self {
        Self { start: value.start, end: value.end }
    }
}

/// A range bounded inclusively below and above.
///
/// The `RangeInclusive` contains all values with `x >= start`
/// and `x <= last`. It is empty unless `start <= last`.
///
/// # Examples
///
/// ```
/// use core::range::RangeInclusive;
///
/// assert_eq!(RangeInclusive::from(3..=5), RangeInclusive { start: 3, last: 5 });
/// assert_eq!(3 + 4 + 5, RangeInclusive::from(3..=5).into_iter().sum());
/// ```
///
/// # Edition notes
///
/// It is planned that the syntax  `start..=last` will construct this
/// type in a future edition, but it does not do so today.
#[lang = "RangeInclusiveCopy"]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
pub struct RangeInclusive<
    Start,
    #[unstable(feature = "new_range_end_bound", issue = "155456")] End = Start,
> {
    /// The lower bound of the range (inclusive).
    #[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
    pub start: Start,
    /// The upper bound of the range (inclusive).
    #[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
    pub last: End,
}

#[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
impl<Start: fmt::Debug, End: fmt::Debug> fmt::Debug for RangeInclusive<Start, End> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.start.fmt(fmt)?;
        write!(fmt, "..=")?;
        self.last.fmt(fmt)?;
        Ok(())
    }
}

impl<Start: PartialOrd, End: PartialOrd> RangeInclusive<Start, End> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
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
    #[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
    #[rustc_const_unstable(feature = "const_range", issue = "none")]
    pub const fn contains<U>(&self, item: &U) -> bool
    where
        Start: [const] PartialOrd<U>,
        End: [const] PartialOrd<U>,
        U: ?Sized + [const] PartialOrd<Start> + [const] PartialOrd<End>,
    {
        <Self as RangeBounds<Start, End>>::contains(self, item)
    }

    /// Returns `true` if the range contains no items.
    ///
    /// # Examples
    ///
    /// ```
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
    /// use core::range::RangeInclusive;
    ///
    /// assert!(!RangeInclusive::from(3.0..=5.0).is_empty());
    /// assert!( RangeInclusive::from(3.0..=f32::NAN).is_empty());
    /// assert!( RangeInclusive::from(f32::NAN..=5.0).is_empty());
    /// ```
    #[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
    #[inline]
    #[rustc_const_unstable(feature = "const_range", issue = "none")]
    pub const fn is_empty(&self) -> bool
    where
        Start: [const] PartialOrd<End>,
    {
        !(self.start <= self.last)
    }
}

impl<T: Step> RangeInclusive<T> {
    /// Creates an iterator over the elements within this range.
    ///
    /// Shorthand for `.clone().into_iter()`
    ///
    /// # Examples
    ///
    /// ```
    /// use core::range::RangeInclusive;
    ///
    /// let mut i = RangeInclusive::from(3..=8).iter().map(|n| n*n);
    /// assert_eq!(i.next(), Some(9));
    /// assert_eq!(i.next(), Some(16));
    /// assert_eq!(i.next(), Some(25));
    /// ```
    #[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
    #[inline]
    pub fn iter(&self) -> RangeInclusiveIter<T> {
        self.clone().into_iter()
    }
}

#[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
#[rustc_const_unstable(feature = "const_range", issue = "none")]
impl<Start, End> const RangeBounds<Start, End> for RangeInclusive<Start, End> {
    fn start_bound(&self) -> Bound<&Start> {
        Included(&self.start)
    }
    fn end_bound(&self) -> Bound<&End> {
        Included(&self.last)
    }
}

// This impl intentionally does not have `T: ?Sized`;
// see https://github.com/rust-lang/rust/pull/61584 for discussion of why.
//
/// If you need to use this implementation where `T` is unsized,
/// consider using the `RangeBounds` impl for a 2-tuple of [`Bound<&T>`][Bound],
/// i.e. replace `start..=end` with `(Bound::Included(start), Bound::Included(end))`.
#[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
#[rustc_const_unstable(feature = "const_range", issue = "none")]
impl<Start, End> const RangeBounds<Start, End> for RangeInclusive<&Start, &End> {
    fn start_bound(&self) -> Bound<&Start> {
        Included(self.start)
    }
    fn end_bound(&self) -> Bound<&End> {
        Included(self.last)
    }
}

// #[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
#[unstable(feature = "range_into_bounds", issue = "136903")]
#[rustc_const_unstable(feature = "const_range", issue = "none")]
impl<Start, End> const IntoBounds<Start, End> for RangeInclusive<Start, End> {
    fn into_bounds(self) -> (Bound<Start>, Bound<End>) {
        (Included(self.start), Included(self.last))
    }
}

#[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T> const From<RangeInclusive<T>> for legacy::RangeInclusive<T> {
    #[inline]
    fn from(value: RangeInclusive<T>) -> Self {
        Self::new(value.start, value.last)
    }
}
#[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
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

/// A range only bounded inclusively below.
///
/// The `RangeFrom` contains all values with `x >= start`.
///
/// *Note*: Overflow in the [`IntoIterator`] implementation (when the contained
/// data type reaches its numerical limit) is allowed to panic, wrap, or
/// saturate. This behavior is defined by the implementation of the [`Step`]
/// trait. For primitive integers, this follows the normal rules, and respects
/// the overflow checks profile (panic in debug, wrap in release). Unlike
/// its legacy counterpart, the iterator will only panic after yielding the
/// maximum value when overflow checks are enabled.
///
/// [`Step`]: crate::iter::Step
///
/// # Examples
///
/// ```
/// use core::range::RangeFrom;
///
/// assert_eq!(RangeFrom::from(2..), core::range::RangeFrom { start: 2 });
/// assert_eq!(2 + 3 + 4, RangeFrom::from(2..).into_iter().take(3).sum());
/// ```
///
/// # Edition notes
///
/// It is planned that the syntax  `start..` will construct this
/// type in a future edition, but it does not do so today.
#[lang = "RangeFromCopy"]
#[derive(Copy, Hash)]
#[derive_const(Clone, PartialEq, Eq)]
#[stable(feature = "new_range_from_api", since = "1.96.0")]
pub struct RangeFrom<Start> {
    /// The lower bound of the range (inclusive).
    #[stable(feature = "new_range_from_api", since = "1.96.0")]
    pub start: Start,
}

#[stable(feature = "new_range_from_api", since = "1.96.0")]
impl<Start: fmt::Debug> fmt::Debug for RangeFrom<Start> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.start.fmt(fmt)?;
        write!(fmt, "..")?;
        Ok(())
    }
}

impl<Start: Step> RangeFrom<Start> {
    /// Creates an iterator over the elements within this range.
    ///
    /// Shorthand for `.clone().into_iter()`
    ///
    /// # Examples
    ///
    /// ```
    /// use core::range::RangeFrom;
    ///
    /// let mut i = RangeFrom::from(3..).iter().map(|n| n*n);
    /// assert_eq!(i.next(), Some(9));
    /// assert_eq!(i.next(), Some(16));
    /// assert_eq!(i.next(), Some(25));
    /// ```
    #[stable(feature = "new_range_from_api", since = "1.96.0")]
    #[inline]
    pub fn iter(&self) -> RangeFromIter<Start> {
        self.clone().into_iter()
    }
}

impl<Start: PartialOrd> RangeFrom<Start> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
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
    #[stable(feature = "new_range_from_api", since = "1.96.0")]
    #[rustc_const_unstable(feature = "const_range", issue = "none")]
    pub const fn contains<U>(&self, item: &U) -> bool
    where
        Start: [const] PartialOrd<U>,
        U: ?Sized + [const] PartialOrd<Start>,
    {
        <Self as RangeBounds<Start>>::contains(self, item)
    }
}

#[stable(feature = "new_range_from_api", since = "1.96.0")]
#[rustc_const_unstable(feature = "const_range", issue = "none")]
impl<Start, End> const RangeBounds<Start, End> for RangeFrom<Start> {
    fn start_bound(&self) -> Bound<&Start> {
        Included(&self.start)
    }
    fn end_bound(&self) -> Bound<&End> {
        Unbounded
    }
}

// This impl intentionally does not have `T: ?Sized`;
// see https://github.com/rust-lang/rust/pull/61584 for discussion of why.
//
/// If you need to use this implementation where `T` is unsized,
/// consider using the `RangeBounds` impl for a 2-tuple of [`Bound<&T>`][Bound],
/// i.e. replace `start..` with `(Bound::Included(start), Bound::Unbounded)`.
#[stable(feature = "new_range_from_api", since = "1.96.0")]
#[rustc_const_unstable(feature = "const_range", issue = "none")]
impl<Start, End> const RangeBounds<Start, End> for RangeFrom<&Start> {
    fn start_bound(&self) -> Bound<&Start> {
        Included(self.start)
    }
    fn end_bound(&self) -> Bound<&End> {
        Unbounded
    }
}

#[unstable(feature = "range_into_bounds", issue = "136903")]
#[rustc_const_unstable(feature = "const_range", issue = "none")]
impl<T> const IntoBounds<T> for RangeFrom<T> {
    fn into_bounds(self) -> (Bound<T>, Bound<T>) {
        (Included(self.start), Unbounded)
    }
}

#[unstable(feature = "one_sided_range", issue = "69780")]
#[rustc_const_unstable(feature = "const_range", issue = "none")]
impl<Start> const OneSidedRange<Start> for RangeFrom<Start>
where
    Self: RangeBounds<Start>,
{
    fn bound(self) -> (OneSidedRangeBound, Start) {
        (OneSidedRangeBound::StartInclusive, self.start)
    }
}

#[stable(feature = "new_range_from_api", since = "1.96.0")]
#[rustc_const_unstable(feature = "const_index", issue = "143775")]
impl<Start> const From<RangeFrom<Start>> for legacy::RangeFrom<Start> {
    #[inline]
    fn from(value: RangeFrom<Start>) -> Self {
        Self { start: value.start }
    }
}
#[stable(feature = "new_range_from_api", since = "1.96.0")]
#[rustc_const_unstable(feature = "const_index", issue = "143775")]
impl<Start> const From<legacy::RangeFrom<Start>> for RangeFrom<Start> {
    #[inline]
    fn from(value: legacy::RangeFrom<Start>) -> Self {
        Self { start: value.start }
    }
}

/// A range only bounded inclusively above.
///
/// The `RangeToInclusive` contains all values with `x <= last`.
/// It cannot serve as an [`Iterator`] because it doesn't have a starting point.
///
/// # Examples
///
/// ```standalone_crate
/// #![feature(new_range)]
/// assert_eq!((..=5), std::range::RangeToInclusive { last: 5 });
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
///
/// # Edition notes
///
/// It is planned that the syntax  `..=last` will construct this
/// type in a future edition, but it does not do so today.
#[lang = "RangeToInclusiveCopy"]
#[doc(alias = "..=")]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[stable(feature = "new_range_to_inclusive_api", since = "1.96.0")]
pub struct RangeToInclusive<End> {
    /// The upper bound of the range (inclusive)
    #[stable(feature = "new_range_to_inclusive_api", since = "1.96.0")]
    pub last: End,
}

#[stable(feature = "new_range_to_inclusive_api", since = "1.96.0")]
impl<End: fmt::Debug> fmt::Debug for RangeToInclusive<End> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "..=")?;
        self.last.fmt(fmt)?;
        Ok(())
    }
}

impl<End: PartialOrd> RangeToInclusive<End> {
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
    #[stable(feature = "new_range_to_inclusive_api", since = "1.96.0")]
    #[rustc_const_unstable(feature = "const_range", issue = "none")]
    pub const fn contains<U>(&self, item: &U) -> bool
    where
        End: [const] PartialOrd<U>,
        U: ?Sized + [const] PartialOrd<End>,
    {
        <Self as RangeBounds<End>>::contains(self, item)
    }
}

#[stable(feature = "new_range_to_inclusive_api", since = "1.96.0")]
impl<T> From<legacy::RangeToInclusive<T>> for RangeToInclusive<T> {
    fn from(value: legacy::RangeToInclusive<T>) -> Self {
        Self { last: value.end }
    }
}
#[stable(feature = "new_range_to_inclusive_api", since = "1.96.0")]
impl<T> From<RangeToInclusive<T>> for legacy::RangeToInclusive<T> {
    fn from(value: RangeToInclusive<T>) -> Self {
        Self { end: value.last }
    }
}

// RangeToInclusive<End> cannot impl From<RangeTo<End>>
// because underflow would be possible with (..0).into()

#[stable(feature = "new_range_to_inclusive_api", since = "1.96.0")]
#[rustc_const_unstable(feature = "const_range", issue = "none")]
impl<Start, End> const RangeBounds<Start, End> for RangeToInclusive<End> {
    fn start_bound(&self) -> Bound<&Start> {
        Unbounded
    }
    fn end_bound(&self) -> Bound<&End> {
        Included(&self.last)
    }
}

#[stable(feature = "new_range_to_inclusive_api", since = "1.96.0")]
#[rustc_const_unstable(feature = "const_range", issue = "none")]
impl<Start, End> const RangeBounds<Start, End> for RangeToInclusive<&End> {
    fn start_bound(&self) -> Bound<&Start> {
        Unbounded
    }
    fn end_bound(&self) -> Bound<&End> {
        Included(self.last)
    }
}

#[unstable(feature = "range_into_bounds", issue = "136903")]
#[rustc_const_unstable(feature = "const_range", issue = "none")]
impl<Start, End> const IntoBounds<Start, End> for RangeToInclusive<End> {
    fn into_bounds(self) -> (Bound<Start>, Bound<End>) {
        (Unbounded, Included(self.last))
    }
}

#[unstable(feature = "one_sided_range", issue = "69780")]
#[rustc_const_unstable(feature = "const_range", issue = "none")]
impl<End> const OneSidedRange<End> for RangeToInclusive<End>
where
    Self: RangeBounds<End>,
{
    fn bound(self) -> (OneSidedRangeBound, End) {
        (OneSidedRangeBound::EndInclusive, self.last)
    }
}
