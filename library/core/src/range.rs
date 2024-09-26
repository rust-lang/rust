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
pub use crate::ops::{Bound, OneSidedRange, RangeBounds, RangeFull, RangeTo, RangeToInclusive};

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
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
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

#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> RangeBounds<T> for Range<&T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        Excluded(self.end)
    }
}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> From<Range<T>> for legacy::Range<T> {
    #[inline]
    fn from(value: Range<T>) -> Self {
        Self { start: value.start, end: value.end }
    }
}
#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> From<legacy::Range<T>> for Range<T> {
    #[inline]
    fn from(value: legacy::Range<T>) -> Self {
        Self { start: value.start, end: value.end }
    }
}

/// A range bounded inclusively below and above (`start..=end`).
///
/// The `RangeInclusive` `start..=end` contains all values with `x >= start`
/// and `x <= end`. It is empty unless `start <= end`.
///
/// # Examples
///
/// The `start..=end` syntax is a `RangeInclusive`:
///
/// ```
/// #![feature(new_range_api)]
/// use core::range::RangeInclusive;
///
/// assert_eq!(RangeInclusive::from(3..=5), RangeInclusive { start: 3, end: 5 });
/// assert_eq!(3 + 4 + 5, RangeInclusive::from(3..=5).into_iter().sum());
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[unstable(feature = "new_range_api", issue = "125687")]
pub struct RangeInclusive<Idx> {
    /// The lower bound of the range (inclusive).
    #[unstable(feature = "new_range_api", issue = "125687")]
    pub start: Idx,
    /// The upper bound of the range (inclusive).
    #[unstable(feature = "new_range_api", issue = "125687")]
    pub end: Idx,
}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<Idx: fmt::Debug> fmt::Debug for RangeInclusive<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.start.fmt(fmt)?;
        write!(fmt, "..=")?;
        self.end.fmt(fmt)?;
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
        !(self.start <= self.end)
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
    /// The caller is responsible for dealing with `end == usize::MAX`.
    #[inline]
    pub(crate) const fn into_slice_range(self) -> Range<usize> {
        Range { start: self.start, end: self.end + 1 }
    }
}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> RangeBounds<T> for RangeInclusive<T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(&self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        Included(&self.end)
    }
}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> RangeBounds<T> for RangeInclusive<&T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        Included(self.end)
    }
}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> From<RangeInclusive<T>> for legacy::RangeInclusive<T> {
    #[inline]
    fn from(value: RangeInclusive<T>) -> Self {
        Self::new(value.start, value.end)
    }
}
#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> From<legacy::RangeInclusive<T>> for RangeInclusive<T> {
    #[inline]
    fn from(value: legacy::RangeInclusive<T>) -> Self {
        assert!(
            !value.exhausted,
            "attempted to convert from an exhausted `legacy::RangeInclusive` (unspecified behavior)"
        );

        let (start, end) = value.into_inner();
        RangeInclusive { start, end }
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
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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

#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> RangeBounds<T> for RangeFrom<&T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        Unbounded
    }
}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> From<RangeFrom<T>> for legacy::RangeFrom<T> {
    #[inline]
    fn from(value: RangeFrom<T>) -> Self {
        Self { start: value.start }
    }
}
#[unstable(feature = "new_range_api", issue = "125687")]
impl<T> From<legacy::RangeFrom<T>> for RangeFrom<T> {
    #[inline]
    fn from(value: legacy::RangeFrom<T>) -> Self {
        Self { start: value.start }
    }
}
