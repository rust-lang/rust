use crate::fmt;
use crate::hash::{Hash, Hasher};

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
/// let arr = [0, 1, 2, 3, 4];
/// assert_eq!(arr[ ..  ], [0,1,2,3,4]);  // RangeFull
/// assert_eq!(arr[ .. 3], [0,1,2    ]);
/// assert_eq!(arr[ ..=3], [0,1,2,3  ]);
/// assert_eq!(arr[1..  ], [  1,2,3,4]);
/// assert_eq!(arr[1.. 3], [  1,2    ]);
/// assert_eq!(arr[1..=3], [  1,2,3  ]);
/// ```
///
/// [`IntoIterator`]: ../iter/trait.Iterator.html
/// [`Iterator`]: ../iter/trait.IntoIterator.html
/// [slicing index]: ../slice/trait.SliceIndex.html
#[doc(alias = "..")]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RangeFull;

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for RangeFull {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "..")
    }
}

/// A (half-open) range bounded inclusively below and exclusively above
/// (`start..end`).
///
/// The `Range` `start..end` contains all values with `x >= start` and
/// `x < end`. It is empty unless `start < end`.
///
/// # Examples
///
/// ```
/// assert_eq!((3..5), std::ops::Range { start: 3, end: 5 });
/// assert_eq!(3 + 4 + 5, (3..6).sum());
///
/// let arr = [0, 1, 2, 3, 4];
/// assert_eq!(arr[ ..  ], [0,1,2,3,4]);
/// assert_eq!(arr[ .. 3], [0,1,2    ]);
/// assert_eq!(arr[ ..=3], [0,1,2,3  ]);
/// assert_eq!(arr[1..  ], [  1,2,3,4]);
/// assert_eq!(arr[1.. 3], [  1,2    ]);  // Range
/// assert_eq!(arr[1..=3], [  1,2,3  ]);
/// ```
#[doc(alias = "..")]
#[derive(Clone, PartialEq, Eq, Hash)] // not Copy -- see #27186
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
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.start.fmt(fmt)?;
        write!(fmt, "..")?;
        self.end.fmt(fmt)?;
        Ok(())
    }
}

impl<Idx: PartialOrd<Idx>> Range<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
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
    #[stable(feature = "range_contains", since = "1.35.0")]
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
/// let arr = [0, 1, 2, 3, 4];
/// assert_eq!(arr[ ..  ], [0,1,2,3,4]);
/// assert_eq!(arr[ .. 3], [0,1,2    ]);
/// assert_eq!(arr[ ..=3], [0,1,2,3  ]);
/// assert_eq!(arr[1..  ], [  1,2,3,4]);  // RangeFrom
/// assert_eq!(arr[1.. 3], [  1,2    ]);
/// assert_eq!(arr[1..=3], [  1,2,3  ]);
/// ```
///
/// [`Iterator`]: ../iter/trait.IntoIterator.html
#[doc(alias = "..")]
#[derive(Clone, PartialEq, Eq, Hash)] // not Copy -- see #27186
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RangeFrom<Idx> {
    /// The lower bound of the range (inclusive).
    #[stable(feature = "rust1", since = "1.0.0")]
    pub start: Idx,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<Idx: fmt::Debug> fmt::Debug for RangeFrom<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.start.fmt(fmt)?;
        write!(fmt, "..")?;
        Ok(())
    }
}

impl<Idx: PartialOrd<Idx>> RangeFrom<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
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
    #[stable(feature = "range_contains", since = "1.35.0")]
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
/// let arr = [0, 1, 2, 3, 4];
/// assert_eq!(arr[ ..  ], [0,1,2,3,4]);
/// assert_eq!(arr[ .. 3], [0,1,2    ]);  // RangeTo
/// assert_eq!(arr[ ..=3], [0,1,2,3  ]);
/// assert_eq!(arr[1..  ], [  1,2,3,4]);
/// assert_eq!(arr[1.. 3], [  1,2    ]);
/// assert_eq!(arr[1..=3], [  1,2,3  ]);
/// ```
///
/// [`IntoIterator`]: ../iter/trait.Iterator.html
/// [`Iterator`]: ../iter/trait.IntoIterator.html
/// [slicing index]: ../slice/trait.SliceIndex.html
#[doc(alias = "..")]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RangeTo<Idx> {
    /// The upper bound of the range (exclusive).
    #[stable(feature = "rust1", since = "1.0.0")]
    pub end: Idx,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<Idx: fmt::Debug> fmt::Debug for RangeTo<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "..")?;
        self.end.fmt(fmt)?;
        Ok(())
    }
}

impl<Idx: PartialOrd<Idx>> RangeTo<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
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
    #[stable(feature = "range_contains", since = "1.35.0")]
    pub fn contains<U>(&self, item: &U) -> bool
    where
        Idx: PartialOrd<U>,
        U: ?Sized + PartialOrd<Idx>,
    {
        <Self as RangeBounds<Idx>>::contains(self, item)
    }
}

/// A range bounded inclusively below and above (`start..=end`).
///
/// The `RangeInclusive` `start..=end` contains all values with `x >= start`
/// and `x <= end`. It is empty unless `start <= end`.
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
/// assert_eq!((3..=5), std::ops::RangeInclusive::new(3, 5));
/// assert_eq!(3 + 4 + 5, (3..=5).sum());
///
/// let arr = [0, 1, 2, 3, 4];
/// assert_eq!(arr[ ..  ], [0,1,2,3,4]);
/// assert_eq!(arr[ .. 3], [0,1,2    ]);
/// assert_eq!(arr[ ..=3], [0,1,2,3  ]);
/// assert_eq!(arr[1..  ], [  1,2,3,4]);
/// assert_eq!(arr[1.. 3], [  1,2    ]);
/// assert_eq!(arr[1..=3], [  1,2,3  ]);  // RangeInclusive
/// ```
#[doc(alias = "..=")]
#[derive(Clone)] // not Copy -- see #27186
#[stable(feature = "inclusive_range", since = "1.26.0")]
pub struct RangeInclusive<Idx> {
    pub(crate) start: Idx,
    pub(crate) end: Idx,
    pub(crate) is_empty: Option<bool>,
    // This field is:
    //  - `None` when next() or next_back() was never called
    //  - `Some(false)` when `start <= end` assuming no overflow
    //  - `Some(true)` otherwise
    // The field cannot be a simple `bool` because the `..=` constructor can
    // accept non-PartialOrd types, also we want the constructor to be const.
}

trait RangeInclusiveEquality: Sized {
    fn canonicalized_is_empty(range: &RangeInclusive<Self>) -> bool;
}

impl<T> RangeInclusiveEquality for T {
    #[inline]
    default fn canonicalized_is_empty(range: &RangeInclusive<Self>) -> bool {
        range.is_empty.unwrap_or_default()
    }
}

impl<T: PartialOrd> RangeInclusiveEquality for T {
    #[inline]
    fn canonicalized_is_empty(range: &RangeInclusive<Self>) -> bool {
        range.is_empty()
    }
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<Idx: PartialEq> PartialEq for RangeInclusive<Idx> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start
            && self.end == other.end
            && RangeInclusiveEquality::canonicalized_is_empty(self)
                == RangeInclusiveEquality::canonicalized_is_empty(other)
    }
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<Idx: Eq> Eq for RangeInclusive<Idx> {}

#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<Idx: Hash> Hash for RangeInclusive<Idx> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.start.hash(state);
        self.end.hash(state);
        RangeInclusiveEquality::canonicalized_is_empty(self).hash(state);
    }
}

impl<Idx> RangeInclusive<Idx> {
    /// Creates a new inclusive range. Equivalent to writing `start..=end`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::RangeInclusive;
    ///
    /// assert_eq!(3..=5, RangeInclusive::new(3, 5));
    /// ```
    #[stable(feature = "inclusive_range_methods", since = "1.27.0")]
    #[inline]
    #[rustc_promotable]
    pub const fn new(start: Idx, end: Idx) -> Self {
        Self {
            start,
            end,
            is_empty: None,
        }
    }

    /// Returns the lower bound of the range (inclusive).
    ///
    /// When using an inclusive range for iteration, the values of `start()` and
    /// [`end()`] are unspecified after the iteration ended. To determine
    /// whether the inclusive range is empty, use the [`is_empty()`] method
    /// instead of comparing `start() > end()`.
    ///
    /// Note: the value returned by this method is unspecified after the range
    /// has been iterated to exhaustion.
    ///
    /// [`end()`]: #method.end
    /// [`is_empty()`]: #method.is_empty
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!((3..=5).start(), &3);
    /// ```
    #[stable(feature = "inclusive_range_methods", since = "1.27.0")]
    #[inline]
    pub const fn start(&self) -> &Idx {
        &self.start
    }

    /// Returns the upper bound of the range (inclusive).
    ///
    /// When using an inclusive range for iteration, the values of [`start()`]
    /// and `end()` are unspecified after the iteration ended. To determine
    /// whether the inclusive range is empty, use the [`is_empty()`] method
    /// instead of comparing `start() > end()`.
    ///
    /// Note: the value returned by this method is unspecified after the range
    /// has been iterated to exhaustion.
    ///
    /// [`start()`]: #method.start
    /// [`is_empty()`]: #method.is_empty
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!((3..=5).end(), &5);
    /// ```
    #[stable(feature = "inclusive_range_methods", since = "1.27.0")]
    #[inline]
    pub const fn end(&self) -> &Idx {
        &self.end
    }

    /// Destructures the `RangeInclusive` into (lower bound, upper (inclusive) bound).
    ///
    /// Note: the value returned by this method is unspecified after the range
    /// has been iterated to exhaustion.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!((3..=5).into_inner(), (3, 5));
    /// ```
    #[stable(feature = "inclusive_range_methods", since = "1.27.0")]
    #[inline]
    pub fn into_inner(self) -> (Idx, Idx) {
        (self.start, self.end)
    }
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
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
    #[stable(feature = "range_contains", since = "1.35.0")]
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
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.is_empty.unwrap_or_else(|| !(self.start <= self.end))
    }

    // If this range's `is_empty` is field is unknown (`None`), update it to be a concrete value.
    #[inline]
    pub(crate) fn compute_is_empty(&mut self) {
        if self.is_empty.is_none() {
            self.is_empty = Some(!(self.start <= self.end));
        }
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
/// let arr = [0, 1, 2, 3, 4];
/// assert_eq!(arr[ ..  ], [0,1,2,3,4]);
/// assert_eq!(arr[ .. 3], [0,1,2    ]);
/// assert_eq!(arr[ ..=3], [0,1,2,3  ]);  // RangeToInclusive
/// assert_eq!(arr[1..  ], [  1,2,3,4]);
/// assert_eq!(arr[1.. 3], [  1,2    ]);
/// assert_eq!(arr[1..=3], [  1,2,3  ]);
/// ```
///
/// [`IntoIterator`]: ../iter/trait.Iterator.html
/// [`Iterator`]: ../iter/trait.IntoIterator.html
/// [slicing index]: ../slice/trait.SliceIndex.html
#[doc(alias = "..=")]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[stable(feature = "inclusive_range", since = "1.26.0")]
pub struct RangeToInclusive<Idx> {
    /// The upper bound of the range (inclusive)
    #[stable(feature = "inclusive_range", since = "1.26.0")]
    pub end: Idx,
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<Idx: fmt::Debug> fmt::Debug for RangeToInclusive<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "..=")?;
        self.end.fmt(fmt)?;
        Ok(())
    }
}

impl<Idx: PartialOrd<Idx>> RangeToInclusive<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
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
    #[stable(feature = "range_contains", since = "1.35.0")]
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
/// use std::ops::Bound::*;
/// use std::ops::RangeBounds;
///
/// assert_eq!((..100).start_bound(), Unbounded);
/// assert_eq!((1..12).start_bound(), Included(&1));
/// assert_eq!((1..12).end_bound(), Excluded(&12));
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

impl<T: Clone> Bound<&T> {
    /// Map a `Bound<&T>` to a `Bound<T>` by cloning the contents of the bound.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bound_cloned)]
    /// use std::ops::Bound::*;
    /// use std::ops::RangeBounds;
    ///
    /// assert_eq!((1..12).start_bound(), Included(&1));
    /// assert_eq!((1..12).start_bound().cloned(), Included(1));
    /// ```
    #[unstable(feature = "bound_cloned", issue = "61356")]
    pub fn cloned(self) -> Bound<T> {
        match self {
            Bound::Unbounded => Bound::Unbounded,
            Bound::Included(x) => Bound::Included(x.clone()),
            Bound::Excluded(x) => Bound::Excluded(x.clone()),
        }
    }
}

#[stable(feature = "collections_range", since = "1.28.0")]
/// `RangeBounds` is implemented by Rust's built-in range types, produced
/// by range syntax like `..`, `a..`, `..b`, `..=c`, `d..e`, or `f..=g`.
pub trait RangeBounds<T: ?Sized> {
    /// Start index bound.
    ///
    /// Returns the start value as a `Bound`.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() {
    /// use std::ops::Bound::*;
    /// use std::ops::RangeBounds;
    ///
    /// assert_eq!((..10).start_bound(), Unbounded);
    /// assert_eq!((3..10).start_bound(), Included(&3));
    /// # }
    /// ```
    #[stable(feature = "collections_range", since = "1.28.0")]
    fn start_bound(&self) -> Bound<&T>;

    /// End index bound.
    ///
    /// Returns the end value as a `Bound`.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() {
    /// use std::ops::Bound::*;
    /// use std::ops::RangeBounds;
    ///
    /// assert_eq!((3..).end_bound(), Unbounded);
    /// assert_eq!((3..10).end_bound(), Excluded(&10));
    /// # }
    /// ```
    #[stable(feature = "collections_range", since = "1.28.0")]
    fn end_bound(&self) -> Bound<&T>;

    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f32;
    ///
    /// assert!( (3..5).contains(&4));
    /// assert!(!(3..5).contains(&2));
    ///
    /// assert!( (0.0..1.0).contains(&0.5));
    /// assert!(!(0.0..1.0).contains(&f32::NAN));
    /// assert!(!(0.0..f32::NAN).contains(&0.5));
    /// assert!(!(f32::NAN..1.0).contains(&0.5));
    #[stable(feature = "range_contains", since = "1.35.0")]
    fn contains<U>(&self, item: &U) -> bool
    where
        T: PartialOrd<U>,
        U: ?Sized + PartialOrd<T>,
    {
        (match self.start_bound() {
            Included(ref start) => *start <= item,
            Excluded(ref start) => *start < item,
            Unbounded => true,
        }) && (match self.end_bound() {
            Included(ref end) => item <= *end,
            Excluded(ref end) => item < *end,
            Unbounded => true,
        })
    }
}

use self::Bound::{Excluded, Included, Unbounded};

#[stable(feature = "collections_range", since = "1.28.0")]
impl<T: ?Sized> RangeBounds<T> for RangeFull {
    fn start_bound(&self) -> Bound<&T> {
        Unbounded
    }
    fn end_bound(&self) -> Bound<&T> {
        Unbounded
    }
}

#[stable(feature = "collections_range", since = "1.28.0")]
impl<T> RangeBounds<T> for RangeFrom<T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(&self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        Unbounded
    }
}

#[stable(feature = "collections_range", since = "1.28.0")]
impl<T> RangeBounds<T> for RangeTo<T> {
    fn start_bound(&self) -> Bound<&T> {
        Unbounded
    }
    fn end_bound(&self) -> Bound<&T> {
        Excluded(&self.end)
    }
}

#[stable(feature = "collections_range", since = "1.28.0")]
impl<T> RangeBounds<T> for Range<T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(&self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        Excluded(&self.end)
    }
}

#[stable(feature = "collections_range", since = "1.28.0")]
impl<T> RangeBounds<T> for RangeInclusive<T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(&self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        Included(&self.end)
    }
}

#[stable(feature = "collections_range", since = "1.28.0")]
impl<T> RangeBounds<T> for RangeToInclusive<T> {
    fn start_bound(&self) -> Bound<&T> {
        Unbounded
    }
    fn end_bound(&self) -> Bound<&T> {
        Included(&self.end)
    }
}

#[stable(feature = "collections_range", since = "1.28.0")]
impl<T> RangeBounds<T> for (Bound<T>, Bound<T>) {
    fn start_bound(&self) -> Bound<&T> {
        match *self {
            (Included(ref start), _) => Included(start),
            (Excluded(ref start), _) => Excluded(start),
            (Unbounded, _) => Unbounded,
        }
    }

    fn end_bound(&self) -> Bound<&T> {
        match *self {
            (_, Included(ref end)) => Included(end),
            (_, Excluded(ref end)) => Excluded(end),
            (_, Unbounded) => Unbounded,
        }
    }
}

#[stable(feature = "collections_range", since = "1.28.0")]
impl<'a, T: ?Sized + 'a> RangeBounds<T> for (Bound<&'a T>, Bound<&'a T>) {
    fn start_bound(&self) -> Bound<&T> {
        self.0
    }

    fn end_bound(&self) -> Bound<&T> {
        self.1
    }
}

#[stable(feature = "collections_range", since = "1.28.0")]
impl<T> RangeBounds<T> for RangeFrom<&T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        Unbounded
    }
}

#[stable(feature = "collections_range", since = "1.28.0")]
impl<T> RangeBounds<T> for RangeTo<&T> {
    fn start_bound(&self) -> Bound<&T> {
        Unbounded
    }
    fn end_bound(&self) -> Bound<&T> {
        Excluded(self.end)
    }
}

#[stable(feature = "collections_range", since = "1.28.0")]
impl<T> RangeBounds<T> for Range<&T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        Excluded(self.end)
    }
}

#[stable(feature = "collections_range", since = "1.28.0")]
impl<T> RangeBounds<T> for RangeInclusive<&T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        Included(self.end)
    }
}

#[stable(feature = "collections_range", since = "1.28.0")]
impl<T> RangeBounds<T> for RangeToInclusive<&T> {
    fn start_bound(&self) -> Bound<&T> {
        Unbounded
    }
    fn end_bound(&self) -> Bound<&T> {
        Included(self.end)
    }
}
