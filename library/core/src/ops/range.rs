use crate::fmt;
use crate::hash::Hash;

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
/// assert_eq!(.., std::ops::RangeFull);
/// ```
///
/// It does not have an [`IntoIterator`] implementation, so you can't use it in
/// a `for` loop directly. This won't compile:
///
/// ```compile_fail,E0277
/// for i in .. {
///     // ...
/// }
/// ```
///
/// Used as a [slicing index], `RangeFull` produces the full array as a slice.
///
/// ```
/// let arr = [0, 1, 2, 3, 4];
/// assert_eq!(arr[ ..  ], [0, 1, 2, 3, 4]); // This is the `RangeFull`
/// assert_eq!(arr[ .. 3], [0, 1, 2      ]);
/// assert_eq!(arr[ ..=3], [0, 1, 2, 3   ]);
/// assert_eq!(arr[1..  ], [   1, 2, 3, 4]);
/// assert_eq!(arr[1.. 3], [   1, 2      ]);
/// assert_eq!(arr[1..=3], [   1, 2, 3   ]);
/// ```
///
/// [slicing index]: crate::slice::SliceIndex
#[lang = "RangeFull"]
#[doc(alias = "..")]
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
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
/// The range `start..end` contains all values with `start <= x < end`.
/// It is empty if `start >= end`.
///
/// # Examples
///
/// The `start..end` syntax is a `Range`:
///
/// ```
/// assert_eq!((3..5), std::ops::Range { start: 3, end: 5 });
/// assert_eq!(3 + 4 + 5, (3..6).sum());
/// ```
///
/// ```
/// let arr = [0, 1, 2, 3, 4];
/// assert_eq!(arr[ ..  ], [0, 1, 2, 3, 4]);
/// assert_eq!(arr[ .. 3], [0, 1, 2      ]);
/// assert_eq!(arr[ ..=3], [0, 1, 2, 3   ]);
/// assert_eq!(arr[1..  ], [   1, 2, 3, 4]);
/// assert_eq!(arr[1.. 3], [   1, 2      ]); // This is a `Range`
/// assert_eq!(arr[1..=3], [   1, 2, 3   ]);
/// ```
#[lang = "Range"]
#[doc(alias = "..")]
#[derive(Clone, Default, PartialEq, Eq, Hash)] // not Copy -- see #27186
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
    #[inline]
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
    /// assert!(!(3..5).is_empty());
    /// assert!( (3..3).is_empty());
    /// assert!( (3..2).is_empty());
    /// ```
    ///
    /// The range is empty if either side is incomparable:
    ///
    /// ```
    /// assert!(!(3.0..5.0).is_empty());
    /// assert!( (3.0..f32::NAN).is_empty());
    /// assert!( (f32::NAN..5.0).is_empty());
    /// ```
    #[inline]
    #[stable(feature = "range_is_empty", since = "1.47.0")]
    pub fn is_empty(&self) -> bool {
        !(self.start < self.end)
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
/// assert_eq!((2..), std::ops::RangeFrom { start: 2 });
/// assert_eq!(2 + 3 + 4, (2..).take(3).sum());
/// ```
///
/// ```
/// let arr = [0, 1, 2, 3, 4];
/// assert_eq!(arr[ ..  ], [0, 1, 2, 3, 4]);
/// assert_eq!(arr[ .. 3], [0, 1, 2      ]);
/// assert_eq!(arr[ ..=3], [0, 1, 2, 3   ]);
/// assert_eq!(arr[1..  ], [   1, 2, 3, 4]); // This is a `RangeFrom`
/// assert_eq!(arr[1.. 3], [   1, 2      ]);
/// assert_eq!(arr[1..=3], [   1, 2, 3   ]);
/// ```
#[lang = "RangeFrom"]
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
    /// assert!(!(3..).contains(&2));
    /// assert!( (3..).contains(&3));
    /// assert!( (3..).contains(&1_000_000_000));
    ///
    /// assert!( (0.0..).contains(&0.5));
    /// assert!(!(0.0..).contains(&f32::NAN));
    /// assert!(!(f32::NAN..).contains(&0.5));
    /// ```
    #[inline]
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
/// assert_eq!(arr[ ..  ], [0, 1, 2, 3, 4]);
/// assert_eq!(arr[ .. 3], [0, 1, 2      ]); // This is a `RangeTo`
/// assert_eq!(arr[ ..=3], [0, 1, 2, 3   ]);
/// assert_eq!(arr[1..  ], [   1, 2, 3, 4]);
/// assert_eq!(arr[1.. 3], [   1, 2      ]);
/// assert_eq!(arr[1..=3], [   1, 2, 3   ]);
/// ```
///
/// [slicing index]: crate::slice::SliceIndex
#[lang = "RangeTo"]
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
    /// assert!( (..5).contains(&-1_000_000_000));
    /// assert!( (..5).contains(&4));
    /// assert!(!(..5).contains(&5));
    ///
    /// assert!( (..1.0).contains(&0.5));
    /// assert!(!(..1.0).contains(&f32::NAN));
    /// assert!(!(..f32::NAN).contains(&0.5));
    /// ```
    #[inline]
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
/// [fused]: crate::iter::FusedIterator
/// [`.is_empty()`]: RangeInclusive::is_empty
///
/// # Examples
///
/// The `start..=end` syntax is a `RangeInclusive`:
///
/// ```
/// assert_eq!((3..=5), std::ops::RangeInclusive::new(3, 5));
/// assert_eq!(3 + 4 + 5, (3..=5).sum());
/// ```
///
/// ```
/// let arr = [0, 1, 2, 3, 4];
/// assert_eq!(arr[ ..  ], [0, 1, 2, 3, 4]);
/// assert_eq!(arr[ .. 3], [0, 1, 2      ]);
/// assert_eq!(arr[ ..=3], [0, 1, 2, 3   ]);
/// assert_eq!(arr[1..  ], [   1, 2, 3, 4]);
/// assert_eq!(arr[1.. 3], [   1, 2      ]);
/// assert_eq!(arr[1..=3], [   1, 2, 3   ]); // This is a `RangeInclusive`
/// ```
#[lang = "RangeInclusive"]
#[doc(alias = "..=")]
#[derive(Clone, PartialEq, Eq, Hash)] // not Copy -- see #27186
#[stable(feature = "inclusive_range", since = "1.26.0")]
pub struct RangeInclusive<Idx> {
    // Note that the fields here are not public to allow changing the
    // representation in the future; in particular, while we could plausibly
    // expose start/end, modifying them without changing (future/current)
    // private fields may lead to incorrect behavior, so we don't want to
    // support that mode.
    pub(crate) start: Idx,
    pub(crate) end: Idx,

    // This field is:
    //  - `false` upon construction
    //  - `false` when iteration has yielded an element and the iterator is not exhausted
    //  - `true` when iteration has been used to exhaust the iterator
    //
    // This is required to support PartialEq and Hash without a PartialOrd bound or specialization.
    pub(crate) exhausted: bool,
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
    #[lang = "range_inclusive_new"]
    #[stable(feature = "inclusive_range_methods", since = "1.27.0")]
    #[inline]
    #[rustc_promotable]
    #[rustc_const_stable(feature = "const_range_new", since = "1.32.0")]
    pub const fn new(start: Idx, end: Idx) -> Self {
        Self { start, end, exhausted: false }
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
    /// [`end()`]: RangeInclusive::end
    /// [`is_empty()`]: RangeInclusive::is_empty
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!((3..=5).start(), &3);
    /// ```
    #[stable(feature = "inclusive_range_methods", since = "1.27.0")]
    #[rustc_const_stable(feature = "const_inclusive_range_methods", since = "1.32.0")]
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
    /// [`start()`]: RangeInclusive::start
    /// [`is_empty()`]: RangeInclusive::is_empty
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!((3..=5).end(), &5);
    /// ```
    #[stable(feature = "inclusive_range_methods", since = "1.27.0")]
    #[rustc_const_stable(feature = "const_inclusive_range_methods", since = "1.32.0")]
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
    #[rustc_const_unstable(feature = "const_range_bounds", issue = "108082")]
    pub const fn into_inner(self) -> (Idx, Idx) {
        (self.start, self.end)
    }
}

impl RangeInclusive<usize> {
    /// Converts to an exclusive `Range` for `SliceIndex` implementations.
    /// The caller is responsible for dealing with `end == usize::MAX`.
    #[inline]
    pub(crate) const fn into_slice_range(self) -> Range<usize> {
        // If we're not exhausted, we want to simply slice `start..end + 1`.
        // If we are exhausted, then slicing with `end + 1..end + 1` gives us an
        // empty range that is still subject to bounds-checks for that endpoint.
        let exclusive_end = self.end + 1;
        let start = if self.exhausted { exclusive_end } else { self.start };
        start..exclusive_end
    }
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<Idx: fmt::Debug> fmt::Debug for RangeInclusive<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.start.fmt(fmt)?;
        write!(fmt, "..=")?;
        self.end.fmt(fmt)?;
        if self.exhausted {
            write!(fmt, " (exhausted)")?;
        }
        Ok(())
    }
}

impl<Idx: PartialOrd<Idx>> RangeInclusive<Idx> {
    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
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
    ///
    /// This method always returns `false` after iteration has finished:
    ///
    /// ```
    /// let mut r = 3..=5;
    /// assert!(r.contains(&3) && r.contains(&5));
    /// for _ in r.by_ref() {}
    /// // Precise field values are unspecified here
    /// assert!(!r.contains(&3) && !r.contains(&5));
    /// ```
    #[inline]
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
    /// assert!(!(3..=5).is_empty());
    /// assert!(!(3..=3).is_empty());
    /// assert!( (3..=2).is_empty());
    /// ```
    ///
    /// The range is empty if either side is incomparable:
    ///
    /// ```
    /// assert!(!(3.0..=5.0).is_empty());
    /// assert!( (3.0..=f32::NAN).is_empty());
    /// assert!( (f32::NAN..=5.0).is_empty());
    /// ```
    ///
    /// This method returns `true` after iteration has finished:
    ///
    /// ```
    /// let mut r = 3..=5;
    /// for _ in r.by_ref() {}
    /// // Precise field values are unspecified here
    /// assert!(r.is_empty());
    /// ```
    #[stable(feature = "range_is_empty", since = "1.47.0")]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.exhausted || !(self.start <= self.end)
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
/// assert_eq!(arr[ ..  ], [0, 1, 2, 3, 4]);
/// assert_eq!(arr[ .. 3], [0, 1, 2      ]);
/// assert_eq!(arr[ ..=3], [0, 1, 2, 3   ]); // This is a `RangeToInclusive`
/// assert_eq!(arr[1..  ], [   1, 2, 3, 4]);
/// assert_eq!(arr[1.. 3], [   1, 2      ]);
/// assert_eq!(arr[1..=3], [   1, 2, 3   ]);
/// ```
///
/// [slicing index]: crate::slice::SliceIndex
#[lang = "RangeToInclusive"]
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
    /// assert!( (..=5).contains(&-1_000_000_000));
    /// assert!( (..=5).contains(&5));
    /// assert!(!(..=5).contains(&6));
    ///
    /// assert!( (..=1.0).contains(&1.0));
    /// assert!(!(..=1.0).contains(&f32::NAN));
    /// assert!(!(..=f32::NAN).contains(&0.5));
    /// ```
    #[inline]
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
///     println!("{key}: {value}");
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

impl<T> Bound<T> {
    /// Converts from `&Bound<T>` to `Bound<&T>`.
    #[inline]
    #[stable(feature = "bound_as_ref_shared", since = "1.65.0")]
    pub fn as_ref(&self) -> Bound<&T> {
        match *self {
            Included(ref x) => Included(x),
            Excluded(ref x) => Excluded(x),
            Unbounded => Unbounded,
        }
    }

    /// Converts from `&mut Bound<T>` to `Bound<&mut T>`.
    #[inline]
    #[unstable(feature = "bound_as_ref", issue = "80996")]
    pub fn as_mut(&mut self) -> Bound<&mut T> {
        match *self {
            Included(ref mut x) => Included(x),
            Excluded(ref mut x) => Excluded(x),
            Unbounded => Unbounded,
        }
    }

    /// Maps a `Bound<T>` to a `Bound<U>` by applying a function to the contained value (including
    /// both `Included` and `Excluded`), returning a `Bound` of the same kind.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::Bound::*;
    ///
    /// let bound_string = Included("Hello, World!");
    ///
    /// assert_eq!(bound_string.map(|s| s.len()), Included(13));
    /// ```
    ///
    /// ```
    /// use std::ops::Bound;
    /// use Bound::*;
    ///
    /// let unbounded_string: Bound<String> = Unbounded;
    ///
    /// assert_eq!(unbounded_string.map(|s| s.len()), Unbounded);
    /// ```
    #[inline]
    #[stable(feature = "bound_map", since = "1.77.0")]
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Bound<U> {
        match self {
            Unbounded => Unbounded,
            Included(x) => Included(f(x)),
            Excluded(x) => Excluded(f(x)),
        }
    }
}

impl<T: Clone> Bound<&T> {
    /// Map a `Bound<&T>` to a `Bound<T>` by cloning the contents of the bound.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::Bound::*;
    /// use std::ops::RangeBounds;
    ///
    /// assert_eq!((1..12).start_bound(), Included(&1));
    /// assert_eq!((1..12).start_bound().cloned(), Included(1));
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "bound_cloned", since = "1.55.0")]
    pub fn cloned(self) -> Bound<T> {
        match self {
            Bound::Unbounded => Bound::Unbounded,
            Bound::Included(x) => Bound::Included(x.clone()),
            Bound::Excluded(x) => Bound::Excluded(x.clone()),
        }
    }
}

/// `RangeBounds` is implemented by Rust's built-in range types, produced
/// by range syntax like `..`, `a..`, `..b`, `..=c`, `d..e`, or `f..=g`.
#[stable(feature = "collections_range", since = "1.28.0")]
#[rustc_diagnostic_item = "RangeBounds"]
pub trait RangeBounds<T: ?Sized> {
    /// Start index bound.
    ///
    /// Returns the start value as a `Bound`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::Bound::*;
    /// use std::ops::RangeBounds;
    ///
    /// assert_eq!((..10).start_bound(), Unbounded);
    /// assert_eq!((3..10).start_bound(), Included(&3));
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
    /// use std::ops::Bound::*;
    /// use std::ops::RangeBounds;
    ///
    /// assert_eq!((3..).end_bound(), Unbounded);
    /// assert_eq!((3..10).end_bound(), Excluded(&10));
    /// ```
    #[stable(feature = "collections_range", since = "1.28.0")]
    fn end_bound(&self) -> Bound<&T>;

    /// Returns `true` if `item` is contained in the range.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!( (3..5).contains(&4));
    /// assert!(!(3..5).contains(&2));
    ///
    /// assert!( (0.0..1.0).contains(&0.5));
    /// assert!(!(0.0..1.0).contains(&f32::NAN));
    /// assert!(!(0.0..f32::NAN).contains(&0.5));
    /// assert!(!(f32::NAN..1.0).contains(&0.5));
    #[inline]
    #[stable(feature = "range_contains", since = "1.35.0")]
    fn contains<U>(&self, item: &U) -> bool
    where
        T: PartialOrd<U>,
        U: ?Sized + PartialOrd<T>,
    {
        (match self.start_bound() {
            Included(start) => start <= item,
            Excluded(start) => start < item,
            Unbounded => true,
        }) && (match self.end_bound() {
            Included(end) => item <= end,
            Excluded(end) => item < end,
            Unbounded => true,
        })
    }

    /// Returns `true` if the range contains no items.
    /// One-sided ranges (`RangeFrom`, etc) always return `true`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(range_bounds_is_empty)]
    /// use std::ops::RangeBounds;
    ///
    /// assert!(!(3..).is_empty());
    /// assert!(!(..2).is_empty());
    /// assert!(!RangeBounds::is_empty(&(3..5)));
    /// assert!( RangeBounds::is_empty(&(3..3)));
    /// assert!( RangeBounds::is_empty(&(3..2)));
    /// ```
    ///
    /// The range is empty if either side is incomparable:
    ///
    /// ```
    /// #![feature(range_bounds_is_empty)]
    /// use std::ops::RangeBounds;
    ///
    /// assert!(!RangeBounds::is_empty(&(3.0..5.0)));
    /// assert!( RangeBounds::is_empty(&(3.0..f32::NAN)));
    /// assert!( RangeBounds::is_empty(&(f32::NAN..5.0)));
    /// ```
    ///
    /// But never empty is either side is unbounded:
    ///
    /// ```
    /// #![feature(range_bounds_is_empty)]
    /// use std::ops::RangeBounds;
    ///
    /// assert!(!(..0).is_empty());
    /// assert!(!(i32::MAX..).is_empty());
    /// assert!(!RangeBounds::<u8>::is_empty(&(..)));
    /// ```
    ///
    /// `(Excluded(a), Excluded(b))` is only empty if `a >= b`:
    ///
    /// ```
    /// #![feature(range_bounds_is_empty)]
    /// use std::ops::Bound::*;
    /// use std::ops::RangeBounds;
    ///
    /// assert!(!(Excluded(1), Excluded(3)).is_empty());
    /// assert!(!(Excluded(1), Excluded(2)).is_empty());
    /// assert!( (Excluded(1), Excluded(1)).is_empty());
    /// assert!( (Excluded(2), Excluded(1)).is_empty());
    /// assert!( (Excluded(3), Excluded(1)).is_empty());
    /// ```
    #[unstable(feature = "range_bounds_is_empty", issue = "137300")]
    fn is_empty(&self) -> bool
    where
        T: PartialOrd,
    {
        !match (self.start_bound(), self.end_bound()) {
            (Unbounded, _) | (_, Unbounded) => true,
            (Included(start), Excluded(end))
            | (Excluded(start), Included(end))
            | (Excluded(start), Excluded(end)) => start < end,
            (Included(start), Included(end)) => start <= end,
        }
    }
}

/// Used to convert a range into start and end bounds, consuming the
/// range by value.
///
/// `IntoBounds` is implemented by Rust’s built-in range types, produced
/// by range syntax like `..`, `a..`, `..b`, `..=c`, `d..e`, or `f..=g`.
#[unstable(feature = "range_into_bounds", issue = "136903")]
pub trait IntoBounds<T>: RangeBounds<T> {
    /// Convert this range into the start and end bounds.
    /// Returns `(start_bound, end_bound)`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(range_into_bounds)]
    /// use std::ops::Bound::*;
    /// use std::ops::IntoBounds;
    ///
    /// assert_eq!((0..5).into_bounds(), (Included(0), Excluded(5)));
    /// assert_eq!((..=7).into_bounds(), (Unbounded, Included(7)));
    /// ```
    fn into_bounds(self) -> (Bound<T>, Bound<T>);

    /// Compute the intersection of  `self` and `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(range_into_bounds)]
    /// use std::ops::Bound::*;
    /// use std::ops::IntoBounds;
    ///
    /// assert_eq!((3..).intersect(..5), (Included(3), Excluded(5)));
    /// assert_eq!((-12..387).intersect(0..256), (Included(0), Excluded(256)));
    /// assert_eq!((1..5).intersect(..), (Included(1), Excluded(5)));
    /// assert_eq!((1..=9).intersect(0..10), (Included(1), Included(9)));
    /// assert_eq!((7..=13).intersect(8..13), (Included(8), Excluded(13)));
    /// ```
    ///
    /// Combine with `is_empty` to determine if two ranges overlap.
    ///
    /// ```
    /// #![feature(range_into_bounds)]
    /// #![feature(range_bounds_is_empty)]
    /// use std::ops::{RangeBounds, IntoBounds};
    ///
    /// assert!(!(3..).intersect(..5).is_empty());
    /// assert!(!(-12..387).intersect(0..256).is_empty());
    /// assert!((1..5).intersect(6..).is_empty());
    /// ```
    fn intersect<R>(self, other: R) -> (Bound<T>, Bound<T>)
    where
        Self: Sized,
        T: Ord,
        R: Sized + IntoBounds<T>,
    {
        let (self_start, self_end) = IntoBounds::into_bounds(self);
        let (other_start, other_end) = IntoBounds::into_bounds(other);

        let start = match (self_start, other_start) {
            (Included(a), Included(b)) => Included(Ord::max(a, b)),
            (Excluded(a), Excluded(b)) => Excluded(Ord::max(a, b)),
            (Unbounded, Unbounded) => Unbounded,

            (x, Unbounded) | (Unbounded, x) => x,

            (Included(i), Excluded(e)) | (Excluded(e), Included(i)) => {
                if i > e {
                    Included(i)
                } else {
                    Excluded(e)
                }
            }
        };
        let end = match (self_end, other_end) {
            (Included(a), Included(b)) => Included(Ord::min(a, b)),
            (Excluded(a), Excluded(b)) => Excluded(Ord::min(a, b)),
            (Unbounded, Unbounded) => Unbounded,

            (x, Unbounded) | (Unbounded, x) => x,

            (Included(i), Excluded(e)) | (Excluded(e), Included(i)) => {
                if i < e {
                    Included(i)
                } else {
                    Excluded(e)
                }
            }
        };

        (start, end)
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

#[unstable(feature = "range_into_bounds", issue = "136903")]
impl<T> IntoBounds<T> for RangeFull {
    fn into_bounds(self) -> (Bound<T>, Bound<T>) {
        (Unbounded, Unbounded)
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

#[unstable(feature = "range_into_bounds", issue = "136903")]
impl<T> IntoBounds<T> for RangeFrom<T> {
    fn into_bounds(self) -> (Bound<T>, Bound<T>) {
        (Included(self.start), Unbounded)
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

#[unstable(feature = "range_into_bounds", issue = "136903")]
impl<T> IntoBounds<T> for RangeTo<T> {
    fn into_bounds(self) -> (Bound<T>, Bound<T>) {
        (Unbounded, Excluded(self.end))
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

#[unstable(feature = "range_into_bounds", issue = "136903")]
impl<T> IntoBounds<T> for Range<T> {
    fn into_bounds(self) -> (Bound<T>, Bound<T>) {
        (Included(self.start), Excluded(self.end))
    }
}

#[stable(feature = "collections_range", since = "1.28.0")]
impl<T> RangeBounds<T> for RangeInclusive<T> {
    fn start_bound(&self) -> Bound<&T> {
        Included(&self.start)
    }
    fn end_bound(&self) -> Bound<&T> {
        if self.exhausted {
            // When the iterator is exhausted, we usually have start == end,
            // but we want the range to appear empty, containing nothing.
            Excluded(&self.end)
        } else {
            Included(&self.end)
        }
    }
}

#[unstable(feature = "range_into_bounds", issue = "136903")]
impl<T> IntoBounds<T> for RangeInclusive<T> {
    fn into_bounds(self) -> (Bound<T>, Bound<T>) {
        (
            Included(self.start),
            if self.exhausted {
                // When the iterator is exhausted, we usually have start == end,
                // but we want the range to appear empty, containing nothing.
                Excluded(self.end)
            } else {
                Included(self.end)
            },
        )
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

#[unstable(feature = "range_into_bounds", issue = "136903")]
impl<T> IntoBounds<T> for RangeToInclusive<T> {
    fn into_bounds(self) -> (Bound<T>, Bound<T>) {
        (Unbounded, Included(self.end))
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

#[unstable(feature = "range_into_bounds", issue = "136903")]
impl<T> IntoBounds<T> for (Bound<T>, Bound<T>) {
    fn into_bounds(self) -> (Bound<T>, Bound<T>) {
        self
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

/// An internal helper for `split_off` functions indicating
/// which end a `OneSidedRange` is bounded on.
#[unstable(feature = "one_sided_range", issue = "69780")]
#[allow(missing_debug_implementations)]
pub enum OneSidedRangeBound {
    /// The range is bounded inclusively from below and is unbounded above.
    StartInclusive,
    /// The range is bounded exclusively from above and is unbounded below.
    End,
    /// The range is bounded inclusively from above and is unbounded below.
    EndInclusive,
}

/// `OneSidedRange` is implemented for built-in range types that are unbounded
/// on one side. For example, `a..`, `..b` and `..=c` implement `OneSidedRange`,
/// but `..`, `d..e`, and `f..=g` do not.
///
/// Types that implement `OneSidedRange<T>` must return `Bound::Unbounded`
/// from one of `RangeBounds::start_bound` or `RangeBounds::end_bound`.
#[unstable(feature = "one_sided_range", issue = "69780")]
pub trait OneSidedRange<T: ?Sized>: RangeBounds<T> {
    /// An internal-only helper function for `split_off` and
    /// `split_off_mut` that returns the bound of the one-sided range.
    fn bound(self) -> (OneSidedRangeBound, T);
}

#[unstable(feature = "one_sided_range", issue = "69780")]
impl<T> OneSidedRange<T> for RangeTo<T>
where
    Self: RangeBounds<T>,
{
    fn bound(self) -> (OneSidedRangeBound, T) {
        (OneSidedRangeBound::End, self.end)
    }
}

#[unstable(feature = "one_sided_range", issue = "69780")]
impl<T> OneSidedRange<T> for RangeFrom<T>
where
    Self: RangeBounds<T>,
{
    fn bound(self) -> (OneSidedRangeBound, T) {
        (OneSidedRangeBound::StartInclusive, self.start)
    }
}

#[unstable(feature = "one_sided_range", issue = "69780")]
impl<T> OneSidedRange<T> for RangeToInclusive<T>
where
    Self: RangeBounds<T>,
{
    fn bound(self) -> (OneSidedRangeBound, T) {
        (OneSidedRangeBound::EndInclusive, self.end)
    }
}
