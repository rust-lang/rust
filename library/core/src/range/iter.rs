use crate::iter::{
    FusedIterator, Step, TrustedLen, TrustedRandomAccess, TrustedRandomAccessNoCoerce, TrustedStep,
};
use crate::mem;
use crate::num::NonZero;
use crate::range::{Range, RangeFrom, RangeInclusive, legacy};

/// By-value [`Range`] iterator.
#[stable(feature = "new_range_api", since = "1.96.0")]
#[derive(Debug, Clone)]
pub struct RangeIter<A>(pub(crate) legacy::Range<A>);

impl<A> RangeIter<A> {
    #[unstable(feature = "new_range_remainder", issue = "154458")]
    /// Returns the remainder of the range being iterated over.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(new_range_remainder)]
    ///
    /// let range = core::range::Range::from(3..11);
    /// let mut iter = range.into_iter();
    /// assert_eq!(iter.clone().remainder(), range);
    /// iter.next();
    /// assert_eq!(iter.clone().remainder(), core::range::Range::from(4..11));
    /// iter.by_ref().for_each(drop);
    /// assert!(iter.remainder().is_empty());
    /// ```
    pub fn remainder(self) -> Range<A> {
        Range { start: self.0.start, end: self.0.end }
    }
}

/// Safety: This macro must only be used on types that are `Copy` and result in ranges
/// which have an exact `size_hint()` where the upper bound must not be `None`.
macro_rules! unsafe_range_trusted_random_access_impl {
    ($($t:ty)*) => ($(
        #[doc(hidden)]
        #[unstable(feature = "trusted_random_access", issue = "none")]
        unsafe impl TrustedRandomAccess for RangeIter<$t> {}

        #[doc(hidden)]
        #[unstable(feature = "trusted_random_access", issue = "none")]
        unsafe impl TrustedRandomAccessNoCoerce for RangeIter<$t> {
            const MAY_HAVE_SIDE_EFFECT: bool = false;
        }
    )*)
}

unsafe_range_trusted_random_access_impl! {
    usize u8 u16
    isize i8 i16
}

#[cfg(target_pointer_width = "32")]
unsafe_range_trusted_random_access_impl! {
    u32 i32
}

#[cfg(target_pointer_width = "64")]
unsafe_range_trusted_random_access_impl! {
    u32 i32
    u64 i64
}

#[stable(feature = "new_range_api", since = "1.96.0")]
impl<A: Step> Iterator for RangeIter<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.0.count()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<A> {
        self.0.nth(n)
    }

    #[inline]
    fn last(self) -> Option<A> {
        self.0.last()
    }

    #[inline]
    fn min(self) -> Option<A>
    where
        A: Ord,
    {
        self.0.min()
    }

    #[inline]
    fn max(self) -> Option<A>
    where
        A: Ord,
    {
        self.0.max()
    }

    #[inline]
    fn is_sorted(self) -> bool {
        true
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.0.advance_by(n)
    }

    #[inline]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item
    where
        Self: TrustedRandomAccessNoCoerce,
    {
        // SAFETY: The TrustedRandomAccess contract requires that callers only pass an index
        // that is in bounds.
        // Additionally Self: TrustedRandomAccess is only implemented for Copy types
        // which means even repeated reads of the same index would be safe.
        unsafe { Step::forward_unchecked(self.0.start.clone(), idx) }
    }
}

#[stable(feature = "new_range_api", since = "1.96.0")]
impl<A: Step> DoubleEndedIterator for RangeIter<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        self.0.next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<A> {
        self.0.nth_back(n)
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.0.advance_back_by(n)
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: TrustedStep> TrustedLen for RangeIter<A> {}

#[stable(feature = "new_range_api", since = "1.96.0")]
impl<A: Step> FusedIterator for RangeIter<A> {}

#[stable(feature = "new_range_api", since = "1.96.0")]
impl<A: Step> IntoIterator for Range<A> {
    type Item = A;
    type IntoIter = RangeIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        RangeIter(self.into())
    }
}

/// By-value [`RangeInclusive`] iterator.
#[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
#[derive(Debug, Clone)]
pub struct RangeInclusiveIter<A>(legacy::RangeInclusive<A>);

impl<A: Step> RangeInclusiveIter<A> {
    /// Returns the remainder of the range being iterated over.
    ///
    /// If the iterator is exhausted or empty, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(new_range_remainder)]
    ///
    /// let range = core::range::RangeInclusive::from(3..=11);
    /// let mut iter = range.into_iter();
    /// assert_eq!(iter.clone().remainder().unwrap(), range);
    /// iter.next();
    /// assert_eq!(iter.clone().remainder().unwrap(), core::range::RangeInclusive::from(4..=11));
    /// iter.by_ref().for_each(drop);
    /// assert!(iter.remainder().is_none());
    /// ```
    #[unstable(feature = "new_range_remainder", issue = "154458")]
    pub fn remainder(self) -> Option<RangeInclusive<A>> {
        if self.0.is_empty() {
            return None;
        }

        Some(RangeInclusive { start: self.0.start, last: self.0.end })
    }
}

#[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
impl<A: Step> Iterator for RangeInclusiveIter<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.0.count()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<A> {
        self.0.nth(n)
    }

    #[inline]
    fn last(self) -> Option<A> {
        self.0.last()
    }

    #[inline]
    fn min(self) -> Option<A>
    where
        A: Ord,
    {
        self.0.min()
    }

    #[inline]
    fn max(self) -> Option<A>
    where
        A: Ord,
    {
        self.0.max()
    }

    #[inline]
    fn is_sorted(self) -> bool {
        true
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.0.advance_by(n)
    }
}

#[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
impl<A: Step> DoubleEndedIterator for RangeInclusiveIter<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        self.0.next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<A> {
        self.0.nth_back(n)
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.0.advance_back_by(n)
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: TrustedStep> TrustedLen for RangeInclusiveIter<A> {}

#[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
impl<A: Step> FusedIterator for RangeInclusiveIter<A> {}

#[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
impl<A: Step> IntoIterator for RangeInclusive<A> {
    type Item = A;
    type IntoIter = RangeInclusiveIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        RangeInclusiveIter(self.into())
    }
}

// These macros generate `ExactSizeIterator` impls for various range types.
//
// * `ExactSizeIterator::len` is required to always return an exact `usize`,
//   so no range can be longer than `usize::MAX`.
// * For integer types in `Range<_>` this is the case for types narrower than or as wide as `usize`.
//   For integer types in `RangeInclusive<_>`
//   this is the case for types *strictly narrower* than `usize`
//   since e.g. `(0..=u64::MAX).len()` would be `u64::MAX + 1`.
macro_rules! range_exact_iter_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "new_range_api", since = "1.96.0")]
        impl ExactSizeIterator for RangeIter<$t> { }
    )*)
}

macro_rules! range_incl_exact_iter_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
        impl ExactSizeIterator for RangeInclusiveIter<$t> { }
    )*)
}

range_exact_iter_impl! {
    usize u8 u16
    isize i8 i16
}

range_incl_exact_iter_impl! {
    u8
    i8
}

/// By-value [`RangeFrom`] iterator.
#[stable(feature = "new_range_from_api", since = "1.96.0")]
#[derive(Debug, Clone)]
pub struct RangeFromIter<A> {
    /// None when the type have been exhausted
    start: Option<A>,
}

impl<A: Step> RangeFromIter<A> {
    /// Returns the remainder of the range being iterated over.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(new_range_remainder)]
    ///
    /// let range = core::range::RangeFrom::from(3..);
    /// let mut iter = range.into_iter();
    /// assert_eq!(iter.clone().remainder(), Some(range));
    /// iter.next();
    /// assert_eq!(iter.remainder(), Some(core::range::RangeFrom::from(4..)));
    /// ```
    #[inline]
    #[rustc_inherit_overflow_checks]
    #[unstable(feature = "new_range_remainder", issue = "154458")]
    pub fn remainder(self) -> Option<RangeFrom<A>> {
        let start = self.start?;
        Some(RangeFrom { start })
    }
}

#[stable(feature = "new_range_from_api", since = "1.96.0")]
impl<A: Step> Iterator for RangeFromIter<A> {
    type Item = A;

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn next(&mut self) -> Option<A> {
        let current = self.start.clone()?;
        let next = Step::forward_checked(current, 1);
        mem::replace(&mut self.start, next)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn nth(&mut self, n: usize) -> Option<A> {
        let current = self.start.clone()?;
        let plus_n = Step::forward_checked(current, n);
        let next = plus_n.clone().and_then(|x| Step::forward_checked(x, 1));
        self.start = next;
        plus_n
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: TrustedStep> TrustedLen for RangeFromIter<A> {}

#[stable(feature = "new_range_from_api", since = "1.96.0")]
impl<A: Step> FusedIterator for RangeFromIter<A> {}

#[stable(feature = "new_range_from_api", since = "1.96.0")]
impl<A: Step> IntoIterator for RangeFrom<A> {
    type Item = A;
    type IntoIter = RangeFromIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        RangeFromIter { start: Some(self.start) }
    }
}
