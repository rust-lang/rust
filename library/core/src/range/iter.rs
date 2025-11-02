use crate::iter::{
    FusedIterator, Step, TrustedLen, TrustedRandomAccess, TrustedRandomAccessNoCoerce, TrustedStep,
};
use crate::num::NonZero;
use crate::range::{Range, RangeFrom, RangeInclusive, legacy};

/// By-value [`Range`] iterator.
#[unstable(feature = "new_range_api", issue = "125687")]
#[derive(Debug, Clone)]
pub struct IterRange<A>(legacy::Range<A>);

impl<A> IterRange<A> {
    /// Returns the remainder of the range being iterated over.
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
        unsafe impl TrustedRandomAccess for IterRange<$t> {}

        #[doc(hidden)]
        #[unstable(feature = "trusted_random_access", issue = "none")]
        unsafe impl TrustedRandomAccessNoCoerce for IterRange<$t> {
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

#[unstable(feature = "new_range_api", issue = "125687")]
impl<A: Step> Iterator for IterRange<A> {
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

#[unstable(feature = "new_range_api", issue = "125687")]
impl<A: Step> DoubleEndedIterator for IterRange<A> {
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
unsafe impl<A: TrustedStep> TrustedLen for IterRange<A> {}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<A: Step> FusedIterator for IterRange<A> {}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<A: Step> IntoIterator for Range<A> {
    type Item = A;
    type IntoIter = IterRange<A>;

    fn into_iter(self) -> Self::IntoIter {
        IterRange(self.into())
    }
}

/// By-value [`RangeInclusive`] iterator.
#[unstable(feature = "new_range_api", issue = "125687")]
#[derive(Debug, Clone)]
pub struct IterRangeInclusive<A>(legacy::RangeInclusive<A>);

impl<A: Step> IterRangeInclusive<A> {
    /// Returns the remainder of the range being iterated over.
    ///
    /// If the iterator is exhausted or empty, returns `None`.
    pub fn remainder(self) -> Option<RangeInclusive<A>> {
        if self.0.is_empty() {
            return None;
        }

        Some(RangeInclusive { start: self.0.start, last: self.0.end })
    }
}

#[unstable(feature = "trusted_random_access", issue = "none")]
impl<A: Step> Iterator for IterRangeInclusive<A> {
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

#[unstable(feature = "new_range_api", issue = "125687")]
impl<A: Step> DoubleEndedIterator for IterRangeInclusive<A> {
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
unsafe impl<A: TrustedStep> TrustedLen for IterRangeInclusive<A> {}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<A: Step> FusedIterator for IterRangeInclusive<A> {}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<A: Step> IntoIterator for RangeInclusive<A> {
    type Item = A;
    type IntoIter = IterRangeInclusive<A>;

    fn into_iter(self) -> Self::IntoIter {
        IterRangeInclusive(self.into())
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
        #[unstable(feature = "new_range_api", issue = "125687")]
        impl ExactSizeIterator for IterRange<$t> { }
    )*)
}

macro_rules! range_incl_exact_iter_impl {
    ($($t:ty)*) => ($(
        #[unstable(feature = "new_range_api", issue = "125687")]
        impl ExactSizeIterator for IterRangeInclusive<$t> { }
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
#[unstable(feature = "new_range_api", issue = "125687")]
#[derive(Debug, Clone)]
pub struct IterRangeFrom<A>(legacy::RangeFrom<A>);

impl<A> IterRangeFrom<A> {
    /// Returns the remainder of the range being iterated over.
    pub fn remainder(self) -> RangeFrom<A> {
        RangeFrom { start: self.0.start }
    }
}

#[unstable(feature = "trusted_random_access", issue = "none")]
impl<A: Step> Iterator for IterRangeFrom<A> {
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
    fn nth(&mut self, n: usize) -> Option<A> {
        self.0.nth(n)
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: TrustedStep> TrustedLen for IterRangeFrom<A> {}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<A: Step> FusedIterator for IterRangeFrom<A> {}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<A: Step> IntoIterator for RangeFrom<A> {
    type Item = A;
    type IntoIter = IterRangeFrom<A>;

    fn into_iter(self) -> Self::IntoIter {
        IterRangeFrom(self.into())
    }
}
