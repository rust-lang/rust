use crate::iter::{
    FusedIterator, Step, TrustedLen, TrustedRandomAccess, TrustedRandomAccessNoCoerce, TrustedStep,
};
use crate::num::NonZero;
use crate::range::{Range, RangeFrom, RangeInclusive, legacy};
use crate::{intrinsics, mem};

/// By-value [`Range`] iterator.
#[unstable(feature = "new_range_api", issue = "125687")]
#[derive(Debug, Clone)]
pub struct RangeIter<A>(legacy::Range<A>);

impl<A> RangeIter<A> {
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

#[unstable(feature = "new_range_api", issue = "125687")]
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

#[unstable(feature = "new_range_api", issue = "125687")]
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

#[unstable(feature = "new_range_api", issue = "125687")]
impl<A: Step> FusedIterator for RangeIter<A> {}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<A: Step> IntoIterator for Range<A> {
    type Item = A;
    type IntoIter = RangeIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        RangeIter(self.into())
    }
}

/// By-value [`RangeInclusive`] iterator.
#[unstable(feature = "new_range_api", issue = "125687")]
#[derive(Debug, Clone)]
pub struct RangeInclusiveIter<A>(legacy::RangeInclusive<A>);

impl<A: Step> RangeInclusiveIter<A> {
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

#[unstable(feature = "new_range_api", issue = "125687")]
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

#[unstable(feature = "new_range_api", issue = "125687")]
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

#[unstable(feature = "new_range_api", issue = "125687")]
impl<A: Step> FusedIterator for RangeInclusiveIter<A> {}

#[unstable(feature = "new_range_api", issue = "125687")]
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
        #[unstable(feature = "new_range_api", issue = "125687")]
        impl ExactSizeIterator for RangeIter<$t> { }
    )*)
}

macro_rules! range_incl_exact_iter_impl {
    ($($t:ty)*) => ($(
        #[unstable(feature = "new_range_api", issue = "125687")]
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
#[unstable(feature = "new_range_api", issue = "125687")]
#[derive(Debug, Clone)]
pub struct RangeFromIter<A> {
    start: A,
    /// Whether the first element of the iterator has yielded.
    /// Only used when overflow checks are enabled.
    first: bool,
}

impl<A: Step> RangeFromIter<A> {
    /// Returns the remainder of the range being iterated over.
    #[inline]
    #[rustc_inherit_overflow_checks]
    pub fn remainder(self) -> RangeFrom<A> {
        if intrinsics::overflow_checks() {
            if !self.first {
                return RangeFrom { start: Step::forward(self.start, 1) };
            }
        }

        RangeFrom { start: self.start }
    }
}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<A: Step> Iterator for RangeFromIter<A> {
    type Item = A;

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn next(&mut self) -> Option<A> {
        if intrinsics::overflow_checks() {
            if self.first {
                self.first = false;
                return Some(self.start.clone());
            }

            self.start = Step::forward(self.start.clone(), 1);
            return Some(self.start.clone());
        }

        let n = Step::forward(self.start.clone(), 1);
        Some(mem::replace(&mut self.start, n))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn nth(&mut self, n: usize) -> Option<A> {
        if intrinsics::overflow_checks() {
            if self.first {
                self.first = false;

                let plus_n = Step::forward(self.start.clone(), n);
                self.start = plus_n.clone();
                return Some(plus_n);
            }

            let plus_n = Step::forward(self.start.clone(), n);
            self.start = Step::forward(plus_n.clone(), 1);
            return Some(self.start.clone());
        }

        let plus_n = Step::forward(self.start.clone(), n);
        self.start = Step::forward(plus_n.clone(), 1);
        Some(plus_n)
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: TrustedStep> TrustedLen for RangeFromIter<A> {}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<A: Step> FusedIterator for RangeFromIter<A> {}

#[unstable(feature = "new_range_api", issue = "125687")]
impl<A: Step> IntoIterator for RangeFrom<A> {
    type Item = A;
    type IntoIter = RangeFromIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        RangeFromIter { start: self.start, first: true }
    }
}
