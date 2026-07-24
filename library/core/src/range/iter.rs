use crate::hint::cold_path;
use crate::iter::{
    FusedIterator, Step, TrustedLen, TrustedRandomAccess, TrustedRandomAccessNoCoerce, TrustedStep,
};
use crate::marker::Destruct;
use crate::num::NonZero;
use crate::ops::Try;
use crate::range::{Range, RangeFrom, RangeInclusive, legacy};
use crate::{fmt, intrinsics, mem};

/// By-value [`Range`] iterator.
#[stable(feature = "new_range_api", since = "1.96.0")]
#[derive(Debug, Clone)]
pub struct RangeIter<A>(legacy::Range<A>);

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
#[derive(Clone)]
pub struct RangeInclusiveIter<A> {
    // When created from `start..=last`, this range is
    // - Preferably `start..(last+1)`, so we only need to delegate to the exclusive range
    // - If necessary (because `last` is a maximal element) `start..last`,
    //   with the `is_inclusive` field set to `true`
    range: legacy::Range<A>,
    // Preferably this is `false`, denoting that we successfully converted the inclusive
    // range into an exclusive range, and thus have no need for extra handling.
    // If this is true, however, that means that we must return one final item
    // after iterating it as an exclusive range.
    // This must only be true if the iterator is non-empty, implying
    // `range.start <= range.end`. (If the original inclusive range is empty because
    // `!(start <= last)`, it's stored as the empty exclusive range `start..last` )
    is_inclusive: bool,
}

#[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
impl<A: fmt::Debug> fmt::Debug for RangeInclusiveIter<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { range: legacy::Range { start, end }, is_inclusive } = self;
        let inclusive;
        let exclusive;
        let field: &dyn fmt::Debug = if *is_inclusive {
            inclusive = &start..=&end;
            &inclusive
        } else {
            exclusive = &start..&end;
            &exclusive
        };
        fmt::Formatter::debug_tuple_field1_finish(f, "RangeInclusiveIter", field)
    }
}

impl<A: Step> RangeInclusiveIter<A> {
    #[rustc_const_unstable(feature = "const_range", issue = "none")]
    #[inline]
    const fn is_empty(&self) -> bool
    where
        A: [const] PartialOrd,
    {
        let Self { range, is_inclusive } = self;
        // We expect to need the range comparison (since inclusive is rare),
        // so run it outside the `if` to tell the backend that it's ok to look
        // at those fields unconditionally.
        let range_is_empty = range.is_empty();
        if *is_inclusive {
            debug_assert!(range.start <= range.end);
            false
        } else {
            range_is_empty
        }
    }

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
        if self.is_empty() {
            return None;
        }

        let Self { range: legacy::Range { start, end }, is_inclusive } = self;
        let last = if is_inclusive {
            end
        } else {
            // Can't overflow because the range isn't empty
            Step::backward(end, 1)
        };
        Some(RangeInclusive { start, last })
    }
}

#[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
impl<A: Step> Iterator for RangeInclusiveIter<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        // Conveniently, regardless of whether we were able to convert to exclusive,
        // the normal case is to return a value when `range.start < range.end`.
        // Only rarely do we need to handle something else.

        let Self { range, is_inclusive } = self;
        if let next @ Some(_) = range.next() {
            next
        } else {
            cold_path();
            if *is_inclusive {
                // Tighter invariant check than normal because we only get here after
                // having already returned all the previous items.
                // As an exclusive range it's empty, but we give out one more.
                debug_assert!(range.start == range.end);
                *is_inclusive = false;
                // Because we're going forward, prefer giving out `start` since
                // that's what the `range.next()` above returned.
                let last = range.start.clone();
                Some(last)
            } else {
                None
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let Self { range, is_inclusive } = self;
        let (low, high) = range.size_hint();
        let extra = *is_inclusive as usize;
        (low.saturating_add(extra), try { high?.checked_add(extra)? })
    }

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn count(self) -> usize {
        let Self { range, is_inclusive } = self;
        let extra = is_inclusive as usize;
        range.count() + extra
    }

    impl_fold_via_try_fold! { fold -> try_fold }

    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R + Destruct,
        R: Try<Output = B>,
    {
        let Self { range, is_inclusive } = self;
        let mut accum = init;

        accum = range.try_fold(accum, &mut f)?;

        if *is_inclusive {
            cold_path();
            debug_assert!(range.start == range.end);
            *is_inclusive = false;
            let last = range.start.clone();
            // Update the state before this call so it happens even if `?` short-circuits
            accum = f(accum, last)?;
        }

        try { accum }
    }

    #[inline]
    fn last(self) -> Option<A> {
        { self }.next_back()
    }

    #[inline]
    fn min(self) -> Option<A>
    where
        A: Ord,
    {
        { self }.next()
    }

    #[inline]
    fn max(self) -> Option<A>
    where
        A: Ord,
    {
        { self }.next_back()
    }

    #[inline]
    fn is_sorted(self) -> bool {
        true
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        let Self { range, is_inclusive } = self;
        match range.advance_by(n) {
            Ok(()) => Ok(()),
            Err(remainder) => {
                if *is_inclusive {
                    cold_path();
                    debug_assert!(range.start == range.end);
                    // `remainder` is `NonZero`, so we always pass the final element
                    *is_inclusive = false;
                    if let Some(remainder) = NonZero::new(remainder.get() - 1) {
                        Err(remainder)
                    } else {
                        Ok(())
                    }
                } else {
                    Err(remainder)
                }
            }
        }
    }
}

#[stable(feature = "new_range_inclusive_api", since = "1.95.0")]
impl<A: Step> DoubleEndedIterator for RangeInclusiveIter<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        // Sadly when iterating backwards we have to always check whether we're
        // inclusive, even though it's rare.

        let Self { range, is_inclusive } = self;
        if *is_inclusive {
            cold_path();
            debug_assert!(range.start <= range.end);
            *is_inclusive = false;
            let last = range.end.clone();
            Some(last)
        } else {
            range.next_back()
        }
    }

    impl_fold_via_try_fold! { rfold -> try_rfold }

    fn try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R + Destruct,
        R: Try<Output = B>,
    {
        let Self { range, is_inclusive } = self;
        let mut accum = init;

        if *is_inclusive {
            cold_path();
            debug_assert!(range.start <= range.end);
            *is_inclusive = false;
            let last = range.end.clone();
            // Update the state before this call so it happens even if `?` short-circuits
            accum = f(accum, last)?;
        }

        accum = range.try_rfold(accum, f)?;

        try { accum }
    }

    #[inline]
    fn advance_back_by(&mut self, mut n: usize) -> Result<(), NonZero<usize>> {
        let Self { range, is_inclusive } = self;
        if *is_inclusive {
            cold_path();
            debug_assert!(range.start <= range.end);
            *is_inclusive = false;
            if let Some(new_n) = n.checked_sub(1) {
                n = new_n;
            } else {
                return Ok(());
            }
        }

        range.advance_back_by(n)
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

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // This is the core opportunity for us to do something different from the
        // legacy `RangeInclusive` type. For the old one `into_iter` is forced to
        // be identity, but here we can try to adjust it *outside* the loop.

        let Self { start, last } = self;
        let is_inclusive;
        let end = if let Some(end) = Step::forward_checked(last.clone(), 1) {
            is_inclusive = false;
            end
        } else {
            is_inclusive = start <= last;
            if !is_inclusive {
                // This is unreachable for `Ord` types, but `Step` accepts partial orders.
                // So it's possible for the range to be empty even if `last` is
                // a maximal element in the DAG.
                debug_assert_eq!(PartialOrd::partial_cmp(&start, &last), None);
            }
            last
        };
        let range = legacy::Range { start, end };
        RangeInclusiveIter { range, is_inclusive }
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
        impl ExactSizeIterator for RangeInclusiveIter<$t> {
            fn is_empty(&self) -> bool {
                self.is_empty()
            }
        }
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
    start: A,
    /// Whether the maximum value of the iterator has yielded.
    /// Only used when overflow checks are enabled.
    exhausted: bool,
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
    /// assert_eq!(iter.clone().remainder(), range);
    /// iter.next();
    /// assert_eq!(iter.remainder(), core::range::RangeFrom::from(4..));
    /// ```
    #[inline]
    #[rustc_inherit_overflow_checks]
    #[unstable(feature = "new_range_remainder", issue = "154458")]
    pub fn remainder(self) -> RangeFrom<A> {
        // Need to handle this case even if overflow-checks are disabled,
        // because a `RangeFromIter` could be exhausted in a crate with
        // overflow-checks enabled, but then passed to a crate with them
        // disabled before this is called.
        if self.exhausted {
            return RangeFrom { start: Step::forward(self.start, 1) };
        }

        RangeFrom { start: self.start }
    }
}

#[stable(feature = "new_range_from_api", since = "1.96.0")]
impl<A: Step> Iterator for RangeFromIter<A> {
    type Item = A;

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn next(&mut self) -> Option<A> {
        if self.exhausted {
            // This should panic if overflow checks are enabled, since
            // `forward_checked` returned `None` in prior iteration.
            self.start = Step::forward(self.start.clone(), 1);

            // If we get here, if means this iterator was exhausted by a crate
            // with overflow-checks enabled, but now we're iterating in a crate with
            // overflow-checks disabled. Since we successfully incremented `self.start`
            // above (in many cases this will wrap around to MIN), we now unset
            // the flag so we don't repeat this process in the next iteration.
            //
            // This could also happen if `forward_checked` returned None but
            // (for whatever reason, not applicable to any std implementors)
            // `forward` doesn't panic when overflow-checks are enabled. In that
            // case, this is also the correct behavior.
            self.exhausted = false;
        }
        if intrinsics::overflow_checks() {
            let Some(n) = Step::forward_checked(self.start.clone(), 1) else {
                self.exhausted = true;
                return Some(self.start.clone());
            };
            return Some(mem::replace(&mut self.start, n));
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
        // Typically `forward` will cause an overflow-check panic here,
        // but unset the exhausted flag to handle the uncommon cases.
        // See the comments in `next` for more details.
        if self.exhausted {
            self.start = Step::forward(self.start.clone(), 1);
            self.exhausted = false;
        }
        if intrinsics::overflow_checks() {
            let plus_n = Step::forward(self.start.clone(), n);
            if let Some(plus_n1) = Step::forward_checked(plus_n.clone(), 1) {
                self.start = plus_n1;
            } else {
                self.start = plus_n.clone();
                self.exhausted = true;
            }
            return Some(plus_n);
        }

        let plus_n = Step::forward(self.start.clone(), n);
        self.start = Step::forward(plus_n.clone(), 1);
        Some(plus_n)
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
        RangeFromIter { start: self.start, exhausted: false }
    }
}
