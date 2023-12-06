use crate::mem;
use crate::num::NonZeroUsize;
use crate::ops::{self, Try};

use super::{
    FusedIterator, Step, TrustedLen, TrustedRandomAccess, TrustedRandomAccessNoCoerce, TrustedStep,
};

macro_rules! range_exact_iter_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl ExactSizeIterator for ops::range::legacy::Range<$t> { }
    )*)
}

/// Safety: This macro must only be used on types that are `Copy` and result in ranges
/// which have an exact `size_hint()` where the upper bound must not be `None`.
macro_rules! unsafe_range_trusted_random_access_impl {
    ($($t:ty)*) => ($(
        #[doc(hidden)]
        #[unstable(feature = "trusted_random_access", issue = "none")]
        unsafe impl TrustedRandomAccess for ops::range::legacy::Range<$t> {}

        #[doc(hidden)]
        #[unstable(feature = "trusted_random_access", issue = "none")]
        unsafe impl TrustedRandomAccessNoCoerce for ops::range::legacy::Range<$t> {
            const MAY_HAVE_SIDE_EFFECT: bool = false;
        }
    )*)
}

macro_rules! range_incl_exact_iter_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "inclusive_range", since = "1.26.0")]
        impl ExactSizeIterator for ops::range::legacy::RangeInclusive<$t> { }
    )*)
}

/// Specialization implementations for `Range`.
trait RangeIteratorImpl {
    type Item;

    // Iterator
    fn spec_next(&mut self) -> Option<Self::Item>;
    fn spec_nth(&mut self, n: usize) -> Option<Self::Item>;
    fn spec_advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize>;

    // DoubleEndedIterator
    fn spec_next_back(&mut self) -> Option<Self::Item>;
    fn spec_nth_back(&mut self, n: usize) -> Option<Self::Item>;
    fn spec_advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize>;
}

impl<A: Step> RangeIteratorImpl for ops::range::legacy::Range<A> {
    type Item = A;

    #[inline]
    default fn spec_next(&mut self) -> Option<A> {
        if self.start < self.end {
            let n =
                Step::forward_checked(self.start.clone(), 1).expect("`Step` invariants not upheld");
            Some(mem::replace(&mut self.start, n))
        } else {
            None
        }
    }

    #[inline]
    default fn spec_nth(&mut self, n: usize) -> Option<A> {
        if let Some(plus_n) = Step::forward_checked(self.start.clone(), n) {
            if plus_n < self.end {
                self.start =
                    Step::forward_checked(plus_n.clone(), 1).expect("`Step` invariants not upheld");
                return Some(plus_n);
            }
        }

        self.start = self.end.clone();
        None
    }

    #[inline]
    default fn spec_advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        let available = if self.start <= self.end {
            Step::steps_between(&self.start, &self.end).unwrap_or(usize::MAX)
        } else {
            0
        };

        let taken = available.min(n);

        self.start =
            Step::forward_checked(self.start.clone(), taken).expect("`Step` invariants not upheld");

        NonZeroUsize::new(n - taken).map_or(Ok(()), Err)
    }

    #[inline]
    default fn spec_next_back(&mut self) -> Option<A> {
        if self.start < self.end {
            self.end =
                Step::backward_checked(self.end.clone(), 1).expect("`Step` invariants not upheld");
            Some(self.end.clone())
        } else {
            None
        }
    }

    #[inline]
    default fn spec_nth_back(&mut self, n: usize) -> Option<A> {
        if let Some(minus_n) = Step::backward_checked(self.end.clone(), n) {
            if minus_n > self.start {
                self.end =
                    Step::backward_checked(minus_n, 1).expect("`Step` invariants not upheld");
                return Some(self.end.clone());
            }
        }

        self.end = self.start.clone();
        None
    }

    #[inline]
    default fn spec_advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        let available = if self.start <= self.end {
            Step::steps_between(&self.start, &self.end).unwrap_or(usize::MAX)
        } else {
            0
        };

        let taken = available.min(n);

        self.end =
            Step::backward_checked(self.end.clone(), taken).expect("`Step` invariants not upheld");

        NonZeroUsize::new(n - taken).map_or(Ok(()), Err)
    }
}

impl<T: TrustedStep> RangeIteratorImpl for ops::range::legacy::Range<T> {
    #[inline]
    fn spec_next(&mut self) -> Option<T> {
        if self.start < self.end {
            let old = self.start;
            // SAFETY: just checked precondition
            self.start = unsafe { Step::forward_unchecked(old, 1) };
            Some(old)
        } else {
            None
        }
    }

    #[inline]
    fn spec_nth(&mut self, n: usize) -> Option<T> {
        if let Some(plus_n) = Step::forward_checked(self.start, n) {
            if plus_n < self.end {
                // SAFETY: just checked precondition
                self.start = unsafe { Step::forward_unchecked(plus_n, 1) };
                return Some(plus_n);
            }
        }

        self.start = self.end;
        None
    }

    #[inline]
    fn spec_advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        let available = if self.start <= self.end {
            Step::steps_between(&self.start, &self.end).unwrap_or(usize::MAX)
        } else {
            0
        };

        let taken = available.min(n);

        // SAFETY: the conditions above ensure that the count is in bounds. If start <= end
        // then steps_between either returns a bound to which we clamp or returns None which
        // together with the initial inequality implies more than usize::MAX steps.
        // Otherwise 0 is returned which always safe to use.
        self.start = unsafe { Step::forward_unchecked(self.start, taken) };

        NonZeroUsize::new(n - taken).map_or(Ok(()), Err)
    }

    #[inline]
    fn spec_next_back(&mut self) -> Option<T> {
        if self.start < self.end {
            // SAFETY: just checked precondition
            self.end = unsafe { Step::backward_unchecked(self.end, 1) };
            Some(self.end)
        } else {
            None
        }
    }

    #[inline]
    fn spec_nth_back(&mut self, n: usize) -> Option<T> {
        if let Some(minus_n) = Step::backward_checked(self.end, n) {
            if minus_n > self.start {
                // SAFETY: just checked precondition
                self.end = unsafe { Step::backward_unchecked(minus_n, 1) };
                return Some(self.end);
            }
        }

        self.end = self.start;
        None
    }

    #[inline]
    fn spec_advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        let available = if self.start <= self.end {
            Step::steps_between(&self.start, &self.end).unwrap_or(usize::MAX)
        } else {
            0
        };

        let taken = available.min(n);

        // SAFETY: same as the spec_advance_by() implementation
        self.end = unsafe { Step::backward_unchecked(self.end, taken) };

        NonZeroUsize::new(n - taken).map_or(Ok(()), Err)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> Iterator for ops::range::legacy::Range<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        self.spec_next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.start < self.end {
            let hint = Step::steps_between(&self.start, &self.end);
            (hint.unwrap_or(usize::MAX), hint)
        } else {
            (0, Some(0))
        }
    }

    #[inline]
    fn count(self) -> usize {
        if self.start < self.end {
            Step::steps_between(&self.start, &self.end).expect("count overflowed usize")
        } else {
            0
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<A> {
        self.spec_nth(n)
    }

    #[inline]
    fn last(mut self) -> Option<A> {
        self.next_back()
    }

    #[inline]
    fn min(mut self) -> Option<A>
    where
        A: Ord,
    {
        self.next()
    }

    #[inline]
    fn max(mut self) -> Option<A>
    where
        A: Ord,
    {
        self.next_back()
    }

    #[inline]
    fn is_sorted(self) -> bool {
        true
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        self.spec_advance_by(n)
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
        unsafe { Step::forward_unchecked(self.start.clone(), idx) }
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
range_exact_iter_impl! {
    usize u8 u16
    isize i8 i16

    // These are incorrect per the reasoning above,
    // but removing them would be a breaking change as they were stabilized in Rust 1.0.0.
    // So e.g. `(0..66_000_u32).len()` for example will compile without error or warnings
    // on 16-bit platforms, but continue to give a wrong result.
    u32
    i32
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

range_incl_exact_iter_impl! {
    u8
    i8

    // These are incorrect per the reasoning above,
    // but removing them would be a breaking change as they were stabilized in Rust 1.26.0.
    // So e.g. `(0..=u16::MAX).len()` for example will compile without error or warnings
    // on 16-bit platforms, but continue to give a wrong result.
    u16
    i16
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> DoubleEndedIterator for ops::range::legacy::Range<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        self.spec_next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<A> {
        self.spec_nth_back(n)
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        self.spec_advance_back_by(n)
    }
}

// Safety:
// The following invariants for `Step::steps_between` exist:
//
// > * `steps_between(&a, &b) == Some(n)` only if `a <= b`
// >   * Note that `a <= b` does _not_ imply `steps_between(&a, &b) != None`;
// >     this is the case when it would require more than `usize::MAX` steps to
// >     get to `b`
// > * `steps_between(&a, &b) == None` if `a > b`
//
// The first invariant is what is generally required for `TrustedLen` to be
// sound. The note addendum satisfies an additional `TrustedLen` invariant.
//
// > The upper bound must only be `None` if the actual iterator length is larger
// > than `usize::MAX`
//
// The second invariant logically follows the first so long as the `PartialOrd`
// implementation is correct; regardless it is explicitly stated. If `a < b`
// then `(0, Some(0))` is returned by `ops::range::legacy::Range<A: Step>::size_hint`. As such
// the second invariant is upheld.
#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: TrustedStep> TrustedLen for ops::range::legacy::Range<A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<A: Step> FusedIterator for ops::range::legacy::Range<A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> Iterator for ops::range::legacy::RangeFrom<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        let n = Step::forward(self.start.clone(), 1);
        Some(mem::replace(&mut self.start, n))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<A> {
        let plus_n = Step::forward(self.start.clone(), n);
        self.start = Step::forward(plus_n.clone(), 1);
        Some(plus_n)
    }
}

// Safety: See above implementation for `ops::range::legacy::Range<A>`
#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: TrustedStep> TrustedLen for ops::range::legacy::RangeFrom<A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<A: Step> FusedIterator for ops::range::legacy::RangeFrom<A> {}

trait RangeInclusiveIteratorImpl {
    type Item;

    // Iterator
    fn spec_next(&mut self) -> Option<Self::Item>;
    fn spec_try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>;

    // DoubleEndedIterator
    fn spec_next_back(&mut self) -> Option<Self::Item>;
    fn spec_try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>;
}

impl<A: Step> RangeInclusiveIteratorImpl for ops::range::legacy::RangeInclusive<A> {
    type Item = A;

    #[inline]
    default fn spec_next(&mut self) -> Option<A> {
        if self.is_empty() {
            return None;
        }
        let is_iterating = self.start < self.end;
        Some(if is_iterating {
            let n =
                Step::forward_checked(self.start.clone(), 1).expect("`Step` invariants not upheld");
            mem::replace(&mut self.start, n)
        } else {
            self.exhausted = true;
            self.start.clone()
        })
    }

    #[inline]
    default fn spec_try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, A) -> R,
        R: Try<Output = B>,
    {
        if self.is_empty() {
            return try { init };
        }

        let mut accum = init;

        while self.start < self.end {
            let n =
                Step::forward_checked(self.start.clone(), 1).expect("`Step` invariants not upheld");
            let n = mem::replace(&mut self.start, n);
            accum = f(accum, n)?;
        }

        self.exhausted = true;

        if self.start == self.end {
            accum = f(accum, self.start.clone())?;
        }

        try { accum }
    }

    #[inline]
    default fn spec_next_back(&mut self) -> Option<A> {
        if self.is_empty() {
            return None;
        }
        let is_iterating = self.start < self.end;
        Some(if is_iterating {
            let n =
                Step::backward_checked(self.end.clone(), 1).expect("`Step` invariants not upheld");
            mem::replace(&mut self.end, n)
        } else {
            self.exhausted = true;
            self.end.clone()
        })
    }

    #[inline]
    default fn spec_try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, A) -> R,
        R: Try<Output = B>,
    {
        if self.is_empty() {
            return try { init };
        }

        let mut accum = init;

        while self.start < self.end {
            let n =
                Step::backward_checked(self.end.clone(), 1).expect("`Step` invariants not upheld");
            let n = mem::replace(&mut self.end, n);
            accum = f(accum, n)?;
        }

        self.exhausted = true;

        if self.start == self.end {
            accum = f(accum, self.start.clone())?;
        }

        try { accum }
    }
}

impl<T: TrustedStep> RangeInclusiveIteratorImpl for ops::range::legacy::RangeInclusive<T> {
    #[inline]
    fn spec_next(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let is_iterating = self.start < self.end;
        Some(if is_iterating {
            // SAFETY: just checked precondition
            let n = unsafe { Step::forward_unchecked(self.start.clone(), 1) };
            mem::replace(&mut self.start, n)
        } else {
            self.exhausted = true;
            self.start.clone()
        })
    }

    #[inline]
    fn spec_try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, T) -> R,
        R: Try<Output = B>,
    {
        if self.is_empty() {
            return try { init };
        }

        let mut accum = init;

        while self.start < self.end {
            // SAFETY: just checked precondition
            let n = unsafe { Step::forward_unchecked(self.start.clone(), 1) };
            let n = mem::replace(&mut self.start, n);
            accum = f(accum, n)?;
        }

        self.exhausted = true;

        if self.start == self.end {
            accum = f(accum, self.start.clone())?;
        }

        try { accum }
    }

    #[inline]
    fn spec_next_back(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let is_iterating = self.start < self.end;
        Some(if is_iterating {
            // SAFETY: just checked precondition
            let n = unsafe { Step::backward_unchecked(self.end.clone(), 1) };
            mem::replace(&mut self.end, n)
        } else {
            self.exhausted = true;
            self.end.clone()
        })
    }

    #[inline]
    fn spec_try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, T) -> R,
        R: Try<Output = B>,
    {
        if self.is_empty() {
            return try { init };
        }

        let mut accum = init;

        while self.start < self.end {
            // SAFETY: just checked precondition
            let n = unsafe { Step::backward_unchecked(self.end.clone(), 1) };
            let n = mem::replace(&mut self.end, n);
            accum = f(accum, n)?;
        }

        self.exhausted = true;

        if self.start == self.end {
            accum = f(accum, self.start.clone())?;
        }

        try { accum }
    }
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<A: Step> Iterator for ops::range::legacy::RangeInclusive<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        self.spec_next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.is_empty() {
            return (0, Some(0));
        }

        match Step::steps_between(&self.start, &self.end) {
            Some(hint) => (hint.saturating_add(1), hint.checked_add(1)),
            None => (usize::MAX, None),
        }
    }

    #[inline]
    fn count(self) -> usize {
        if self.is_empty() {
            return 0;
        }

        Step::steps_between(&self.start, &self.end)
            .and_then(|steps| steps.checked_add(1))
            .expect("count overflowed usize")
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<A> {
        if self.is_empty() {
            return None;
        }

        if let Some(plus_n) = Step::forward_checked(self.start.clone(), n) {
            use crate::cmp::Ordering::*;

            match plus_n.partial_cmp(&self.end) {
                Some(Less) => {
                    self.start = Step::forward(plus_n.clone(), 1);
                    return Some(plus_n);
                }
                Some(Equal) => {
                    self.start = plus_n.clone();
                    self.exhausted = true;
                    return Some(plus_n);
                }
                _ => {}
            }
        }

        self.start = self.end.clone();
        self.exhausted = true;
        None
    }

    #[inline]
    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        self.spec_try_fold(init, f)
    }

    impl_fold_via_try_fold! { fold -> try_fold }

    #[inline]
    fn last(mut self) -> Option<A> {
        self.next_back()
    }

    #[inline]
    fn min(mut self) -> Option<A>
    where
        A: Ord,
    {
        self.next()
    }

    #[inline]
    fn max(mut self) -> Option<A>
    where
        A: Ord,
    {
        self.next_back()
    }

    #[inline]
    fn is_sorted(self) -> bool {
        true
    }
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<A: Step> DoubleEndedIterator for ops::range::legacy::RangeInclusive<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        self.spec_next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<A> {
        if self.is_empty() {
            return None;
        }

        if let Some(minus_n) = Step::backward_checked(self.end.clone(), n) {
            use crate::cmp::Ordering::*;

            match minus_n.partial_cmp(&self.start) {
                Some(Greater) => {
                    self.end = Step::backward(minus_n.clone(), 1);
                    return Some(minus_n);
                }
                Some(Equal) => {
                    self.end = minus_n.clone();
                    self.exhausted = true;
                    return Some(minus_n);
                }
                _ => {}
            }
        }

        self.end = self.start.clone();
        self.exhausted = true;
        None
    }

    #[inline]
    fn try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        self.spec_try_rfold(init, f)
    }

    impl_fold_via_try_fold! { rfold -> try_rfold }
}

// Safety: See above implementation for `ops::range::legacy::Range<A>`
#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: TrustedStep> TrustedLen for ops::range::legacy::RangeInclusive<A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<A: Step> FusedIterator for ops::range::legacy::RangeInclusive<A> {}
