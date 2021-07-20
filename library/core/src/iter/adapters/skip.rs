use crate::intrinsics::unlikely;
use crate::iter::adapters::zip::try_get_unchecked;
use crate::iter::{
    adapters::SourceIter, FusedIterator, InPlaceIterable, TrustedLen, TrustedRandomAccess,
};
use crate::ops::{ControlFlow, Try};

/// An iterator that skips over `n` elements of `iter`.
///
/// This `struct` is created by the [`skip`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`skip`]: Iterator::skip
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Skip<I> {
    iter: I,
    n: usize,
}

impl<I> Skip<I> {
    pub(in crate::iter) fn new(iter: I, n: usize) -> Skip<I> {
        Skip { iter, n }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Skip<I>
where
    I: Iterator,
{
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        if unlikely(self.n > 0) {
            self.iter.nth(crate::mem::take(&mut self.n) - 1);
        }
        self.iter.next()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        // Can't just add n + self.n due to overflow.
        if self.n > 0 {
            let to_skip = self.n;
            self.n = 0;
            // nth(n) skips n+1
            self.iter.nth(to_skip - 1)?;
        }
        self.iter.nth(n)
    }

    #[inline]
    fn count(mut self) -> usize {
        if self.n > 0 {
            // nth(n) skips n+1
            if self.iter.nth(self.n - 1).is_none() {
                return 0;
            }
        }
        self.iter.count()
    }

    #[inline]
    fn last(mut self) -> Option<I::Item> {
        if self.n > 0 {
            // nth(n) skips n+1
            self.iter.nth(self.n - 1)?;
        }
        self.iter.last()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();

        let lower = lower.saturating_sub(self.n);
        let upper = match upper {
            Some(x) => Some(x.saturating_sub(self.n)),
            None => None,
        };

        (lower, upper)
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        let n = self.n;
        self.n = 0;
        if n > 0 {
            // nth(n) skips n+1
            if self.iter.nth(n - 1).is_none() {
                return try { init };
            }
        }
        self.iter.try_fold(init, fold)
    }

    #[inline]
    fn fold<Acc, Fold>(mut self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        if self.n > 0 {
            // nth(n) skips n+1
            if self.iter.nth(self.n - 1).is_none() {
                return init;
            }
        }
        self.iter.fold(init, fold)
    }

    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item
    where
        Self: TrustedRandomAccess,
    {
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        //
        // Dropping the skipped prefix when index 0 is passed is safe
        // since
        // * the caller passing index 0 means that the inner iterator has more items than `self.n`
        // * TRA contract requires that get_unchecked will only be called once
        //   (unless elements are copyable)
        // * it does not conflict with in-place iteration since index 0 must be accessed
        //   before something is written into the storage used by the prefix
        unsafe {
            if Self::MAY_HAVE_SIDE_EFFECT && idx == 0 {
                for skipped_idx in 0..self.n {
                    drop(try_get_unchecked(&mut self.iter, skipped_idx));
                }
            }

            try_get_unchecked(&mut self.iter, idx + self.n)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Skip<I> where I: ExactSizeIterator {}

#[stable(feature = "double_ended_skip_iterator", since = "1.9.0")]
impl<I> DoubleEndedIterator for Skip<I>
where
    I: DoubleEndedIterator + ExactSizeIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len() > 0 { self.iter.next_back() } else { None }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<I::Item> {
        let len = self.len();
        if n < len {
            self.iter.nth_back(n)
        } else {
            if len > 0 {
                // consume the original iterator
                self.iter.nth_back(len - 1);
            }
            None
        }
    }

    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        fn check<T, Acc, R: Try<Output = Acc>>(
            mut n: usize,
            mut fold: impl FnMut(Acc, T) -> R,
        ) -> impl FnMut(Acc, T) -> ControlFlow<R, Acc> {
            move |acc, x| {
                n -= 1;
                let r = fold(acc, x);
                if n == 0 { ControlFlow::Break(r) } else { ControlFlow::from_try(r) }
            }
        }

        let n = self.len();
        if n == 0 { try { init } } else { self.iter.try_rfold(init, check(n, fold)).into_try() }
    }

    fn rfold<Acc, Fold>(mut self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn ok<Acc, T>(mut f: impl FnMut(Acc, T) -> Acc) -> impl FnMut(Acc, T) -> Result<Acc, !> {
            move |acc, x| Ok(f(acc, x))
        }

        self.try_rfold(init, ok(fold)).unwrap()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> FusedIterator for Skip<I> where I: FusedIterator {}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<S: Iterator, I: Iterator> SourceIter for Skip<I>
where
    I: SourceIter<Source = S>,
{
    type Source = S;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut S {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.iter) }
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I: InPlaceIterable> InPlaceIterable for Skip<I> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<I> TrustedRandomAccess for Skip<I>
where
    I: TrustedRandomAccess,
{
    const MAY_HAVE_SIDE_EFFECT: bool = I::MAY_HAVE_SIDE_EFFECT;
}

// SAFETY: This adapter is shortening. TrustedLen requires the upper bound to be calculated correctly.
// These requirements can only be satisfied when the upper bound of the inner iterator's upper
// bound is never `None`. I: TrustedRandomAccess happens to provide this guarantee while
// I: TrustedLen would not.
#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<I> TrustedLen for Skip<I> where I: Iterator + TrustedRandomAccess {}
