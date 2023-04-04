use crate::intrinsics::unlikely;
use crate::iter::{adapters::SourceIter, FusedIterator, InPlaceIterable};
use crate::num::NonZeroUsize;
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
            self.iter.nth(crate::mem::take(&mut self.n))
        } else {
            self.iter.next()
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        if self.n > 0 {
            let skip: usize = crate::mem::take(&mut self.n);
            // Checked add to handle overflow case.
            let n = match skip.checked_add(n) {
                Some(nth) => nth,
                None => {
                    // In case of overflow, load skip value, before loading `n`.
                    // Because the amount of elements to iterate is beyond `usize::MAX`, this
                    // is split into two `nth` calls where the `skip` `nth` call is discarded.
                    self.iter.nth(skip - 1)?;
                    n
                }
            };
            // Load nth element including skip.
            self.iter.nth(n)
        } else {
            self.iter.nth(n)
        }
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

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn advance_by(&mut self, mut n: usize) -> Result<(), NonZeroUsize> {
        let skip_inner = self.n;
        let skip_and_advance = skip_inner.saturating_add(n);

        let remainder = match self.iter.advance_by(skip_and_advance) {
            Ok(()) => 0,
            Err(n) => n.get(),
        };
        let advanced_inner = skip_and_advance - remainder;
        n -= advanced_inner.saturating_sub(skip_inner);
        self.n = self.n.saturating_sub(advanced_inner);

        // skip_and_advance may have saturated
        if unlikely(remainder == 0 && n > 0) {
            n = match self.iter.advance_by(n) {
                Ok(()) => 0,
                Err(n) => n.get(),
            }
        }

        NonZeroUsize::new(n).map_or(Ok(()), Err)
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

    impl_fold_via_try_fold! { rfold -> try_rfold }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        let min = crate::cmp::min(self.len(), n);
        let rem = self.iter.advance_back_by(min);
        assert!(rem.is_ok(), "ExactSizeIterator contract violation");
        NonZeroUsize::new(n - min).map_or(Ok(()), Err)
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> FusedIterator for Skip<I> where I: FusedIterator {}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I> SourceIter for Skip<I>
where
    I: SourceIter,
{
    type Source = I::Source;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut I::Source {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.iter) }
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I: InPlaceIterable> InPlaceIterable for Skip<I> {}
