use crate::cmp;
use crate::iter::adapters::SourceIter;
use crate::iter::{FusedIterator, InPlaceIterable, TrustedFused, TrustedLen, TrustedRandomAccess};
use crate::num::NonZero;
use crate::ops::{ControlFlow, Try};

/// An iterator that only iterates over the first `n` iterations of `iter`.
///
/// This `struct` is created by the [`take`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`take`]: Iterator::take
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Take<I> {
    iter: I,
    n: usize,
}

impl<I> Take<I> {
    pub(in crate::iter) fn new(iter: I, n: usize) -> Take<I> {
        Take { iter, n }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Take<I>
where
    I: Iterator,
{
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<<I as Iterator>::Item> {
        if self.n != 0 {
            self.n -= 1;
            self.iter.next()
        } else {
            None
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        if self.n > n {
            self.n -= n + 1;
            self.iter.nth(n)
        } else {
            if self.n > 0 {
                self.iter.nth(self.n - 1);
                self.n = 0;
            }
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.n == 0 {
            return (0, Some(0));
        }

        let (lower, upper) = self.iter.size_hint();

        let lower = cmp::min(lower, self.n);

        let upper = match upper {
            Some(x) if x < self.n => Some(x),
            _ => Some(self.n),
        };

        (lower, upper)
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        fn check<'a, T, Acc, R: Try<Output = Acc>>(
            n: &'a mut usize,
            mut fold: impl FnMut(Acc, T) -> R + 'a,
        ) -> impl FnMut(Acc, T) -> ControlFlow<R, Acc> + 'a {
            move |acc, x| {
                *n -= 1;
                let r = fold(acc, x);
                if *n == 0 { ControlFlow::Break(r) } else { ControlFlow::from_try(r) }
            }
        }

        if self.n == 0 {
            try { init }
        } else {
            let n = &mut self.n;
            self.iter.try_fold(init, check(n, fold)).into_try()
        }
    }

    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        Self::spec_fold(self, init, f)
    }

    #[inline]
    fn for_each<F: FnMut(Self::Item)>(self, f: F) {
        Self::spec_for_each(self, f)
    }

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        let min = self.n.min(n);
        let rem = match self.iter.advance_by(min) {
            Ok(()) => 0,
            Err(rem) => rem.get(),
        };
        let advanced = min - rem;
        self.n -= advanced;
        NonZero::new(n - advanced).map_or(Ok(()), Err)
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I> SourceIter for Take<I>
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
unsafe impl<I: InPlaceIterable> InPlaceIterable for Take<I> {
    const EXPAND_BY: Option<NonZero<usize>> = I::EXPAND_BY;
    const MERGE_BY: Option<NonZero<usize>> = I::MERGE_BY;
}

#[stable(feature = "double_ended_take_iterator", since = "1.38.0")]
impl<I> DoubleEndedIterator for Take<I>
where
    I: DoubleEndedIterator + ExactSizeIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.n == 0 {
            None
        } else {
            let n = self.n;
            self.n -= 1;
            self.iter.nth_back(self.iter.len().saturating_sub(n))
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.iter.len();
        if self.n > n {
            let m = len.saturating_sub(self.n) + n;
            self.n -= n + 1;
            self.iter.nth_back(m)
        } else {
            if len > 0 {
                self.iter.nth_back(len - 1);
            }
            None
        }
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        if self.n == 0 {
            try { init }
        } else {
            let len = self.iter.len();
            if len > self.n && self.iter.nth_back(len - self.n - 1).is_none() {
                try { init }
            } else {
                self.iter.try_rfold(init, fold)
            }
        }
    }

    #[inline]
    fn rfold<Acc, Fold>(mut self, init: Acc, fold: Fold) -> Acc
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        if self.n == 0 {
            init
        } else {
            let len = self.iter.len();
            if len > self.n && self.iter.nth_back(len - self.n - 1).is_none() {
                init
            } else {
                self.iter.rfold(init, fold)
            }
        }
    }

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        // The amount by which the inner iterator needs to be shortened for it to be
        // at most as long as the take() amount.
        let trim_inner = self.iter.len().saturating_sub(self.n);
        // The amount we need to advance inner to fulfill the caller's request.
        // take(), advance_by() and len() all can be at most usize, so we don't have to worry
        // about having to advance more than usize::MAX here.
        let advance_by = trim_inner.saturating_add(n);

        let remainder = match self.iter.advance_back_by(advance_by) {
            Ok(()) => 0,
            Err(rem) => rem.get(),
        };
        let advanced_by_inner = advance_by - remainder;
        let advanced_by = advanced_by_inner - trim_inner;
        self.n -= advanced_by;
        NonZero::new(n - advanced_by).map_or(Ok(()), Err)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Take<I> where I: ExactSizeIterator {}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> FusedIterator for Take<I> where I: FusedIterator {}

#[unstable(issue = "none", feature = "trusted_fused")]
unsafe impl<I: TrustedFused> TrustedFused for Take<I> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<I: TrustedLen> TrustedLen for Take<I> {}

trait SpecTake: Iterator {
    fn spec_fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B;

    fn spec_for_each<F: FnMut(Self::Item)>(self, f: F);
}

impl<I: Iterator> SpecTake for Take<I> {
    #[inline]
    default fn spec_fold<B, F>(mut self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        use crate::ops::NeverShortCircuit;
        self.try_fold(init, NeverShortCircuit::wrap_mut_2(f)).0
    }

    #[inline]
    default fn spec_for_each<F: FnMut(Self::Item)>(mut self, f: F) {
        // The default implementation would use a unit accumulator, so we can
        // avoid a stateful closure by folding over the remaining number
        // of items we wish to return instead.
        fn check<'a, Item>(
            mut action: impl FnMut(Item) + 'a,
        ) -> impl FnMut(usize, Item) -> Option<usize> + 'a {
            move |more, x| {
                action(x);
                more.checked_sub(1)
            }
        }

        let remaining = self.n;
        if remaining > 0 {
            self.iter.try_fold(remaining - 1, check(f));
        }
    }
}

impl<I: Iterator + TrustedRandomAccess> SpecTake for Take<I> {
    #[inline]
    fn spec_fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;
        let end = self.n.min(self.iter.size());
        for i in 0..end {
            // SAFETY: i < end <= self.iter.size() and we discard the iterator at the end
            let val = unsafe { self.iter.__iterator_get_unchecked(i) };
            acc = f(acc, val);
        }
        acc
    }

    #[inline]
    fn spec_for_each<F: FnMut(Self::Item)>(mut self, mut f: F) {
        let end = self.n.min(self.iter.size());
        for i in 0..end {
            // SAFETY: i < end <= self.iter.size() and we discard the iterator at the end
            let val = unsafe { self.iter.__iterator_get_unchecked(i) };
            f(val);
        }
    }
}

#[stable(feature = "exact_size_take_repeat", since = "1.82.0")]
impl<T: Clone> DoubleEndedIterator for Take<crate::iter::Repeat<T>> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.next()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.nth(n)
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        self.try_fold(init, fold)
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.fold(init, fold)
    }

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.advance_by(n)
    }
}

// Note: It may be tempting to impl DoubleEndedIterator for Take<RepeatWith>.
// One must fight that temptation since such implementation wouldn’t be correct
// because we have no way to return value of nth invocation of repeater followed
// by n-1st without remembering all results.

#[stable(feature = "exact_size_take_repeat", since = "1.82.0")]
impl<T: Clone> ExactSizeIterator for Take<crate::iter::Repeat<T>> {
    fn len(&self) -> usize {
        self.n
    }
}

#[stable(feature = "exact_size_take_repeat", since = "1.82.0")]
impl<F: FnMut() -> A, A> ExactSizeIterator for Take<crate::iter::RepeatWith<F>> {
    fn len(&self) -> usize {
        self.n
    }
}
