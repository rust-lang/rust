use crate::fmt;
use crate::iter::adapters::SourceIter;
use crate::iter::{FusedIterator, InPlaceIterable, TrustedFused};
use crate::num::NonZero;
use crate::ops::Try;

/// An iterator that rejects elements while `predicate` returns `true`.
///
/// This `struct` is created by the [`skip_while`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`skip_while`]: Iterator::skip_while
/// [`Iterator`]: trait.Iterator.html
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct SkipWhile<I, P> {
    iter: I,
    flag: bool,
    predicate: P,
}

impl<I, P> SkipWhile<I, P> {
    pub(in crate::iter) fn new(iter: I, predicate: P) -> SkipWhile<I, P> {
        SkipWhile { iter, flag: false, predicate }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<I: fmt::Debug, P> fmt::Debug for SkipWhile<I, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SkipWhile").field("iter", &self.iter).field("flag", &self.flag).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator, P> Iterator for SkipWhile<I, P>
where
    P: FnMut(&I::Item) -> bool,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        fn check<'a, T>(
            flag: &'a mut bool,
            pred: &'a mut impl FnMut(&T) -> bool,
        ) -> impl FnMut(&T) -> bool + 'a {
            move |x| {
                if *flag || !pred(x) {
                    *flag = true;
                    true
                } else {
                    false
                }
            }
        }

        let flag = &mut self.flag;
        let pred = &mut self.predicate;
        self.iter.find(check(flag, pred))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, mut init: Acc, mut fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        if !self.flag {
            match self.next() {
                Some(v) => init = fold(init, v)?,
                None => return try { init },
            }
        }
        self.iter.try_fold(init, fold)
    }

    #[inline]
    fn fold<Acc, Fold>(mut self, mut init: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        if !self.flag {
            match self.next() {
                Some(v) => init = fold(init, v),
                None => return init,
            }
        }
        self.iter.fold(init, fold)
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I, P> FusedIterator for SkipWhile<I, P>
where
    I: FusedIterator,
    P: FnMut(&I::Item) -> bool,
{
}

#[unstable(issue = "none", feature = "trusted_fused")]
unsafe impl<I: TrustedFused, P> TrustedFused for SkipWhile<I, P> {}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<P, I> SourceIter for SkipWhile<I, P>
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
unsafe impl<I: InPlaceIterable, F> InPlaceIterable for SkipWhile<I, F> {
    const EXPAND_BY: Option<NonZero<usize>> = I::EXPAND_BY;
    const MERGE_BY: Option<NonZero<usize>> = I::MERGE_BY;
}
