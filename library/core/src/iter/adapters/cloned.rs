use core::num::NonZero;

use crate::cmp::Ordering;
use crate::iter::adapters::zip::try_get_unchecked;
use crate::iter::adapters::{SourceIter, TrustedRandomAccess, TrustedRandomAccessNoCoerce};
use crate::iter::{FusedIterator, InPlaceIterable, TrustedLen, UncheckedIterator};
use crate::ops::Try;

/// An iterator that clones the elements of an underlying iterator.
///
/// This `struct` is created by the [`cloned`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`cloned`]: Iterator::cloned
/// [`Iterator`]: trait.Iterator.html
#[stable(feature = "iter_cloned", since = "1.1.0")]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
pub struct Cloned<I> {
    it: I,
}

impl<I> Cloned<I> {
    pub(in crate::iter) fn new(it: I) -> Cloned<I> {
        Cloned { it }
    }
}

fn clone_try_fold<T: Clone, Acc, R>(mut f: impl FnMut(Acc, T) -> R) -> impl FnMut(Acc, &T) -> R {
    move |acc, elt| f(acc, elt.clone())
}

#[stable(feature = "iter_cloned", since = "1.1.0")]
impl<'a, I, T: 'a> Iterator for Cloned<I>
where
    I: Iterator<Item = &'a T>,
    T: Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.it.next().cloned()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.it.count()
    }

    fn last(self) -> Option<T> {
        self.it.last().cloned()
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.it.advance_by(n)
    }

    fn nth(&mut self, n: usize) -> Option<T> {
        self.it.nth(n).cloned()
    }

    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        self.it.try_fold(init, clone_try_fold(f))
    }

    fn fold<Acc, F>(self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        self.it.map(T::clone).fold(init, f)
    }

    fn find<P>(&mut self, mut predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        self.it.find(move |x| predicate(&x)).cloned()
    }

    fn max_by<F>(self, mut compare: F) -> Option<Self::Item>
    where
        F: FnMut(&Self::Item, &Self::Item) -> Ordering,
    {
        self.it.max_by(move |&x, &y| compare(x, y)).cloned()
    }

    fn min_by<F>(self, mut compare: F) -> Option<Self::Item>
    where
        F: FnMut(&Self::Item, &Self::Item) -> Ordering,
    {
        self.it.min_by(move |&x, &y| compare(x, y)).cloned()
    }

    fn cmp<O>(self, other: O) -> Ordering
    where
        O: IntoIterator<Item = Self::Item>,
        Self::Item: Ord,
    {
        self.it.cmp_by(other, |x, y| x.cmp(&y))
    }

    fn partial_cmp<O>(self, other: O) -> Option<Ordering>
    where
        O: IntoIterator,
        Self::Item: PartialOrd<O::Item>,
    {
        self.it.partial_cmp_by(other, |x, y| x.partial_cmp(&y))
    }

    fn eq<O>(self, other: O) -> bool
    where
        O: IntoIterator,
        Self::Item: PartialEq<O::Item>,
    {
        self.it.eq_by(other, |x, y| x == &y)
    }

    fn is_sorted_by<F>(self, mut compare: F) -> bool
    where
        F: FnMut(&Self::Item, &Self::Item) -> bool,
    {
        self.it.is_sorted_by(move |&x, &y| compare(x, y))
    }

    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> T
    where
        Self: TrustedRandomAccessNoCoerce,
    {
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        unsafe { try_get_unchecked(&mut self.it, idx).clone() }
    }
}

#[stable(feature = "iter_cloned", since = "1.1.0")]
impl<'a, I, T: 'a> DoubleEndedIterator for Cloned<I>
where
    I: DoubleEndedIterator<Item = &'a T>,
    T: Clone,
{
    fn next_back(&mut self) -> Option<T> {
        self.it.next_back().cloned()
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.it.advance_back_by(n)
    }

    fn try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        self.it.try_rfold(init, clone_try_fold(f))
    }

    fn rfold<Acc, F>(self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        self.it.map(T::clone).rfold(init, f)
    }

    fn rfind<P>(&mut self, mut predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        self.it.rfind(move |x| predicate(&x)).cloned()
    }
}

#[stable(feature = "iter_cloned", since = "1.1.0")]
impl<'a, I, T: 'a> ExactSizeIterator for Cloned<I>
where
    I: ExactSizeIterator<Item = &'a T>,
    T: Clone,
{
    #[inline]
    fn len(&self) -> usize {
        self.it.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.it.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<'a, I, T: 'a> FusedIterator for Cloned<I>
where
    I: FusedIterator<Item = &'a T>,
    T: Clone,
{
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<I> TrustedRandomAccess for Cloned<I> where I: TrustedRandomAccess {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<I> TrustedRandomAccessNoCoerce for Cloned<I>
where
    I: TrustedRandomAccessNoCoerce,
{
    const MAY_HAVE_SIDE_EFFECT: bool = true;
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<'a, I, T: 'a> TrustedLen for Cloned<I>
where
    I: TrustedLen<Item = &'a T>,
    T: Clone,
{
}

impl<'a, I, T: 'a> UncheckedIterator for Cloned<I>
where
    I: UncheckedIterator<Item = &'a T>,
    T: Clone,
{
    unsafe fn next_unchecked(&mut self) -> T {
        // SAFETY: `Cloned` is 1:1 with the inner iterator, so if the caller promised
        // that there's an element left, the inner iterator has one too.
        let item = unsafe { self.it.next_unchecked() };
        item.clone()
    }
}

#[stable(feature = "default_iters", since = "1.70.0")]
impl<I: Default> Default for Cloned<I> {
    /// Creates a `Cloned` iterator from the default value of `I`
    /// ```
    /// # use core::slice;
    /// # use core::iter::Cloned;
    /// let iter: Cloned<slice::Iter<'_, u8>> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        Self::new(Default::default())
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I> SourceIter for Cloned<I>
where
    I: SourceIter,
{
    type Source = I::Source;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut I::Source {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.it) }
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I: InPlaceIterable> InPlaceIterable for Cloned<I> {
    const EXPAND_BY: Option<NonZero<usize>> = I::EXPAND_BY;
    const MERGE_BY: Option<NonZero<usize>> = I::MERGE_BY;
}
