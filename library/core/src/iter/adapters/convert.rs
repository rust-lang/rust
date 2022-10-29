use crate::iter::adapters::{
    zip::try_get_unchecked, TrustedRandomAccess, TrustedRandomAccessNoCoerce,
};
use crate::iter::{FusedIterator, TrustedLen};
use crate::marker::PhantomData;
use crate::ops::Try;

/// An iterator that converts the elements of an underlying iterator.
///
/// This `struct` is created by the [`map_into`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`map_into`]: Iterator::map_into
/// [`Iterator`]: trait.Iterator.html
#[unstable(feature = "converting_iterators", issue = "none")]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
pub struct MapInto<I, T> {
    it: I,
    into: PhantomData<T>,
}

impl<I, T> MapInto<I, T> {
    pub(in crate::iter) fn new(it: I) -> MapInto<I, T> {
        MapInto { it, into: PhantomData }
    }
}

fn try_fold_into<T: Into<U>, U, Acc, R>(mut f: impl FnMut(Acc, U) -> R) -> impl FnMut(Acc, T) -> R {
    move |acc, elt| f(acc, elt.into())
}

#[unstable(feature = "converting_iterators", issue = "none")]
impl<I, T> Iterator for MapInto<I, T>
where
    I: Iterator,
    I::Item: Into<T>,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.it.next().map(Into::into)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }

    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        self.it.try_fold(init, try_fold_into(f))
    }

    fn fold<Acc, F>(self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        self.it.map(Into::into).fold(init, f)
    }

    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> T
    where
        Self: TrustedRandomAccessNoCoerce,
    {
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        unsafe { try_get_unchecked(&mut self.it, idx).into() }
    }
}

#[unstable(feature = "converting_iterators", issue = "none")]
impl<I, T> DoubleEndedIterator for MapInto<I, T>
where
    I: DoubleEndedIterator,
    I::Item: Into<T>,
{
    fn next_back(&mut self) -> Option<T> {
        self.it.next_back().map(Into::into)
    }

    fn try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        self.it.try_rfold(init, try_fold_into(f))
    }

    fn rfold<Acc, F>(self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        self.it.map(Into::into).rfold(init, f)
    }
}

#[unstable(feature = "converting_iterators", issue = "none")]
impl<I, T> ExactSizeIterator for MapInto<I, T>
where
    I: ExactSizeIterator,
    I::Item: Into<T>,
{
    fn len(&self) -> usize {
        self.it.len()
    }

    fn is_empty(&self) -> bool {
        self.it.is_empty()
    }
}

#[unstable(feature = "converting_iterators", issue = "none")]
impl<I, T> FusedIterator for MapInto<I, T>
where
    I: FusedIterator,
    I::Item: Into<T>,
{
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<I, T> TrustedRandomAccess for MapInto<I, T> where I: TrustedRandomAccess {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<I, T> TrustedRandomAccessNoCoerce for MapInto<I, T>
where
    I: TrustedRandomAccessNoCoerce,
{
    const MAY_HAVE_SIDE_EFFECT: bool = true;
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<I, T> TrustedLen for MapInto<I, T>
where
    I: TrustedLen,
    I::Item: Into<T>,
{
}
