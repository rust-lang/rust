use crate::iter::adapters::{
    zip::try_get_unchecked, TrustedRandomAccess, TrustedRandomAccessNoCoerce,
};
use crate::iter::{FusedIterator, TrustedLen, UncheckedIterator};
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

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }

    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
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

    fn try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
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
}

#[stable(feature = "iter_cloned", since = "1.1.0")]
impl<'a, I, T: 'a> ExactSizeIterator for Cloned<I>
where
    I: ExactSizeIterator<Item = &'a T>,
    T: Clone,
{
    fn len(&self) -> usize {
        self.it.len()
    }

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
