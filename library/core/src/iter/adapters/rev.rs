use crate::iter::{FusedIterator, TrustedLen};
use crate::ops::Try;

/// A double-ended iterator with the direction inverted.
///
/// This `struct` is created by the [`rev`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`rev`]: Iterator::rev
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Rev<T> {
    iter: T,
}

impl<T> Rev<T> {
    pub(in crate::iter) fn new(iter: T) -> Rev<T> {
        Rev { iter }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Rev<I>
where
    I: DoubleEndedIterator,
{
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<<I as Iterator>::Item> {
        self.iter.next_back()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), usize> {
        self.iter.advance_back_by(n)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<<I as Iterator>::Item> {
        self.iter.nth_back(n)
    }

    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok = B>,
    {
        self.iter.try_rfold(init, f)
    }

    fn fold<Acc, F>(self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        self.iter.rfold(init, f)
    }

    #[inline]
    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        self.iter.rfind(predicate)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> DoubleEndedIterator for Rev<I>
where
    I: DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<<I as Iterator>::Item> {
        self.iter.next()
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), usize> {
        self.iter.advance_by(n)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<<I as Iterator>::Item> {
        self.iter.nth(n)
    }

    fn try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok = B>,
    {
        self.iter.try_fold(init, f)
    }

    fn rfold<Acc, F>(self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        self.iter.fold(init, f)
    }

    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        self.iter.find(predicate)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Rev<I>
where
    I: ExactSizeIterator + DoubleEndedIterator,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> FusedIterator for Rev<I> where I: FusedIterator + DoubleEndedIterator {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<I> TrustedLen for Rev<I> where I: TrustedLen + DoubleEndedIterator {}
