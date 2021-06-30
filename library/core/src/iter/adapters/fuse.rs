use crate::intrinsics;
use crate::iter::adapters::zip::try_get_unchecked;
use crate::iter::{
    DoubleEndedIterator, ExactSizeIterator, FusedIterator, TrustedLen, TrustedRandomAccess,
};
use crate::ops::Try;

/// An iterator that yields `None` forever after the underlying iterator
/// yields `None` once.
///
/// This `struct` is created by [`Iterator::fuse`]. See its documentation
/// for more.
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Fuse<I> {
    // NOTE: for `I: FusedIterator`, this is always assumed `Some`!
    iter: Option<I>,
}
impl<I> Fuse<I> {
    pub(in crate::iter) fn new(iter: I) -> Fuse<I> {
        Fuse { iter: Some(iter) }
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> FusedIterator for Fuse<I> where I: Iterator {}

/// Fuse the iterator if the expression is `None`.
macro_rules! fuse {
    ($self:ident . iter . $($call:tt)+) => {
        match $self.iter {
            Some(ref mut iter) => match iter.$($call)+ {
                None => {
                    $self.iter = None;
                    None
                }
                item => item,
            },
            None => None,
        }
    };
}

// Any implementation here is made internal to avoid exposing default fns outside this trait
#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Fuse<I>
where
    I: Iterator,
{
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        fuse!(self.iter.next())
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        fuse!(self.iter.nth(n))
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        match self.iter {
            Some(iter) => iter.last(),
            None => None,
        }
    }

    #[inline]
    fn count(self) -> usize {
        match self.iter {
            Some(iter) => iter.count(),
            None => 0,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.iter {
            Some(ref iter) => iter.size_hint(),
            None => (0, Some(0)),
        }
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, mut acc: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        if let Some(ref mut iter) = self.iter {
            acc = iter.try_fold(acc, fold)?;
            self.iter = None;
        }
        try { acc }
    }

    #[inline]
    fn fold<Acc, Fold>(self, mut acc: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        if let Some(iter) = self.iter {
            acc = iter.fold(acc, fold);
        }
        acc
    }

    #[inline]
    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        fuse!(self.iter.find(predicate))
    }

    #[inline]
    #[doc(hidden)]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item
    where
        Self: TrustedRandomAccess,
    {
        match self.iter {
            // SAFETY: the caller must uphold the contract for
            // `Iterator::__iterator_get_unchecked`.
            Some(ref mut iter) => unsafe { try_get_unchecked(iter, idx) },
            // SAFETY: the caller asserts there is an item at `i`, so we're not exhausted.
            None => unsafe { intrinsics::unreachable() },
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> DoubleEndedIterator for Fuse<I>
where
    I: DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<<I as Iterator>::Item> {
        fuse!(self.iter.next_back())
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<<I as Iterator>::Item> {
        fuse!(self.iter.nth_back(n))
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, mut acc: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        if let Some(ref mut iter) = self.iter {
            acc = iter.try_rfold(acc, fold)?;
            self.iter = None;
        }
        try { acc }
    }

    #[inline]
    fn rfold<Acc, Fold>(self, mut acc: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        if let Some(iter) = self.iter {
            acc = iter.rfold(acc, fold);
        }
        acc
    }

    #[inline]
    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        fuse!(self.iter.rfind(predicate))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Fuse<I>
where
    I: ExactSizeIterator,
{
    fn len(&self) -> usize {
        match self.iter {
            Some(ref iter) => iter.len(),
            None => 0,
        }
    }

    fn is_empty(&self) -> bool {
        match self.iter {
            Some(ref iter) => iter.is_empty(),
            None => true,
        }
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
// SAFETY: `TrustedLen` requires that an accurate length is reported via `size_hint()`. As `Fuse`
// is just forwarding this to the wrapped iterator `I` this property is preserved and it is safe to
// implement `TrustedLen` here.
unsafe impl<I> TrustedLen for Fuse<I> where I: TrustedLen {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
// SAFETY: `TrustedRandomAccess` requires that `size_hint()` must be exact and cheap to call, and
// `Iterator::__iterator_get_unchecked()` must be implemented accordingly.
//
// This is safe to implement as `Fuse` is just forwarding these to the wrapped iterator `I`, which
// preserves these properties.
unsafe impl<I> TrustedRandomAccess for Fuse<I>
where
    I: TrustedRandomAccess,
{
    const MAY_HAVE_SIDE_EFFECT: bool = I::MAY_HAVE_SIDE_EFFECT;
}
