use crate::intrinsics;
use crate::iter::adapters::zip::try_get_unchecked;
use crate::iter::{
    DoubleEndedIterator, ExactSizeIterator, FusedIterator, TrustedLen, TrustedRandomAccess,
    TrustedRandomAccessNoCoerce,
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
    // NOTE: for `I: FusedIterator`, we never bother setting `None`, but
    // we still have to be prepared for that state due to variance.
    // See rust-lang/rust#85863
    iter: Option<I>,
}
impl<I> Fuse<I> {
    pub(in crate::iter) fn new(iter: I) -> Fuse<I> {
        Fuse { iter: Some(iter) }
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> FusedIterator for Fuse<I> where I: Iterator {}

// Any specialized implementation here is made internal
// to avoid exposing default fns outside this trait.
#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Fuse<I>
where
    I: Iterator,
{
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        FuseImpl::next(self)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        FuseImpl::nth(self, n)
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
    fn try_fold<Acc, Fold, R>(&mut self, acc: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        FuseImpl::try_fold(self, acc, fold)
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
        FuseImpl::find(self, predicate)
    }

    #[inline]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item
    where
        Self: TrustedRandomAccessNoCoerce,
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
        FuseImpl::next_back(self)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<<I as Iterator>::Item> {
        FuseImpl::nth_back(self, n)
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, acc: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        FuseImpl::try_rfold(self, acc, fold)
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
        FuseImpl::rfind(self, predicate)
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

#[stable(feature = "default_iters", since = "CURRENT_RUSTC_VERSION")]
impl<I: Default> Default for Fuse<I> {
    /// Creates a `Fuse` iterator from the default value of `I`.
    ///
    /// ```
    /// # use core::slice;
    /// # use std::iter::Fuse;
    /// let iter: Fuse<slice::Iter<'_, u8>> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        Fuse { iter: Default::default() }
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
unsafe impl<I> TrustedRandomAccess for Fuse<I> where I: TrustedRandomAccess {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<I> TrustedRandomAccessNoCoerce for Fuse<I>
where
    I: TrustedRandomAccessNoCoerce,
{
    const MAY_HAVE_SIDE_EFFECT: bool = I::MAY_HAVE_SIDE_EFFECT;
}

/// Fuse specialization trait
///
/// We only need to worry about `&mut self` methods, which
/// may exhaust the iterator without consuming it.
#[doc(hidden)]
trait FuseImpl<I> {
    type Item;

    // Functions specific to any normal Iterators
    fn next(&mut self) -> Option<Self::Item>;
    fn nth(&mut self, n: usize) -> Option<Self::Item>;
    fn try_fold<Acc, Fold, R>(&mut self, acc: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>;
    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool;

    // Functions specific to DoubleEndedIterators
    fn next_back(&mut self) -> Option<Self::Item>
    where
        I: DoubleEndedIterator;
    fn nth_back(&mut self, n: usize) -> Option<Self::Item>
    where
        I: DoubleEndedIterator;
    fn try_rfold<Acc, Fold, R>(&mut self, acc: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
        I: DoubleEndedIterator;
    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
        I: DoubleEndedIterator;
}

/// General `Fuse` impl which sets `iter = None` when exhausted.
#[doc(hidden)]
impl<I> FuseImpl<I> for Fuse<I>
where
    I: Iterator,
{
    type Item = <I as Iterator>::Item;

    #[inline]
    default fn next(&mut self) -> Option<<I as Iterator>::Item> {
        and_then_or_clear(&mut self.iter, Iterator::next)
    }

    #[inline]
    default fn nth(&mut self, n: usize) -> Option<I::Item> {
        and_then_or_clear(&mut self.iter, |iter| iter.nth(n))
    }

    #[inline]
    default fn try_fold<Acc, Fold, R>(&mut self, mut acc: Acc, fold: Fold) -> R
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
    default fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        and_then_or_clear(&mut self.iter, |iter| iter.find(predicate))
    }

    #[inline]
    default fn next_back(&mut self) -> Option<<I as Iterator>::Item>
    where
        I: DoubleEndedIterator,
    {
        and_then_or_clear(&mut self.iter, |iter| iter.next_back())
    }

    #[inline]
    default fn nth_back(&mut self, n: usize) -> Option<<I as Iterator>::Item>
    where
        I: DoubleEndedIterator,
    {
        and_then_or_clear(&mut self.iter, |iter| iter.nth_back(n))
    }

    #[inline]
    default fn try_rfold<Acc, Fold, R>(&mut self, mut acc: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
        I: DoubleEndedIterator,
    {
        if let Some(ref mut iter) = self.iter {
            acc = iter.try_rfold(acc, fold)?;
            self.iter = None;
        }
        try { acc }
    }

    #[inline]
    default fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
        I: DoubleEndedIterator,
    {
        and_then_or_clear(&mut self.iter, |iter| iter.rfind(predicate))
    }
}

/// Specialized `Fuse` impl which doesn't bother clearing `iter` when exhausted.
/// However, we must still be prepared for the possibility that it was already cleared!
#[doc(hidden)]
impl<I> FuseImpl<I> for Fuse<I>
where
    I: FusedIterator,
{
    #[inline]
    fn next(&mut self) -> Option<<I as Iterator>::Item> {
        self.iter.as_mut()?.next()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        self.iter.as_mut()?.nth(n)
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
        }
        try { acc }
    }

    #[inline]
    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        self.iter.as_mut()?.find(predicate)
    }

    #[inline]
    fn next_back(&mut self) -> Option<<I as Iterator>::Item>
    where
        I: DoubleEndedIterator,
    {
        self.iter.as_mut()?.next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<<I as Iterator>::Item>
    where
        I: DoubleEndedIterator,
    {
        self.iter.as_mut()?.nth_back(n)
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, mut acc: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
        I: DoubleEndedIterator,
    {
        if let Some(ref mut iter) = self.iter {
            acc = iter.try_rfold(acc, fold)?;
        }
        try { acc }
    }

    #[inline]
    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
        I: DoubleEndedIterator,
    {
        self.iter.as_mut()?.rfind(predicate)
    }
}

#[inline]
fn and_then_or_clear<T, U>(opt: &mut Option<T>, f: impl FnOnce(&mut T) -> Option<U>) -> Option<U> {
    let x = f(opt.as_mut()?);
    if x.is_none() {
        *opt = None;
    }
    x
}
