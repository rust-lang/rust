use crate::intrinsics;
use crate::iter::adapters::{zip::try_get_unchecked, InPlaceIterable, SourceIter};
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

// NOTE: for `I: FusedIterator`, we assume that the iterator is always `Some`.
// Implementing this as a directly-expanded macro helps codegen performance.
macro_rules! unchecked {
    ($self:ident) => {
        match $self {
            Fuse { iter: Some(iter) } => iter,
            // SAFETY: the specialized iterator never sets `None`
            Fuse { iter: None } => unsafe { intrinsics::unreachable() },
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
        FuseImpl::next(self)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        FuseImpl::nth(self, n)
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        FuseImpl::last(self)
    }

    #[inline]
    fn count(self) -> usize {
        FuseImpl::count(self)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        FuseImpl::size_hint(self)
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
    fn fold<Acc, Fold>(self, acc: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        FuseImpl::fold(self, acc, fold)
    }

    #[inline]
    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        FuseImpl::find(self, predicate)
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
    fn rfold<Acc, Fold>(self, acc: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        FuseImpl::rfold(self, acc, fold)
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
        FuseImpl::len(self)
    }

    fn is_empty(&self) -> bool {
        FuseImpl::is_empty(self)
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

// Fuse specialization trait
#[doc(hidden)]
trait FuseImpl<I> {
    type Item;

    // Functions specific to any normal Iterators
    fn next(&mut self) -> Option<Self::Item>;
    fn nth(&mut self, n: usize) -> Option<Self::Item>;
    fn last(self) -> Option<Self::Item>;
    fn count(self) -> usize;
    fn size_hint(&self) -> (usize, Option<usize>);
    fn try_fold<Acc, Fold, R>(&mut self, acc: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>;
    fn fold<Acc, Fold>(self, acc: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc;
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
    fn rfold<Acc, Fold>(self, acc: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
        I: DoubleEndedIterator;
    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
        I: DoubleEndedIterator;

    // Functions specific to ExactSizeIterator
    fn len(&self) -> usize
    where
        I: ExactSizeIterator;
    fn is_empty(&self) -> bool
    where
        I: ExactSizeIterator;
}

// General Fuse impl
#[doc(hidden)]
impl<I> FuseImpl<I> for Fuse<I>
where
    I: Iterator,
{
    type Item = <I as Iterator>::Item;

    #[inline]
    default fn next(&mut self) -> Option<<I as Iterator>::Item> {
        fuse!(self.iter.next())
    }

    #[inline]
    default fn nth(&mut self, n: usize) -> Option<I::Item> {
        fuse!(self.iter.nth(n))
    }

    #[inline]
    default fn last(self) -> Option<I::Item> {
        match self.iter {
            Some(iter) => iter.last(),
            None => None,
        }
    }

    #[inline]
    default fn count(self) -> usize {
        match self.iter {
            Some(iter) => iter.count(),
            None => 0,
        }
    }

    #[inline]
    default fn size_hint(&self) -> (usize, Option<usize>) {
        match self.iter {
            Some(ref iter) => iter.size_hint(),
            None => (0, Some(0)),
        }
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
    default fn fold<Acc, Fold>(self, mut acc: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        if let Some(iter) = self.iter {
            acc = iter.fold(acc, fold);
        }
        acc
    }

    #[inline]
    default fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        fuse!(self.iter.find(predicate))
    }

    #[inline]
    default fn next_back(&mut self) -> Option<<I as Iterator>::Item>
    where
        I: DoubleEndedIterator,
    {
        fuse!(self.iter.next_back())
    }

    #[inline]
    default fn nth_back(&mut self, n: usize) -> Option<<I as Iterator>::Item>
    where
        I: DoubleEndedIterator,
    {
        fuse!(self.iter.nth_back(n))
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
    default fn rfold<Acc, Fold>(self, mut acc: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
        I: DoubleEndedIterator,
    {
        if let Some(iter) = self.iter {
            acc = iter.rfold(acc, fold);
        }
        acc
    }

    #[inline]
    default fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
        I: DoubleEndedIterator,
    {
        fuse!(self.iter.rfind(predicate))
    }

    #[inline]
    default fn len(&self) -> usize
    where
        I: ExactSizeIterator,
    {
        match self.iter {
            Some(ref iter) => iter.len(),
            None => 0,
        }
    }

    #[inline]
    default fn is_empty(&self) -> bool
    where
        I: ExactSizeIterator,
    {
        match self.iter {
            Some(ref iter) => iter.is_empty(),
            None => true,
        }
    }
}

#[doc(hidden)]
impl<I> FuseImpl<I> for Fuse<I>
where
    I: FusedIterator,
{
    #[inline]
    fn next(&mut self) -> Option<<I as Iterator>::Item> {
        unchecked!(self).next()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        unchecked!(self).nth(n)
    }

    #[inline]
    fn last(self) -> Option<I::Item> {
        unchecked!(self).last()
    }

    #[inline]
    fn count(self) -> usize {
        unchecked!(self).count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        unchecked!(self).size_hint()
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        unchecked!(self).try_fold(init, fold)
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        unchecked!(self).fold(init, fold)
    }

    #[inline]
    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        unchecked!(self).find(predicate)
    }

    #[inline]
    fn next_back(&mut self) -> Option<<I as Iterator>::Item>
    where
        I: DoubleEndedIterator,
    {
        unchecked!(self).next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<<I as Iterator>::Item>
    where
        I: DoubleEndedIterator,
    {
        unchecked!(self).nth_back(n)
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
        I: DoubleEndedIterator,
    {
        unchecked!(self).try_rfold(init, fold)
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
        I: DoubleEndedIterator,
    {
        unchecked!(self).rfold(init, fold)
    }

    #[inline]
    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
        I: DoubleEndedIterator,
    {
        unchecked!(self).rfind(predicate)
    }

    #[inline]
    fn len(&self) -> usize
    where
        I: ExactSizeIterator,
    {
        unchecked!(self).len()
    }

    #[inline]
    fn is_empty(&self) -> bool
    where
        I: ExactSizeIterator,
    {
        unchecked!(self).is_empty()
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<S: Iterator, I: FusedIterator> SourceIter for Fuse<I>
where
    I: SourceIter<Source = S>,
{
    type Source = S;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut S {
        match self.iter {
            // SAFETY: unsafe function forwarding to unsafe function with the same requirements
            Some(ref mut iter) => unsafe { SourceIter::as_inner(iter) },
            // SAFETY: the specialized iterator never sets `None`
            None => unsafe { intrinsics::unreachable() },
        }
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I: InPlaceIterable> InPlaceIterable for Fuse<I> {}
