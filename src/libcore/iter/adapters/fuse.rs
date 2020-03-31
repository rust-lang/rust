use crate::intrinsics;
use crate::iter::{
    DoubleEndedIterator, ExactSizeIterator, FusedIterator, Iterator, TrustedRandomAccess,
};
use crate::ops::Try;

/// An iterator that yields `None` forever after the underlying iterator
/// yields `None` once.
///
/// This `struct` is created by the [`fuse`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`fuse`]: trait.Iterator.html#method.fuse
/// [`Iterator`]: trait.Iterator.html
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

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Fuse<I>
where
    I: Iterator,
{
    type Item = <I as Iterator>::Item;

    #[inline]
    default fn next(&mut self) -> Option<<I as Iterator>::Item> {
        let next = self.iter.as_mut()?.next();
        if next.is_none() {
            self.iter = None;
        }
        next
    }

    #[inline]
    default fn nth(&mut self, n: usize) -> Option<I::Item> {
        let nth = self.iter.as_mut()?.nth(n);
        if nth.is_none() {
            self.iter = None;
        }
        nth
    }

    #[inline]
    default fn last(self) -> Option<I::Item> {
        self.iter?.last()
    }

    #[inline]
    default fn count(self) -> usize {
        self.iter.map_or(0, I::count)
    }

    #[inline]
    default fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.as_ref().map_or((0, Some(0)), I::size_hint)
    }

    #[inline]
    default fn try_fold<Acc, Fold, R>(&mut self, mut acc: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        if let Some(ref mut iter) = self.iter {
            acc = iter.try_fold(acc, fold)?;
            self.iter = None;
        }
        Try::from_ok(acc)
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
        let found = self.iter.as_mut()?.find(predicate);
        if found.is_none() {
            self.iter = None;
        }
        found
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> DoubleEndedIterator for Fuse<I>
where
    I: DoubleEndedIterator,
{
    #[inline]
    default fn next_back(&mut self) -> Option<<I as Iterator>::Item> {
        let next = self.iter.as_mut()?.next_back();
        if next.is_none() {
            self.iter = None;
        }
        next
    }

    #[inline]
    default fn nth_back(&mut self, n: usize) -> Option<<I as Iterator>::Item> {
        let nth = self.iter.as_mut()?.nth_back(n);
        if nth.is_none() {
            self.iter = None;
        }
        nth
    }

    #[inline]
    default fn try_rfold<Acc, Fold, R>(&mut self, mut acc: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        if let Some(ref mut iter) = self.iter {
            acc = iter.try_rfold(acc, fold)?;
            self.iter = None;
        }
        Try::from_ok(acc)
    }

    #[inline]
    default fn rfold<Acc, Fold>(self, mut acc: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
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
    {
        let found = self.iter.as_mut()?.rfind(predicate);
        if found.is_none() {
            self.iter = None;
        }
        found
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Fuse<I>
where
    I: ExactSizeIterator,
{
    default fn len(&self) -> usize {
        self.iter.as_ref().map_or(0, I::len)
    }

    default fn is_empty(&self) -> bool {
        self.iter.as_ref().map_or(true, I::is_empty)
    }
}

// NOTE: for `I: FusedIterator`, we assume that the iterator is always `Some`
impl<I: FusedIterator> Fuse<I> {
    #[inline(always)]
    fn as_inner(&self) -> &I {
        match self.iter {
            Some(ref iter) => iter,
            // SAFETY: the specialized iterator never sets `None`
            None => unsafe { intrinsics::unreachable() },
        }
    }

    #[inline(always)]
    fn as_inner_mut(&mut self) -> &mut I {
        match self.iter {
            Some(ref mut iter) => iter,
            // SAFETY: the specialized iterator never sets `None`
            None => unsafe { intrinsics::unreachable() },
        }
    }

    #[inline(always)]
    fn into_inner(self) -> I {
        match self.iter {
            Some(iter) => iter,
            // SAFETY: the specialized iterator never sets `None`
            None => unsafe { intrinsics::unreachable() },
        }
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> Iterator for Fuse<I>
where
    I: FusedIterator,
{
    #[inline]
    fn next(&mut self) -> Option<<I as Iterator>::Item> {
        self.as_inner_mut().next()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        self.as_inner_mut().nth(n)
    }

    #[inline]
    fn last(self) -> Option<I::Item> {
        self.into_inner().last()
    }

    #[inline]
    fn count(self) -> usize {
        self.into_inner().count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.as_inner().size_hint()
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        self.as_inner_mut().try_fold(init, fold)
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.into_inner().fold(init, fold)
    }

    #[inline]
    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        self.as_inner_mut().find(predicate)
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> DoubleEndedIterator for Fuse<I>
where
    I: DoubleEndedIterator + FusedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<<I as Iterator>::Item> {
        self.as_inner_mut().next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<<I as Iterator>::Item> {
        self.as_inner_mut().nth_back(n)
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        self.as_inner_mut().try_rfold(init, fold)
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.into_inner().rfold(init, fold)
    }

    #[inline]
    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        self.as_inner_mut().rfind(predicate)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Fuse<I>
where
    I: ExactSizeIterator + FusedIterator,
{
    fn len(&self) -> usize {
        self.as_inner().len()
    }

    fn is_empty(&self) -> bool {
        self.as_inner().is_empty()
    }
}

unsafe impl<I> TrustedRandomAccess for Fuse<I>
where
    I: TrustedRandomAccess,
{
    unsafe fn get_unchecked(&mut self, i: usize) -> I::Item {
        match self.iter {
            Some(ref mut iter) => iter.get_unchecked(i),
            // SAFETY: the caller asserts there is an item at `i`, so we're not exhausted.
            None => intrinsics::unreachable(),
        }
    }

    fn may_have_side_effect() -> bool {
        I::may_have_side_effect()
    }
}
