use crate::fmt;
use crate::ops::Try;

use super::super::{Iterator, DoubleEndedIterator, FusedIterator};
use super::Map;

/// An iterator that maps each element to an iterator, and yields the elements
/// of the produced iterators.
///
/// This `struct` is created by the [`flat_map`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`flat_map`]: trait.Iterator.html#method.flat_map
/// [`Iterator`]: trait.Iterator.html
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct FlatMap<I, U: IntoIterator, F> {
    inner: FlattenCompat<Map<I, F>, <U as IntoIterator>::IntoIter>
}
impl<I: Iterator, U: IntoIterator, F: FnMut(I::Item) -> U> FlatMap<I, U, F> {
    pub(in super::super) fn new(iter: I, f: F) -> FlatMap<I, U, F> {
        FlatMap { inner: FlattenCompat::new(iter.map(f)) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Clone, U: Clone + IntoIterator, F: Clone> Clone for FlatMap<I, U, F>
    where <U as IntoIterator>::IntoIter: Clone
{
    fn clone(&self) -> Self { FlatMap { inner: self.inner.clone() } }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<I: fmt::Debug, U: IntoIterator, F> fmt::Debug for FlatMap<I, U, F>
    where U::IntoIter: fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlatMap").field("inner", &self.inner).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator, U: IntoIterator, F> Iterator for FlatMap<I, U, F>
    where F: FnMut(I::Item) -> U,
{
    type Item = U::Item;

    #[inline]
    fn next(&mut self) -> Option<U::Item> { self.inner.next() }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R where
        Self: Sized, Fold: FnMut(Acc, Self::Item) -> R, R: Try<Ok=Acc>
    {
        self.inner.try_fold(init, fold)
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
        where Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.fold(init, fold)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: DoubleEndedIterator, U, F> DoubleEndedIterator for FlatMap<I, U, F>
    where F: FnMut(I::Item) -> U,
          U: IntoIterator,
          U::IntoIter: DoubleEndedIterator
{
    #[inline]
    fn next_back(&mut self) -> Option<U::Item> { self.inner.next_back() }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R where
        Self: Sized, Fold: FnMut(Acc, Self::Item) -> R, R: Try<Ok=Acc>
    {
        self.inner.try_rfold(init, fold)
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
        where Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.rfold(init, fold)
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I, U, F> FusedIterator for FlatMap<I, U, F>
    where I: FusedIterator, U: IntoIterator, F: FnMut(I::Item) -> U {}

/// An iterator that flattens one level of nesting in an iterator of things
/// that can be turned into iterators.
///
/// This `struct` is created by the [`flatten`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`flatten`]: trait.Iterator.html#method.flatten
/// [`Iterator`]: trait.Iterator.html
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "iterator_flatten", since = "1.29.0")]
pub struct Flatten<I: Iterator>
where I::Item: IntoIterator {
    inner: FlattenCompat<I, <I::Item as IntoIterator>::IntoIter>,
}
impl<I: Iterator> Flatten<I>
where I::Item: IntoIterator {
    pub(in super::super) fn new(iter: I) -> Flatten<I> {
        Flatten { inner: FlattenCompat::new(iter) }
    }
}

#[stable(feature = "iterator_flatten", since = "1.29.0")]
impl<I, U> fmt::Debug for Flatten<I>
    where I: Iterator + fmt::Debug, U: Iterator + fmt::Debug,
          I::Item: IntoIterator<IntoIter = U, Item = U::Item>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Flatten").field("inner", &self.inner).finish()
    }
}

#[stable(feature = "iterator_flatten", since = "1.29.0")]
impl<I, U> Clone for Flatten<I>
    where I: Iterator + Clone, U: Iterator + Clone,
          I::Item: IntoIterator<IntoIter = U, Item = U::Item>,
{
    fn clone(&self) -> Self { Flatten { inner: self.inner.clone() } }
}

#[stable(feature = "iterator_flatten", since = "1.29.0")]
impl<I, U> Iterator for Flatten<I>
    where I: Iterator, U: Iterator,
          I::Item: IntoIterator<IntoIter = U, Item = U::Item>
{
    type Item = U::Item;

    #[inline]
    fn next(&mut self) -> Option<U::Item> { self.inner.next() }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R where
        Self: Sized, Fold: FnMut(Acc, Self::Item) -> R, R: Try<Ok=Acc>
    {
        self.inner.try_fold(init, fold)
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
        where Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.fold(init, fold)
    }
}

#[stable(feature = "iterator_flatten", since = "1.29.0")]
impl<I, U> DoubleEndedIterator for Flatten<I>
    where I: DoubleEndedIterator, U: DoubleEndedIterator,
          I::Item: IntoIterator<IntoIter = U, Item = U::Item>
{
    #[inline]
    fn next_back(&mut self) -> Option<U::Item> { self.inner.next_back() }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R where
        Self: Sized, Fold: FnMut(Acc, Self::Item) -> R, R: Try<Ok=Acc>
    {
        self.inner.try_rfold(init, fold)
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
        where Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.rfold(init, fold)
    }
}

#[stable(feature = "iterator_flatten", since = "1.29.0")]
impl<I, U> FusedIterator for Flatten<I>
    where I: FusedIterator, U: Iterator,
          I::Item: IntoIterator<IntoIter = U, Item = U::Item> {}

/// Real logic of both `Flatten` and `FlatMap` which simply delegate to
/// this type.
#[derive(Clone, Debug)]
struct FlattenCompat<I, U> {
    iter: I,
    frontiter: Option<U>,
    backiter: Option<U>,
}
impl<I, U> FlattenCompat<I, U> {
    /// Adapts an iterator by flattening it, for use in `flatten()` and `flat_map()`.
    fn new(iter: I) -> FlattenCompat<I, U> {
        FlattenCompat { iter, frontiter: None, backiter: None }
    }
}

impl<I, U> Iterator for FlattenCompat<I, U>
    where I: Iterator, U: Iterator,
          I::Item: IntoIterator<IntoIter = U, Item = U::Item>
{
    type Item = U::Item;

    #[inline]
    fn next(&mut self) -> Option<U::Item> {
        loop {
            if let Some(ref mut inner) = self.frontiter {
                if let elt@Some(_) = inner.next() { return elt }
            }
            match self.iter.next() {
                None => return self.backiter.as_mut().and_then(|it| it.next()),
                Some(inner) => self.frontiter = Some(inner.into_iter()),
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (flo, fhi) = self.frontiter.as_ref().map_or((0, Some(0)), |it| it.size_hint());
        let (blo, bhi) = self.backiter.as_ref().map_or((0, Some(0)), |it| it.size_hint());
        let lo = flo.saturating_add(blo);
        match (self.iter.size_hint(), fhi, bhi) {
            ((0, Some(0)), Some(a), Some(b)) => (lo, a.checked_add(b)),
            _ => (lo, None)
        }
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, mut init: Acc, mut fold: Fold) -> R where
        Self: Sized, Fold: FnMut(Acc, Self::Item) -> R, R: Try<Ok=Acc>
    {
        if let Some(ref mut front) = self.frontiter {
            init = front.try_fold(init, &mut fold)?;
        }
        self.frontiter = None;

        {
            let frontiter = &mut self.frontiter;
            init = self.iter.try_fold(init, |acc, x| {
                let mut mid = x.into_iter();
                let r = mid.try_fold(acc, &mut fold);
                *frontiter = Some(mid);
                r
            })?;
        }
        self.frontiter = None;

        if let Some(ref mut back) = self.backiter {
            init = back.try_fold(init, &mut fold)?;
        }
        self.backiter = None;

        Try::from_ok(init)
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, mut fold: Fold) -> Acc
        where Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.frontiter.into_iter()
            .chain(self.iter.map(IntoIterator::into_iter))
            .chain(self.backiter)
            .fold(init, |acc, iter| iter.fold(acc, &mut fold))
    }
}

impl<I, U> DoubleEndedIterator for FlattenCompat<I, U>
    where I: DoubleEndedIterator, U: DoubleEndedIterator,
          I::Item: IntoIterator<IntoIter = U, Item = U::Item>
{
    #[inline]
    fn next_back(&mut self) -> Option<U::Item> {
        loop {
            if let Some(ref mut inner) = self.backiter {
                if let elt@Some(_) = inner.next_back() { return elt }
            }
            match self.iter.next_back() {
                None => return self.frontiter.as_mut().and_then(|it| it.next_back()),
                next => self.backiter = next.map(IntoIterator::into_iter),
            }
        }
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, mut init: Acc, mut fold: Fold) -> R where
        Self: Sized, Fold: FnMut(Acc, Self::Item) -> R, R: Try<Ok=Acc>
    {
        if let Some(ref mut back) = self.backiter {
            init = back.try_rfold(init, &mut fold)?;
        }
        self.backiter = None;

        {
            let backiter = &mut self.backiter;
            init = self.iter.try_rfold(init, |acc, x| {
                let mut mid = x.into_iter();
                let r = mid.try_rfold(acc, &mut fold);
                *backiter = Some(mid);
                r
            })?;
        }
        self.backiter = None;

        if let Some(ref mut front) = self.frontiter {
            init = front.try_rfold(init, &mut fold)?;
        }
        self.frontiter = None;

        Try::from_ok(init)
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, mut fold: Fold) -> Acc
        where Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.frontiter.into_iter()
            .chain(self.iter.map(IntoIterator::into_iter))
            .chain(self.backiter)
            .rfold(init, |acc, iter| iter.rfold(acc, &mut fold))
    }
}
