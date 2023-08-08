use crate::fmt;
use crate::iter::{DoubleEndedIterator, Fuse, FusedIterator, Iterator, Map, TrustedLen};
use crate::num::NonZeroUsize;
use crate::ops::{ControlFlow, Try};

/// An iterator that maps each element to an iterator, and yields the elements
/// of the produced iterators.
///
/// This `struct` is created by [`Iterator::flat_map`]. See its documentation
/// for more.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct FlatMap<I, U: IntoIterator, F> {
    inner: FlattenCompat<Map<I, F>, <U as IntoIterator>::IntoIter>,
}

impl<I: Iterator, U: IntoIterator, F: FnMut(I::Item) -> U> FlatMap<I, U, F> {
    pub(in crate::iter) fn new(iter: I, f: F) -> FlatMap<I, U, F> {
        FlatMap { inner: FlattenCompat::new(iter.map(f)) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Clone, U, F: Clone> Clone for FlatMap<I, U, F>
where
    U: Clone + IntoIterator<IntoIter: Clone>,
{
    fn clone(&self) -> Self {
        FlatMap { inner: self.inner.clone() }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<I: fmt::Debug, U, F> fmt::Debug for FlatMap<I, U, F>
where
    U: IntoIterator<IntoIter: fmt::Debug>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlatMap").field("inner", &self.inner).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator, U: IntoIterator, F> Iterator for FlatMap<I, U, F>
where
    F: FnMut(I::Item) -> U,
{
    type Item = U::Item;

    #[inline]
    fn next(&mut self) -> Option<U::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        self.inner.try_fold(init, fold)
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.fold(init, fold)
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        self.inner.advance_by(n)
    }

    #[inline]
    fn count(self) -> usize {
        self.inner.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.inner.last()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: DoubleEndedIterator, U, F> DoubleEndedIterator for FlatMap<I, U, F>
where
    F: FnMut(I::Item) -> U,
    U: IntoIterator<IntoIter: DoubleEndedIterator>,
{
    #[inline]
    fn next_back(&mut self) -> Option<U::Item> {
        self.inner.next_back()
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        self.inner.try_rfold(init, fold)
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.rfold(init, fold)
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        self.inner.advance_back_by(n)
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I, U, F> FusedIterator for FlatMap<I, U, F>
where
    I: FusedIterator,
    U: IntoIterator,
    F: FnMut(I::Item) -> U,
{
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<I, U, F> TrustedLen for FlatMap<I, U, F>
where
    I: Iterator,
    U: IntoIterator,
    F: FnMut(I::Item) -> U,
    FlattenCompat<Map<I, F>, <U as IntoIterator>::IntoIter>: TrustedLen,
{
}

/// An iterator that flattens one level of nesting in an iterator of things
/// that can be turned into iterators.
///
/// This `struct` is created by the [`flatten`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`flatten`]: Iterator::flatten()
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "iterator_flatten", since = "1.29.0")]
pub struct Flatten<I: Iterator<Item: IntoIterator>> {
    inner: FlattenCompat<I, <I::Item as IntoIterator>::IntoIter>,
}

impl<I: Iterator<Item: IntoIterator>> Flatten<I> {
    pub(in super::super) fn new(iter: I) -> Flatten<I> {
        Flatten { inner: FlattenCompat::new(iter) }
    }
}

#[stable(feature = "iterator_flatten", since = "1.29.0")]
impl<I, U> fmt::Debug for Flatten<I>
where
    I: fmt::Debug + Iterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: fmt::Debug + Iterator,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Flatten").field("inner", &self.inner).finish()
    }
}

#[stable(feature = "iterator_flatten", since = "1.29.0")]
impl<I, U> Clone for Flatten<I>
where
    I: Clone + Iterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: Clone + Iterator,
{
    fn clone(&self) -> Self {
        Flatten { inner: self.inner.clone() }
    }
}

#[stable(feature = "iterator_flatten", since = "1.29.0")]
impl<I, U> Iterator for Flatten<I>
where
    I: Iterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: Iterator,
{
    type Item = U::Item;

    #[inline]
    fn next(&mut self) -> Option<U::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        self.inner.try_fold(init, fold)
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.fold(init, fold)
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        self.inner.advance_by(n)
    }

    #[inline]
    fn count(self) -> usize {
        self.inner.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.inner.last()
    }
}

#[stable(feature = "iterator_flatten", since = "1.29.0")]
impl<I, U> DoubleEndedIterator for Flatten<I>
where
    I: DoubleEndedIterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<U::Item> {
        self.inner.next_back()
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        self.inner.try_rfold(init, fold)
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.rfold(init, fold)
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        self.inner.advance_back_by(n)
    }
}

#[stable(feature = "iterator_flatten", since = "1.29.0")]
impl<I, U> FusedIterator for Flatten<I>
where
    I: FusedIterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: Iterator,
{
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<I> TrustedLen for Flatten<I>
where
    I: Iterator<Item: IntoIterator>,
    FlattenCompat<I, <I::Item as IntoIterator>::IntoIter>: TrustedLen,
{
}

#[stable(feature = "default_iters", since = "1.70.0")]
impl<I> Default for Flatten<I>
where
    I: Default + Iterator<Item: IntoIterator>,
{
    /// Creates a `Flatten` iterator from the default value of `I`.
    ///
    /// ```
    /// # use core::slice;
    /// # use std::iter::Flatten;
    /// let iter: Flatten<slice::Iter<'_, [u8; 4]>> = Default::default();
    /// assert_eq!(iter.count(), 0);
    /// ```
    fn default() -> Self {
        Flatten::new(Default::default())
    }
}

/// Real logic of both `Flatten` and `FlatMap` which simply delegate to
/// this type.
#[derive(Clone, Debug)]
#[unstable(feature = "trusted_len", issue = "37572")]
struct FlattenCompat<I, U> {
    iter: Fuse<I>,
    frontiter: Option<U>,
    backiter: Option<U>,
}
impl<I, U> FlattenCompat<I, U>
where
    I: Iterator,
{
    /// Adapts an iterator by flattening it, for use in `flatten()` and `flat_map()`.
    fn new(iter: I) -> FlattenCompat<I, U> {
        FlattenCompat { iter: iter.fuse(), frontiter: None, backiter: None }
    }
}

impl<I, U> FlattenCompat<I, U>
where
    I: Iterator<Item: IntoIterator<IntoIter = U>>,
{
    /// Folds the inner iterators into an accumulator by applying an operation.
    ///
    /// Folds over the inner iterators, not over their elements. Is used by the `fold`, `count`,
    /// and `last` methods.
    #[inline]
    fn iter_fold<Acc, Fold>(self, mut acc: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, U) -> Acc,
    {
        #[inline]
        fn flatten<T: IntoIterator, Acc>(
            fold: &mut impl FnMut(Acc, T::IntoIter) -> Acc,
        ) -> impl FnMut(Acc, T) -> Acc + '_ {
            move |acc, iter| fold(acc, iter.into_iter())
        }

        if let Some(iter) = self.frontiter {
            acc = fold(acc, iter);
        }

        acc = self.iter.fold(acc, flatten(&mut fold));

        if let Some(iter) = self.backiter {
            acc = fold(acc, iter);
        }

        acc
    }

    /// Folds over the inner iterators as long as the given function returns successfully,
    /// always storing the most recent inner iterator in `self.frontiter`.
    ///
    /// Folds over the inner iterators, not over their elements. Is used by the `try_fold` and
    /// `advance_by` methods.
    #[inline]
    fn iter_try_fold<Acc, Fold, R>(&mut self, mut acc: Acc, mut fold: Fold) -> R
    where
        Fold: FnMut(Acc, &mut U) -> R,
        R: Try<Output = Acc>,
    {
        #[inline]
        fn flatten<'a, T: IntoIterator, Acc, R: Try<Output = Acc>>(
            frontiter: &'a mut Option<T::IntoIter>,
            fold: &'a mut impl FnMut(Acc, &mut T::IntoIter) -> R,
        ) -> impl FnMut(Acc, T) -> R + 'a {
            move |acc, iter| fold(acc, frontiter.insert(iter.into_iter()))
        }

        if let Some(iter) = &mut self.frontiter {
            acc = fold(acc, iter)?;
        }
        self.frontiter = None;

        acc = self.iter.try_fold(acc, flatten(&mut self.frontiter, &mut fold))?;
        self.frontiter = None;

        if let Some(iter) = &mut self.backiter {
            acc = fold(acc, iter)?;
        }
        self.backiter = None;

        try { acc }
    }
}

impl<I, U> FlattenCompat<I, U>
where
    I: DoubleEndedIterator<Item: IntoIterator<IntoIter = U>>,
{
    /// Folds the inner iterators into an accumulator by applying an operation, starting form the
    /// back.
    ///
    /// Folds over the inner iterators, not over their elements. Is used by the `rfold` method.
    #[inline]
    fn iter_rfold<Acc, Fold>(self, mut acc: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, U) -> Acc,
    {
        #[inline]
        fn flatten<T: IntoIterator, Acc>(
            fold: &mut impl FnMut(Acc, T::IntoIter) -> Acc,
        ) -> impl FnMut(Acc, T) -> Acc + '_ {
            move |acc, iter| fold(acc, iter.into_iter())
        }

        if let Some(iter) = self.backiter {
            acc = fold(acc, iter);
        }

        acc = self.iter.rfold(acc, flatten(&mut fold));

        if let Some(iter) = self.frontiter {
            acc = fold(acc, iter);
        }

        acc
    }

    /// Folds over the inner iterators in reverse order as long as the given function returns
    /// successfully, always storing the most recent inner iterator in `self.backiter`.
    ///
    /// Folds over the inner iterators, not over their elements. Is used by the `try_rfold` and
    /// `advance_back_by` methods.
    #[inline]
    fn iter_try_rfold<Acc, Fold, R>(&mut self, mut acc: Acc, mut fold: Fold) -> R
    where
        Fold: FnMut(Acc, &mut U) -> R,
        R: Try<Output = Acc>,
    {
        #[inline]
        fn flatten<'a, T: IntoIterator, Acc, R: Try>(
            backiter: &'a mut Option<T::IntoIter>,
            fold: &'a mut impl FnMut(Acc, &mut T::IntoIter) -> R,
        ) -> impl FnMut(Acc, T) -> R + 'a {
            move |acc, iter| fold(acc, backiter.insert(iter.into_iter()))
        }

        if let Some(iter) = &mut self.backiter {
            acc = fold(acc, iter)?;
        }
        self.backiter = None;

        acc = self.iter.try_rfold(acc, flatten(&mut self.backiter, &mut fold))?;
        self.backiter = None;

        if let Some(iter) = &mut self.frontiter {
            acc = fold(acc, iter)?;
        }
        self.frontiter = None;

        try { acc }
    }
}

impl<I, U> Iterator for FlattenCompat<I, U>
where
    I: Iterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: Iterator,
{
    type Item = U::Item;

    #[inline]
    fn next(&mut self) -> Option<U::Item> {
        loop {
            if let elt @ Some(_) = and_then_or_clear(&mut self.frontiter, Iterator::next) {
                return elt;
            }
            match self.iter.next() {
                None => return and_then_or_clear(&mut self.backiter, Iterator::next),
                Some(inner) => self.frontiter = Some(inner.into_iter()),
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (flo, fhi) = self.frontiter.as_ref().map_or((0, Some(0)), U::size_hint);
        let (blo, bhi) = self.backiter.as_ref().map_or((0, Some(0)), U::size_hint);
        let lo = flo.saturating_add(blo);

        if let Some(fixed_size) = <<I as Iterator>::Item as ConstSizeIntoIterator>::size() {
            let (lower, upper) = self.iter.size_hint();

            let lower = lower.saturating_mul(fixed_size).saturating_add(lo);
            let upper =
                try { fhi?.checked_add(bhi?)?.checked_add(fixed_size.checked_mul(upper?)?)? };

            return (lower, upper);
        }

        match (self.iter.size_hint(), fhi, bhi) {
            ((0, Some(0)), Some(a), Some(b)) => (lo, a.checked_add(b)),
            _ => (lo, None),
        }
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        #[inline]
        fn flatten<U: Iterator, Acc, R: Try<Output = Acc>>(
            mut fold: impl FnMut(Acc, U::Item) -> R,
        ) -> impl FnMut(Acc, &mut U) -> R {
            move |acc, iter| iter.try_fold(acc, &mut fold)
        }

        self.iter_try_fold(init, flatten(fold))
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn flatten<U: Iterator, Acc>(
            mut fold: impl FnMut(Acc, U::Item) -> Acc,
        ) -> impl FnMut(Acc, U) -> Acc {
            move |acc, iter| iter.fold(acc, &mut fold)
        }

        self.iter_fold(init, flatten(fold))
    }

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        #[inline]
        #[rustc_inherit_overflow_checks]
        fn advance<U: Iterator>(n: usize, iter: &mut U) -> ControlFlow<(), usize> {
            match iter.advance_by(n) {
                Ok(()) => ControlFlow::Break(()),
                Err(remaining) => ControlFlow::Continue(remaining.get()),
            }
        }

        match self.iter_try_fold(n, advance) {
            ControlFlow::Continue(remaining) => NonZeroUsize::new(remaining).map_or(Ok(()), Err),
            _ => Ok(()),
        }
    }

    #[inline]
    fn count(self) -> usize {
        #[inline]
        #[rustc_inherit_overflow_checks]
        fn count<U: Iterator>(acc: usize, iter: U) -> usize {
            acc + iter.count()
        }

        self.iter_fold(0, count)
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        #[inline]
        fn last<U: Iterator>(last: Option<U::Item>, iter: U) -> Option<U::Item> {
            iter.last().or(last)
        }

        self.iter_fold(None, last)
    }
}

impl<I, U> DoubleEndedIterator for FlattenCompat<I, U>
where
    I: DoubleEndedIterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<U::Item> {
        loop {
            if let elt @ Some(_) = and_then_or_clear(&mut self.backiter, |b| b.next_back()) {
                return elt;
            }
            match self.iter.next_back() {
                None => return and_then_or_clear(&mut self.frontiter, |f| f.next_back()),
                Some(inner) => self.backiter = Some(inner.into_iter()),
            }
        }
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        #[inline]
        fn flatten<U: DoubleEndedIterator, Acc, R: Try<Output = Acc>>(
            mut fold: impl FnMut(Acc, U::Item) -> R,
        ) -> impl FnMut(Acc, &mut U) -> R {
            move |acc, iter| iter.try_rfold(acc, &mut fold)
        }

        self.iter_try_rfold(init, flatten(fold))
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn flatten<U: DoubleEndedIterator, Acc>(
            mut fold: impl FnMut(Acc, U::Item) -> Acc,
        ) -> impl FnMut(Acc, U) -> Acc {
            move |acc, iter| iter.rfold(acc, &mut fold)
        }

        self.iter_rfold(init, flatten(fold))
    }

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        #[inline]
        #[rustc_inherit_overflow_checks]
        fn advance<U: DoubleEndedIterator>(n: usize, iter: &mut U) -> ControlFlow<(), usize> {
            match iter.advance_back_by(n) {
                Ok(()) => ControlFlow::Break(()),
                Err(remaining) => ControlFlow::Continue(remaining.get()),
            }
        }

        match self.iter_try_rfold(n, advance) {
            ControlFlow::Continue(remaining) => NonZeroUsize::new(remaining).map_or(Ok(()), Err),
            _ => Ok(()),
        }
    }
}

unsafe impl<const N: usize, I, T> TrustedLen
    for FlattenCompat<I, <[T; N] as IntoIterator>::IntoIter>
where
    I: TrustedLen<Item = [T; N]>,
{
}

unsafe impl<'a, const N: usize, I, T> TrustedLen
    for FlattenCompat<I, <&'a [T; N] as IntoIterator>::IntoIter>
where
    I: TrustedLen<Item = &'a [T; N]>,
{
}

unsafe impl<'a, const N: usize, I, T> TrustedLen
    for FlattenCompat<I, <&'a mut [T; N] as IntoIterator>::IntoIter>
where
    I: TrustedLen<Item = &'a mut [T; N]>,
{
}

trait ConstSizeIntoIterator: IntoIterator {
    // FIXME(#31844): convert to an associated const once specialization supports that
    fn size() -> Option<usize>;
}

impl<T> ConstSizeIntoIterator for T
where
    T: IntoIterator,
{
    #[inline]
    default fn size() -> Option<usize> {
        None
    }
}

impl<T, const N: usize> ConstSizeIntoIterator for [T; N] {
    #[inline]
    fn size() -> Option<usize> {
        Some(N)
    }
}

impl<T, const N: usize> ConstSizeIntoIterator for &[T; N] {
    #[inline]
    fn size() -> Option<usize> {
        Some(N)
    }
}

impl<T, const N: usize> ConstSizeIntoIterator for &mut [T; N] {
    #[inline]
    fn size() -> Option<usize> {
        Some(N)
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
