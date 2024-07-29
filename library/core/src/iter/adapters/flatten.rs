use crate::iter::adapters::SourceIter;
use crate::iter::{
    Cloned, Copied, Empty, Filter, FilterMap, Fuse, FusedIterator, InPlaceIterable, Map, Once,
    OnceWith, TrustedFused, TrustedLen,
};
use crate::num::NonZero;
use crate::ops::{ControlFlow, Try};
use crate::{array, fmt, option, result};

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

    pub(crate) fn into_parts(self) -> (Option<U::IntoIter>, Option<I>, Option<U::IntoIter>) {
        (
            self.inner.frontiter,
            self.inner.iter.into_inner().map(Map::into_inner),
            self.inner.backiter,
        )
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
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
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
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
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

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I, U, F> InPlaceIterable for FlatMap<I, U, F>
where
    I: InPlaceIterable,
    U: BoundedSize + IntoIterator,
{
    const EXPAND_BY: Option<NonZero<usize>> = const {
        match (I::EXPAND_BY, U::UPPER_BOUND) {
            (Some(m), Some(n)) => m.checked_mul(n),
            _ => None,
        }
    };
    const MERGE_BY: Option<NonZero<usize>> = I::MERGE_BY;
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I, U, F> SourceIter for FlatMap<I, U, F>
where
    I: SourceIter + TrustedFused,
    U: IntoIterator,
{
    type Source = I::Source;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut I::Source {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.inner.iter) }
    }
}

/// Marker trait for iterators/iterables which have a statically known upper
/// bound of the number of items they can produce.
///
/// # Safety
///
/// Implementations must not yield more elements than indicated by UPPER_BOUND if it is `Some`.
/// Used in specializations.  Implementations must not be conditional on lifetimes or
/// user-implementable traits.
#[rustc_specialization_trait]
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe trait BoundedSize {
    const UPPER_BOUND: Option<NonZero<usize>> = NonZero::new(1);
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<T> BoundedSize for Option<T> {}
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<T> BoundedSize for option::IntoIter<T> {}
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<T, U> BoundedSize for Result<T, U> {}
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<T> BoundedSize for result::IntoIter<T> {}
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<T> BoundedSize for Once<T> {}
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<T> BoundedSize for OnceWith<T> {}
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<T, const N: usize> BoundedSize for [T; N] {
    const UPPER_BOUND: Option<NonZero<usize>> = NonZero::new(N);
}
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<T, const N: usize> BoundedSize for array::IntoIter<T, N> {
    const UPPER_BOUND: Option<NonZero<usize>> = NonZero::new(N);
}
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I: BoundedSize, P> BoundedSize for Filter<I, P> {
    const UPPER_BOUND: Option<NonZero<usize>> = I::UPPER_BOUND;
}
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I: BoundedSize, P> BoundedSize for FilterMap<I, P> {
    const UPPER_BOUND: Option<NonZero<usize>> = I::UPPER_BOUND;
}
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I: BoundedSize, F> BoundedSize for Map<I, F> {
    const UPPER_BOUND: Option<NonZero<usize>> = I::UPPER_BOUND;
}
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I: BoundedSize> BoundedSize for Copied<I> {
    const UPPER_BOUND: Option<NonZero<usize>> = I::UPPER_BOUND;
}
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I: BoundedSize> BoundedSize for Cloned<I> {
    const UPPER_BOUND: Option<NonZero<usize>> = I::UPPER_BOUND;
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
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
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
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
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

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I> InPlaceIterable for Flatten<I>
where
    I: InPlaceIterable + Iterator,
    <I as Iterator>::Item: IntoIterator + BoundedSize,
{
    const EXPAND_BY: Option<NonZero<usize>> = const {
        match (I::EXPAND_BY, I::Item::UPPER_BOUND) {
            (Some(m), Some(n)) => m.checked_mul(n),
            _ => None,
        }
    };
    const MERGE_BY: Option<NonZero<usize>> = I::MERGE_BY;
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I> SourceIter for Flatten<I>
where
    I: SourceIter + TrustedFused + Iterator,
    <I as Iterator>::Item: IntoIterator,
{
    type Source = I::Source;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut I::Source {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.inner.iter) }
    }
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

// See also the `OneShot` specialization below.
impl<I, U> Iterator for FlattenCompat<I, U>
where
    I: Iterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: Iterator,
{
    type Item = U::Item;

    #[inline]
    default fn next(&mut self) -> Option<U::Item> {
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
    default fn size_hint(&self) -> (usize, Option<usize>) {
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
    default fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
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
    default fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
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
    default fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        #[inline]
        #[rustc_inherit_overflow_checks]
        fn advance<U: Iterator>(n: usize, iter: &mut U) -> ControlFlow<(), usize> {
            match iter.advance_by(n) {
                Ok(()) => ControlFlow::Break(()),
                Err(remaining) => ControlFlow::Continue(remaining.get()),
            }
        }

        match self.iter_try_fold(n, advance) {
            ControlFlow::Continue(remaining) => NonZero::new(remaining).map_or(Ok(()), Err),
            _ => Ok(()),
        }
    }

    #[inline]
    default fn count(self) -> usize {
        #[inline]
        #[rustc_inherit_overflow_checks]
        fn count<U: Iterator>(acc: usize, iter: U) -> usize {
            acc + iter.count()
        }

        self.iter_fold(0, count)
    }

    #[inline]
    default fn last(self) -> Option<Self::Item> {
        #[inline]
        fn last<U: Iterator>(last: Option<U::Item>, iter: U) -> Option<U::Item> {
            iter.last().or(last)
        }

        self.iter_fold(None, last)
    }
}

// See also the `OneShot` specialization below.
impl<I, U> DoubleEndedIterator for FlattenCompat<I, U>
where
    I: DoubleEndedIterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: DoubleEndedIterator,
{
    #[inline]
    default fn next_back(&mut self) -> Option<U::Item> {
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
    default fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
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
    default fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
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
    default fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        #[inline]
        #[rustc_inherit_overflow_checks]
        fn advance<U: DoubleEndedIterator>(n: usize, iter: &mut U) -> ControlFlow<(), usize> {
            match iter.advance_back_by(n) {
                Ok(()) => ControlFlow::Break(()),
                Err(remaining) => ControlFlow::Continue(remaining.get()),
            }
        }

        match self.iter_try_rfold(n, advance) {
            ControlFlow::Continue(remaining) => NonZero::new(remaining).map_or(Ok(()), Err),
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

/// Specialization trait for iterator types that never return more than one item.
///
/// Note that we still have to deal with the possibility that the iterator was
/// already exhausted before it came into our control.
#[rustc_specialization_trait]
trait OneShot {}

// These all have exactly one item, if not already consumed.
impl<T> OneShot for Once<T> {}
impl<F> OneShot for OnceWith<F> {}
impl<T> OneShot for array::IntoIter<T, 1> {}
impl<T> OneShot for option::IntoIter<T> {}
impl<T> OneShot for option::Iter<'_, T> {}
impl<T> OneShot for option::IterMut<'_, T> {}
impl<T> OneShot for result::IntoIter<T> {}
impl<T> OneShot for result::Iter<'_, T> {}
impl<T> OneShot for result::IterMut<'_, T> {}

// These are always empty, which is fine to optimize too.
impl<T> OneShot for Empty<T> {}
impl<T> OneShot for array::IntoIter<T, 0> {}

// These adaptors never increase the number of items.
// (There are more possible, but for now this matches BoundedSize above.)
impl<I: OneShot> OneShot for Cloned<I> {}
impl<I: OneShot> OneShot for Copied<I> {}
impl<I: OneShot, P> OneShot for Filter<I, P> {}
impl<I: OneShot, P> OneShot for FilterMap<I, P> {}
impl<I: OneShot, F> OneShot for Map<I, F> {}

// Blanket impls pass this property through as well
// (but we can't do `Box<I>` unless we expose this trait to alloc)
impl<I: OneShot> OneShot for &mut I {}

#[inline]
fn into_item<I>(inner: I) -> Option<I::Item>
where
    I: IntoIterator<IntoIter: OneShot>,
{
    inner.into_iter().next()
}

#[inline]
fn flatten_one<I: IntoIterator<IntoIter: OneShot>, Acc>(
    mut fold: impl FnMut(Acc, I::Item) -> Acc,
) -> impl FnMut(Acc, I) -> Acc {
    move |acc, inner| match inner.into_iter().next() {
        Some(item) => fold(acc, item),
        None => acc,
    }
}

#[inline]
fn try_flatten_one<I: IntoIterator<IntoIter: OneShot>, Acc, R: Try<Output = Acc>>(
    mut fold: impl FnMut(Acc, I::Item) -> R,
) -> impl FnMut(Acc, I) -> R {
    move |acc, inner| match inner.into_iter().next() {
        Some(item) => fold(acc, item),
        None => try { acc },
    }
}

#[inline]
fn advance_by_one<I>(n: NonZero<usize>, inner: I) -> Option<NonZero<usize>>
where
    I: IntoIterator<IntoIter: OneShot>,
{
    match inner.into_iter().next() {
        Some(_) => NonZero::new(n.get() - 1),
        None => Some(n),
    }
}

// Specialization: When the inner iterator `U` never returns more than one item, the `frontiter` and
// `backiter` states are a waste, because they'll always have already consumed their item. So in
// this impl, we completely ignore them and just focus on `self.iter`, and we only call the inner
// `U::next()` one time.
//
// It's mostly fine if we accidentally mix this with the more generic impls, e.g. by forgetting to
// specialize one of the methods. If the other impl did set the front or back, we wouldn't see it
// here, but it would be empty anyway; and if the other impl looked for a front or back that we
// didn't bother setting, it would just see `None` (or a previous empty) and move on.
//
// An exception to that is `advance_by(0)` and `advance_back_by(0)`, where the generic impls may set
// `frontiter` or `backiter` without consuming the item, so we **must** override those.
impl<I, U> Iterator for FlattenCompat<I, U>
where
    I: Iterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: Iterator + OneShot,
{
    #[inline]
    fn next(&mut self) -> Option<U::Item> {
        while let Some(inner) = self.iter.next() {
            if let item @ Some(_) = inner.into_iter().next() {
                return item;
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        match <I::Item as ConstSizeIntoIterator>::size() {
            Some(0) => (0, Some(0)),
            Some(1) => (lower, upper),
            _ => (0, upper),
        }
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        self.iter.try_fold(init, try_flatten_one(fold))
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.iter.fold(init, flatten_one(fold))
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        if let Some(n) = NonZero::new(n) {
            self.iter.try_fold(n, advance_by_one).map_or(Ok(()), Err)
        } else {
            // Just advance the outer iterator
            self.iter.advance_by(0)
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.filter_map(into_item).count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.iter.filter_map(into_item).last()
    }
}

// Note: We don't actually care about `U: DoubleEndedIterator`, since forward and backward are the
// same for a one-shot iterator, but we have to keep that to match the default specialization.
impl<I, U> DoubleEndedIterator for FlattenCompat<I, U>
where
    I: DoubleEndedIterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: DoubleEndedIterator + OneShot,
{
    #[inline]
    fn next_back(&mut self) -> Option<U::Item> {
        while let Some(inner) = self.iter.next_back() {
            if let item @ Some(_) = inner.into_iter().next() {
                return item;
            }
        }
        None
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        self.iter.try_rfold(init, try_flatten_one(fold))
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.iter.rfold(init, flatten_one(fold))
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        if let Some(n) = NonZero::new(n) {
            self.iter.try_rfold(n, advance_by_one).map_or(Ok(()), Err)
        } else {
            // Just advance the outer iterator
            self.iter.advance_back_by(0)
        }
    }
}
