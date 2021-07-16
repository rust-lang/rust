use crate::fmt;
use crate::iter::{DoubleEndedIterator, Fuse, FusedIterator, Iterator, Map, TrustedLen};
use crate::ops::Try;

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
unsafe impl<T, I, F, const N: usize> TrustedLen for FlatMap<I, [T; N], F>
where
    I: TrustedLen,
    F: FnMut(I::Item) -> [T; N],
{
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<'a, T, I, F, const N: usize> TrustedLen for FlatMap<I, &'a [T; N], F>
where
    I: TrustedLen,
    F: FnMut(I::Item) -> &'a [T; N],
{
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<'a, T, I, F, const N: usize> TrustedLen for FlatMap<I, &'a mut [T; N], F>
where
    I: TrustedLen,
    F: FnMut(I::Item) -> &'a mut [T; N],
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
    I: TrustedLen,
    <I as Iterator>::Item: TrustedConstSize,
{
}

/// Real logic of both `Flatten` and `FlatMap` which simply delegate to
/// this type.
#[derive(Clone, Debug)]
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

impl<I, U> Iterator for FlattenCompat<I, U>
where
    I: Iterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: Iterator,
{
    type Item = U::Item;

    #[inline]
    fn next(&mut self) -> Option<U::Item> {
        loop {
            if let Some(ref mut inner) = self.frontiter {
                match inner.next() {
                    None => self.frontiter = None,
                    elt @ Some(_) => return elt,
                }
            }
            match self.iter.next() {
                None => match self.backiter.as_mut()?.next() {
                    None => {
                        self.backiter = None;
                        return None;
                    }
                    elt @ Some(_) => return elt,
                },
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
            let upper = upper.and_then(|i| i.checked_mul(fixed_size));
            let upper = fhi
                .zip_with(bhi, usize::checked_add)
                .flatten()
                .zip_with(upper, usize::checked_add)
                .flatten();

            return (lower, upper);
        }

        match (self.iter.size_hint(), fhi, bhi) {
            ((0, Some(0)), Some(a), Some(b)) => (lo, a.checked_add(b)),
            _ => (lo, None),
        }
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, mut init: Acc, mut fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        #[inline]
        fn flatten<'a, T: IntoIterator, Acc, R: Try<Output = Acc>>(
            frontiter: &'a mut Option<T::IntoIter>,
            fold: &'a mut impl FnMut(Acc, T::Item) -> R,
        ) -> impl FnMut(Acc, T) -> R + 'a {
            move |acc, x| {
                let mut mid = x.into_iter();
                let r = mid.try_fold(acc, &mut *fold);
                *frontiter = Some(mid);
                r
            }
        }

        if let Some(ref mut front) = self.frontiter {
            init = front.try_fold(init, &mut fold)?;
        }
        self.frontiter = None;

        init = self.iter.try_fold(init, flatten(&mut self.frontiter, &mut fold))?;
        self.frontiter = None;

        if let Some(ref mut back) = self.backiter {
            init = back.try_fold(init, &mut fold)?;
        }
        self.backiter = None;

        try { init }
    }

    #[inline]
    fn fold<Acc, Fold>(self, mut init: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn flatten<T: IntoIterator, Acc>(
            fold: &mut impl FnMut(Acc, T::Item) -> Acc,
        ) -> impl FnMut(Acc, T) -> Acc + '_ {
            move |acc, x| x.into_iter().fold(acc, &mut *fold)
        }

        if let Some(front) = self.frontiter {
            init = front.fold(init, &mut fold);
        }

        init = self.iter.fold(init, flatten(&mut fold));

        if let Some(back) = self.backiter {
            init = back.fold(init, &mut fold);
        }

        init
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
            if let Some(ref mut inner) = self.backiter {
                match inner.next_back() {
                    None => self.backiter = None,
                    elt @ Some(_) => return elt,
                }
            }
            match self.iter.next_back() {
                None => match self.frontiter.as_mut()?.next_back() {
                    None => {
                        self.frontiter = None;
                        return None;
                    }
                    elt @ Some(_) => return elt,
                },
                next => self.backiter = next.map(IntoIterator::into_iter),
            }
        }
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, mut init: Acc, mut fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        #[inline]
        fn flatten<'a, T: IntoIterator, Acc, R: Try<Output = Acc>>(
            backiter: &'a mut Option<T::IntoIter>,
            fold: &'a mut impl FnMut(Acc, T::Item) -> R,
        ) -> impl FnMut(Acc, T) -> R + 'a
        where
            T::IntoIter: DoubleEndedIterator,
        {
            move |acc, x| {
                let mut mid = x.into_iter();
                let r = mid.try_rfold(acc, &mut *fold);
                *backiter = Some(mid);
                r
            }
        }

        if let Some(ref mut back) = self.backiter {
            init = back.try_rfold(init, &mut fold)?;
        }
        self.backiter = None;

        init = self.iter.try_rfold(init, flatten(&mut self.backiter, &mut fold))?;
        self.backiter = None;

        if let Some(ref mut front) = self.frontiter {
            init = front.try_rfold(init, &mut fold)?;
        }
        self.frontiter = None;

        try { init }
    }

    #[inline]
    fn rfold<Acc, Fold>(self, mut init: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn flatten<T: IntoIterator, Acc>(
            fold: &mut impl FnMut(Acc, T::Item) -> Acc,
        ) -> impl FnMut(Acc, T) -> Acc + '_
        where
            T::IntoIter: DoubleEndedIterator,
        {
            move |acc, x| x.into_iter().rfold(acc, &mut *fold)
        }

        if let Some(back) = self.backiter {
            init = back.rfold(init, &mut fold);
        }

        init = self.iter.rfold(init, flatten(&mut fold));

        if let Some(front) = self.frontiter {
            init = front.rfold(init, &mut fold);
        }

        init
    }
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

#[doc(hidden)]
#[unstable(feature = "std_internals", issue = "none")]
// FIXME(#20400): Instead of this helper trait there should be multiple impl TrustedLen for Flatten<>
//   blocks with different bounds on Iterator::Item but the compiler erroneously considers them overlapping
pub unsafe trait TrustedConstSize: IntoIterator {}

#[unstable(feature = "std_internals", issue = "none")]
unsafe impl<T, const N: usize> TrustedConstSize for [T; N] {}
#[unstable(feature = "std_internals", issue = "none")]
unsafe impl<T, const N: usize> TrustedConstSize for &'_ [T; N] {}
#[unstable(feature = "std_internals", issue = "none")]
unsafe impl<T, const N: usize> TrustedConstSize for &'_ mut [T; N] {}
