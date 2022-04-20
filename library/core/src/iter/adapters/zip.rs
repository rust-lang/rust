use crate::cmp;
use crate::fmt::{self, Debug};
use crate::iter::traits::trusted_random_access::try_get_unchecked;
use crate::iter::{DoubleEndedIterator, ExactSizeIterator, FusedIterator, Iterator};
use crate::iter::{
    InPlaceIterable, SourceIter, TrustedLen, TrustedRandomAccess, TrustedRandomAccessNeedsCleanup,
    TrustedRandomAccessNeedsForwardSetup, TrustedRandomAccessNeedsReverseSetup,
};
use crate::ops::Try;

/// An iterator that iterates two other iterators simultaneously.
///
/// This `struct` is created by [`zip`] or [`Iterator::zip`].
/// See their documentation for more.
#[derive(Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Zip<A, B> {
    a: A,
    b: B,
}

impl<A: Iterator, B: Iterator> Zip<A, B> {
    pub(in crate::iter) fn new(a: A, b: B) -> Zip<A, B> {
        Zip { a, b }
    }
}

/// Converts the arguments to iterators and zips them.
///
/// See the documentation of [`Iterator::zip`] for more.
///
/// # Examples
///
/// ```
/// use std::iter::zip;
///
/// let xs = [1, 2, 3];
/// let ys = [4, 5, 6];
///
/// let mut iter = zip(xs, ys);
///
/// assert_eq!(iter.next().unwrap(), (1, 4));
/// assert_eq!(iter.next().unwrap(), (2, 5));
/// assert_eq!(iter.next().unwrap(), (3, 6));
/// assert!(iter.next().is_none());
///
/// // Nested zips are also possible:
/// let zs = [7, 8, 9];
///
/// let mut iter = zip(zip(xs, ys), zs);
///
/// assert_eq!(iter.next().unwrap(), ((1, 4), 7));
/// assert_eq!(iter.next().unwrap(), ((2, 5), 8));
/// assert_eq!(iter.next().unwrap(), ((3, 6), 9));
/// assert!(iter.next().is_none());
/// ```
#[stable(feature = "iter_zip", since = "1.59.0")]
pub fn zip<A, B>(a: A, b: B) -> Zip<A::IntoIter, B::IntoIter>
where
    A: IntoIterator,
    B: IntoIterator,
{
    Zip::new(a.into_iter(), b.into_iter())
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> Iterator for Zip<A, B>
where
    A: Iterator,
    B: Iterator,
{
    type Item = (A::Item, B::Item);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let x = self.a.next()?;
        let y = self.b.next()?;
        Some((x, y))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.spec_size_hint()
    }

    fn fold<T, F>(self, init: T, f: F) -> T
    where
        Self: Sized,
        F: FnMut(T, Self::Item) -> T,
    {
        self.spec_fold(init, f)
    }

    #[inline]
    #[doc(hidden)]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item
    where
        Self: TrustedRandomAccess,
    {
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        unsafe { (try_get_unchecked(&mut self.a, idx), try_get_unchecked(&mut self.b, idx)) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> DoubleEndedIterator for Zip<A, B>
where
    A: DoubleEndedIterator + ExactSizeIterator,
    B: DoubleEndedIterator + ExactSizeIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<(A::Item, B::Item)> {
        let a_sz = self.a.len();
        let b_sz = self.b.len();
        if a_sz != b_sz {
            // Adjust a, b to equal length
            if a_sz > b_sz {
                for _ in 0..a_sz - b_sz {
                    self.a.next_back();
                }
            } else {
                for _ in 0..b_sz - a_sz {
                    self.b.next_back();
                }
            }
        }
        match (self.a.next_back(), self.b.next_back()) {
            (Some(x), Some(y)) => Some((x, y)),
            (None, None) => None,
            _ => unreachable!(),
        }
    }
}

trait ZipSpec: Iterator {
    fn spec_fold<T, F>(self, init: T, f: F) -> T
    where
        Self: Sized,
        F: FnMut(T, Self::Item) -> T;

    fn spec_try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>;

    fn spec_size_hint(&self) -> (usize, Option<usize>);
}

impl<A, B> ZipSpec for Zip<A, B>
where
    A: Iterator,
    B: Iterator,
{
    default fn spec_fold<T, F>(mut self, init: T, mut f: F) -> T
    where
        Self: Sized,
        F: FnMut(T, Self::Item) -> T,
    {
        let mut accum = init;
        while let Some(x) = self.next() {
            accum = f(accum, x);
        }
        accum
    }

    default fn spec_try_fold<T, F, R>(&mut self, init: T, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(T, Self::Item) -> R,
        R: Try<Output = T>,
    {
        let mut accum = init;
        while let Some(x) = self.next() {
            accum = f(accum, x)?;
        }
        try { accum }
    }

    #[inline]
    default fn spec_size_hint(&self) -> (usize, Option<usize>) {
        let (a_lower, a_upper) = self.a.size_hint();
        let (b_lower, b_upper) = self.b.size_hint();

        let lower = cmp::min(a_lower, b_lower);

        let upper = match (a_upper, b_upper) {
            (Some(x), Some(y)) => Some(cmp::min(x, y)),
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(y),
            (None, None) => None,
        };

        (lower, upper)
    }
}

impl<A, B> ZipSpec for Zip<A, B>
where
    A: Iterator,
    B: Iterator,
    Self: TrustedRandomAccess,
{
    fn spec_fold<T, F>(mut self, init: T, mut f: F) -> T
    where
        Self: Sized,
        F: FnMut(T, Self::Item) -> T,
    {
        let _ = self.advance_by(0);
        let len = self.size();
        let mut accum = init;
        for i in 0..len {
            // SAFETY: each item is only accessed once and we run the cleanup function afterwards
            let x = unsafe { self.__iterator_get_unchecked(i) };
            accum = f(accum, x);
        }
        // FIXME drop-guard or use ForLoopDesugar
        self.cleanup_front(len);
        accum
    }

    fn spec_try_fold<T, F, R>(&mut self, init: T, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(T, Self::Item) -> R,
        R: Try<Output = T>,
    {
        let _ = self.advance_by(0);
        let len = self.size();
        let mut accum = init;
        for i in 0..len {
            // SAFETY: each item is only accessed once and we run the cleanup function afterwards
            let x = unsafe { self.__iterator_get_unchecked(i) };
            accum = f(accum, x)?;
        }
        // FIXME drop-guard or use ForLoopDesugar
        self.cleanup_front(len);
        try { accum }
    }

    #[inline]
    fn spec_size_hint(&self) -> (usize, Option<usize>) {
        let size = cmp::min(self.a.size_hint().0, self.b.size_hint().0);
        (size, Some(size))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> ExactSizeIterator for Zip<A, B>
where
    A: ExactSizeIterator,
    B: ExactSizeIterator,
{
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<A, B> TrustedRandomAccess for Zip<A, B>
where
    A: TrustedRandomAccess,
    B: TrustedRandomAccess,
{
    fn cleanup_front(&mut self, num: usize) {
        self.a.cleanup_front(num);
        self.b.cleanup_front(num);
    }

    fn cleanup_back(&mut self, num: usize) {
        self.a.cleanup_back(num);
        self.b.cleanup_back(num);
    }
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<A, B> TrustedRandomAccessNeedsCleanup for Zip<A, B> where
    A: TrustedRandomAccessNeedsCleanup
{
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<A, B> TrustedRandomAccessNeedsCleanup for Zip<A, B> where
    B: TrustedRandomAccessNeedsCleanup
{
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<A, B> TrustedRandomAccessNeedsCleanup for Zip<A, B>
where
    A: TrustedRandomAccessNeedsCleanup,
    B: TrustedRandomAccessNeedsCleanup,
{
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<A, B> TrustedRandomAccessNeedsForwardSetup for Zip<A, B> where
    A: TrustedRandomAccessNeedsForwardSetup
{
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<A, B> TrustedRandomAccessNeedsForwardSetup for Zip<A, B> where
    B: TrustedRandomAccessNeedsForwardSetup
{
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<A, B> TrustedRandomAccessNeedsForwardSetup for Zip<A, B>
where
    A: TrustedRandomAccessNeedsForwardSetup,
    B: TrustedRandomAccessNeedsForwardSetup,
{
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<A, B> TrustedRandomAccessNeedsReverseSetup for Zip<A, B> where Self: TrustedRandomAccess {}

#[stable(feature = "fused", since = "1.26.0")]
impl<A, B> FusedIterator for Zip<A, B>
where
    A: FusedIterator,
    B: FusedIterator,
{
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A, B> TrustedLen for Zip<A, B>
where
    A: TrustedLen,
    B: TrustedLen,
{
}

// Arbitrarily selects the left side of the zip iteration as extractable "source"
// it would require negative trait bounds to be able to try both
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<A, B> SourceIter for Zip<A, B>
where
    A: SourceIter,
{
    type Source = A::Source;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut A::Source {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.a) }
    }
}

// Since SourceIter forwards the left hand side we do the same here
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<A: InPlaceIterable, B: Iterator> InPlaceIterable for Zip<A, B> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Debug, B: Debug> Debug for Zip<A, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ZipFmt::fmt(self, f)
    }
}

trait ZipFmt<A, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl<A: Debug, B: Debug> ZipFmt<A, B> for Zip<A, B> {
    default fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Zip").field("a", &self.a).field("b", &self.b).finish()
    }
}

impl<A: Debug + TrustedRandomAccess, B: Debug + TrustedRandomAccess> ZipFmt<A, B> for Zip<A, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // It's *not safe* to call fmt on the contained iterators, since once
        // we start iterating they're in strange, potentially unsafe, states.
        f.debug_struct("Zip").finish()
    }
}
