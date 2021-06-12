use crate::cmp;
use crate::fmt::{self, Debug};
use crate::iter::{DoubleEndedIterator, ExactSizeIterator, FusedIterator, Iterator};
use crate::iter::{InPlaceIterable, SourceIter, TrustedLen};

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
    // index, len and a_len are only used by the specialized version of zip
    index: usize,
    len: usize,
    a_len: usize,
}
impl<A: Iterator, B: Iterator> Zip<A, B> {
    pub(in crate::iter) fn new(a: A, b: B) -> Zip<A, B> {
        ZipImpl::new(a, b)
    }
    fn super_nth(&mut self, mut n: usize) -> Option<(A::Item, B::Item)> {
        while let Some(x) = Iterator::next(self) {
            if n == 0 {
                return Some(x);
            }
            n -= 1;
        }
        None
    }
}

/// Converts the arguments to iterators and zips them.
///
/// See the documentation of [`Iterator::zip`] for more.
///
/// # Examples
///
/// ```
/// #![feature(iter_zip)]
/// use std::iter::zip;
///
/// let xs = [1, 2, 3];
/// let ys = [4, 5, 6];
/// for (x, y) in zip(&xs, &ys) {
///     println!("x:{}, y:{}", x, y);
/// }
///
/// // Nested zips are also possible:
/// let zs = [7, 8, 9];
/// for ((x, y), z) in zip(zip(&xs, &ys), &zs) {
///     println!("x:{}, y:{}, z:{}", x, y, z);
/// }
/// ```
#[unstable(feature = "iter_zip", issue = "83574")]
pub fn zip<A, B>(a: A, b: B) -> Zip<A::IntoIter, B::IntoIter>
where
    A: IntoIterator,
    B: IntoIterator,
{
    ZipImpl::new(a.into_iter(), b.into_iter())
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
        ZipImpl::next(self)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        ZipImpl::size_hint(self)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        ZipImpl::nth(self, n)
    }

    #[inline]
    #[doc(hidden)]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item
    where
        Self: TrustedRandomAccess,
    {
        // SAFETY: `ZipImpl::__iterator_get_unchecked` has same safety
        // requirements as `Iterator::__iterator_get_unchecked`.
        unsafe { ZipImpl::get_unchecked(self, idx) }
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
        ZipImpl::next_back(self)
    }
}

// Zip specialization trait
#[doc(hidden)]
trait ZipImpl<A, B> {
    type Item;
    fn new(a: A, b: B) -> Self;
    fn next(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);
    fn nth(&mut self, n: usize) -> Option<Self::Item>;
    fn next_back(&mut self) -> Option<Self::Item>
    where
        A: DoubleEndedIterator + ExactSizeIterator,
        B: DoubleEndedIterator + ExactSizeIterator;
    // This has the same safety requirements as `Iterator::__iterator_get_unchecked`
    unsafe fn get_unchecked(&mut self, idx: usize) -> <Self as Iterator>::Item
    where
        Self: Iterator + TrustedRandomAccess;
}

// General Zip impl
#[doc(hidden)]
impl<A, B> ZipImpl<A, B> for Zip<A, B>
where
    A: Iterator,
    B: Iterator,
{
    type Item = (A::Item, B::Item);
    default fn new(a: A, b: B) -> Self {
        Zip {
            a,
            b,
            index: 0, // unused
            len: 0,   // unused
            a_len: 0, // unused
        }
    }

    #[inline]
    default fn next(&mut self) -> Option<(A::Item, B::Item)> {
        let x = self.a.next()?;
        let y = self.b.next()?;
        Some((x, y))
    }

    #[inline]
    default fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.super_nth(n)
    }

    #[inline]
    default fn next_back(&mut self) -> Option<(A::Item, B::Item)>
    where
        A: DoubleEndedIterator + ExactSizeIterator,
        B: DoubleEndedIterator + ExactSizeIterator,
    {
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

    #[inline]
    default fn size_hint(&self) -> (usize, Option<usize>) {
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

    default unsafe fn get_unchecked(&mut self, _idx: usize) -> <Self as Iterator>::Item
    where
        Self: TrustedRandomAccess,
    {
        unreachable!("Always specialized");
    }
}

#[doc(hidden)]
impl<A, B> ZipImpl<A, B> for Zip<A, B>
where
    A: TrustedRandomAccess + Iterator,
    B: TrustedRandomAccess + Iterator,
{
    fn new(a: A, b: B) -> Self {
        let a_len = a.size();
        let len = cmp::min(a_len, b.size());
        Zip { a, b, index: 0, len, a_len }
    }

    #[inline]
    fn next(&mut self) -> Option<(A::Item, B::Item)> {
        if self.index < self.len {
            let i = self.index;
            self.index += 1;
            // SAFETY: `i` is smaller than `self.len`, thus smaller than `self.a.len()` and `self.b.len()`
            unsafe {
                Some((self.a.__iterator_get_unchecked(i), self.b.__iterator_get_unchecked(i)))
            }
        } else if A::MAY_HAVE_SIDE_EFFECT && self.index < self.a_len {
            let i = self.index;
            self.index += 1;
            self.len += 1;
            // match the base implementation's potential side effects
            // SAFETY: we just checked that `i` < `self.a.len()`
            unsafe {
                self.a.__iterator_get_unchecked(i);
            }
            None
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len - self.index;
        (len, Some(len))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let delta = cmp::min(n, self.len - self.index);
        let end = self.index + delta;
        while self.index < end {
            let i = self.index;
            self.index += 1;
            if A::MAY_HAVE_SIDE_EFFECT {
                // SAFETY: the usage of `cmp::min` to calculate `delta`
                // ensures that `end` is smaller than or equal to `self.len`,
                // so `i` is also smaller than `self.len`.
                unsafe {
                    self.a.__iterator_get_unchecked(i);
                }
            }
            if B::MAY_HAVE_SIDE_EFFECT {
                // SAFETY: same as above.
                unsafe {
                    self.b.__iterator_get_unchecked(i);
                }
            }
        }

        self.super_nth(n - delta)
    }

    #[inline]
    fn next_back(&mut self) -> Option<(A::Item, B::Item)>
    where
        A: DoubleEndedIterator + ExactSizeIterator,
        B: DoubleEndedIterator + ExactSizeIterator,
    {
        if A::MAY_HAVE_SIDE_EFFECT || B::MAY_HAVE_SIDE_EFFECT {
            let sz_a = self.a.size();
            let sz_b = self.b.size();
            // Adjust a, b to equal length, make sure that only the first call
            // of `next_back` does this, otherwise we will break the restriction
            // on calls to `self.next_back()` after calling `get_unchecked()`.
            if sz_a != sz_b {
                let sz_a = self.a.size();
                if A::MAY_HAVE_SIDE_EFFECT && sz_a > self.len {
                    for _ in 0..sz_a - self.len {
                        self.a.next_back();
                    }
                    self.a_len = self.len;
                }
                let sz_b = self.b.size();
                if B::MAY_HAVE_SIDE_EFFECT && sz_b > self.len {
                    for _ in 0..sz_b - self.len {
                        self.b.next_back();
                    }
                }
            }
        }
        if self.index < self.len {
            self.len -= 1;
            self.a_len -= 1;
            let i = self.len;
            // SAFETY: `i` is smaller than the previous value of `self.len`,
            // which is also smaller than or equal to `self.a.len()` and `self.b.len()`
            unsafe {
                Some((self.a.__iterator_get_unchecked(i), self.b.__iterator_get_unchecked(i)))
            }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_unchecked(&mut self, idx: usize) -> <Self as Iterator>::Item {
        let idx = self.index + idx;
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        unsafe { (self.a.__iterator_get_unchecked(idx), self.b.__iterator_get_unchecked(idx)) }
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
    const MAY_HAVE_SIDE_EFFECT: bool = A::MAY_HAVE_SIDE_EFFECT || B::MAY_HAVE_SIDE_EFFECT;
}

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
unsafe impl<S, A, B> SourceIter for Zip<A, B>
where
    A: SourceIter<Source = S>,
    B: Iterator,
    S: Iterator,
{
    type Source = S;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut S {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.a) }
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
// Limited to Item: Copy since interaction between Zip's use of TrustedRandomAccess
// and Drop implementation of the source is unclear.
//
// An additional method returning the number of times the source has been logically advanced
// (without calling next()) would be needed to properly drop the remainder of the source.
unsafe impl<A: InPlaceIterable, B: Iterator> InPlaceIterable for Zip<A, B> where A::Item: Copy {}

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

/// An iterator whose items are random-accessible efficiently
///
/// # Safety
///
/// The iterator's `size_hint` must be exact and cheap to call.
///
/// `size` may not be overridden.
///
/// `<Self as Iterator>::__iterator_get_unchecked` must be safe to call
/// provided the following conditions are met.
///
/// 1. `0 <= idx` and `idx < self.size()`.
/// 2. If `self: !Clone`, then `get_unchecked` is never called with the same
///    index on `self` more than once.
/// 3. After `self.get_unchecked(idx)` has been called then `next_back` will
///    only be called at most `self.size() - idx - 1` times.
/// 4. After `get_unchecked` is called, then only the following methods will be
///    called on `self`:
///     * `std::clone::Clone::clone()`
///     * `std::iter::Iterator::size_hint()`
///     * `std::iter::DoubleEndedIterator::next_back()`
///     * `std::iter::Iterator::__iterator_get_unchecked()`
///     * `std::iter::TrustedRandomAccess::size()`
///
/// Further, given that these conditions are met, it must guarantee that:
///
/// * It does not change the value returned from `size_hint`
/// * It must be safe to call the methods listed above on `self` after calling
///   `get_unchecked`, assuming that the required traits are implemented.
/// * It must also be safe to drop `self` after calling `get_unchecked`.
#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
#[rustc_specialization_trait]
pub unsafe trait TrustedRandomAccess: Sized {
    // Convenience method.
    fn size(&self) -> usize
    where
        Self: Iterator,
    {
        self.size_hint().0
    }
    /// `true` if getting an iterator element may have side effects.
    /// Remember to take inner iterators into account.
    const MAY_HAVE_SIDE_EFFECT: bool;
}

/// Like `Iterator::__iterator_get_unchecked`, but doesn't require the compiler to
/// know that `U: TrustedRandomAccess`.
///
/// ## Safety
///
/// Same requirements calling `get_unchecked` directly.
#[doc(hidden)]
pub(in crate::iter::adapters) unsafe fn try_get_unchecked<I>(it: &mut I, idx: usize) -> I::Item
where
    I: Iterator,
{
    // SAFETY: the caller must uphold the contract for
    // `Iterator::__iterator_get_unchecked`.
    unsafe { it.try_get_unchecked(idx) }
}

unsafe trait SpecTrustedRandomAccess: Iterator {
    /// If `Self: TrustedRandomAccess`, it must be safe to call a
    /// `Iterator::__iterator_get_unchecked(self, index)`.
    unsafe fn try_get_unchecked(&mut self, index: usize) -> Self::Item;
}

unsafe impl<I: Iterator> SpecTrustedRandomAccess for I {
    default unsafe fn try_get_unchecked(&mut self, _: usize) -> Self::Item {
        panic!("Should only be called on TrustedRandomAccess iterators");
    }
}

unsafe impl<I: Iterator + TrustedRandomAccess> SpecTrustedRandomAccess for I {
    unsafe fn try_get_unchecked(&mut self, index: usize) -> Self::Item {
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        unsafe { self.__iterator_get_unchecked(index) }
    }
}
