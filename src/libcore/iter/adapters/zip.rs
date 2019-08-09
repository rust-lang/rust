use crate::cmp;

use super::super::{Iterator, DoubleEndedIterator, ExactSizeIterator, FusedIterator, TrustedLen};

/// An iterator that iterates two other iterators simultaneously.
///
/// This `struct` is created by the [`zip`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`zip`]: trait.Iterator.html#method.zip
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Zip<A, B> {
    a: A,
    b: B,
    // index and len are only used by the specialized version of zip
    index: usize,
    len: usize,
}
impl<A: Iterator, B: Iterator> Zip<A, B> {
    pub(in super::super) fn new(a: A, b: B) -> Zip<A, B> {
        ZipImpl::new(a, b)
    }
    fn super_nth(&mut self, mut n: usize) -> Option<(A::Item, B::Item)> {
        while let Some(x) = Iterator::next(self) {
            if n == 0 { return Some(x) }
            n -= 1;
        }
        None
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> Iterator for Zip<A, B> where A: Iterator, B: Iterator
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
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> DoubleEndedIterator for Zip<A, B> where
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
        where A: DoubleEndedIterator + ExactSizeIterator,
              B: DoubleEndedIterator + ExactSizeIterator;
}

// General Zip impl
#[doc(hidden)]
impl<A, B> ZipImpl<A, B> for Zip<A, B>
    where A: Iterator, B: Iterator
{
    type Item = (A::Item, B::Item);
    default fn new(a: A, b: B) -> Self {
        Zip {
            a,
            b,
            index: 0, // unused
            len: 0, // unused
        }
    }

    #[inline]
    default fn next(&mut self) -> Option<(A::Item, B::Item)> {
        self.a.next().and_then(|x| {
            self.b.next().and_then(|y| {
                Some((x, y))
            })
        })
    }

    #[inline]
    default fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.super_nth(n)
    }

    #[inline]
    default fn next_back(&mut self) -> Option<(A::Item, B::Item)>
        where A: DoubleEndedIterator + ExactSizeIterator,
              B: DoubleEndedIterator + ExactSizeIterator
    {
        let a_sz = self.a.len();
        let b_sz = self.b.len();
        if a_sz != b_sz {
            // Adjust a, b to equal length
            if a_sz > b_sz {
                for _ in 0..a_sz - b_sz { self.a.next_back(); }
            } else {
                for _ in 0..b_sz - a_sz { self.b.next_back(); }
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
            (Some(x), Some(y)) => Some(cmp::min(x,y)),
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(y),
            (None, None) => None
        };

        (lower, upper)
    }
}

#[doc(hidden)]
impl<A, B> ZipImpl<A, B> for Zip<A, B>
    where A: TrustedRandomAccess, B: TrustedRandomAccess
{
    fn new(a: A, b: B) -> Self {
        let len = cmp::min(a.len(), b.len());
        Zip {
            a,
            b,
            index: 0,
            len,
        }
    }

    #[inline]
    fn next(&mut self) -> Option<(A::Item, B::Item)> {
        if self.index < self.len {
            let i = self.index;
            self.index += 1;
            unsafe {
                Some((self.a.get_unchecked(i), self.b.get_unchecked(i)))
            }
        } else if A::may_have_side_effect() && self.index < self.a.len() {
            // match the base implementation's potential side effects
            unsafe {
                self.a.get_unchecked(self.index);
            }
            self.index += 1;
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
            if A::may_have_side_effect() {
                unsafe { self.a.get_unchecked(i); }
            }
            if B::may_have_side_effect() {
                unsafe { self.b.get_unchecked(i); }
            }
        }

        self.super_nth(n - delta)
    }

    #[inline]
    fn next_back(&mut self) -> Option<(A::Item, B::Item)>
        where A: DoubleEndedIterator + ExactSizeIterator,
              B: DoubleEndedIterator + ExactSizeIterator
    {
        // Adjust a, b to equal length
        if A::may_have_side_effect() {
            let sz = self.a.len();
            if sz > self.len {
                for _ in 0..sz - cmp::max(self.len, self.index) {
                    self.a.next_back();
                }
            }
        }
        if B::may_have_side_effect() {
            let sz = self.b.len();
            if sz > self.len {
                for _ in 0..sz - self.len {
                    self.b.next_back();
                }
            }
        }
        if self.index < self.len {
            self.len -= 1;
            let i = self.len;
            unsafe {
                Some((self.a.get_unchecked(i), self.b.get_unchecked(i)))
            }
        } else {
            None
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> ExactSizeIterator for Zip<A, B>
    where A: ExactSizeIterator, B: ExactSizeIterator {}

#[doc(hidden)]
unsafe impl<A, B> TrustedRandomAccess for Zip<A, B>
    where A: TrustedRandomAccess,
          B: TrustedRandomAccess,
{
    unsafe fn get_unchecked(&mut self, i: usize) -> (A::Item, B::Item) {
        (self.a.get_unchecked(i), self.b.get_unchecked(i))
    }

    fn may_have_side_effect() -> bool {
        A::may_have_side_effect() || B::may_have_side_effect()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<A, B> FusedIterator for Zip<A, B>
    where A: FusedIterator, B: FusedIterator, {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A, B> TrustedLen for Zip<A, B>
    where A: TrustedLen, B: TrustedLen,
{}

/// An iterator whose items are random-accessible efficiently
///
/// # Safety
///
/// The iterator's .len() and size_hint() must be exact.
/// `.len()` must be cheap to call.
///
/// .get_unchecked() must return distinct mutable references for distinct
/// indices (if applicable), and must return a valid reference if index is in
/// 0..self.len().
pub(crate) unsafe trait TrustedRandomAccess : ExactSizeIterator {
    unsafe fn get_unchecked(&mut self, i: usize) -> Self::Item;
    /// Returns `true` if getting an iterator element may have
    /// side effects. Remember to take inner iterators into account.
    fn may_have_side_effect() -> bool;
}
