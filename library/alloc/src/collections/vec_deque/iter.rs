use core::iter::{FusedIterator, TrustedLen, TrustedRandomAccess, TrustedRandomAccessNoCoerce};
use core::num::NonZeroUsize;
use core::ops::Try;
use core::{fmt, mem, slice};

/// An iterator over the elements of a `VecDeque`.
///
/// This `struct` is created by the [`iter`] method on [`super::VecDeque`]. See its
/// documentation for more.
///
/// [`iter`]: super::VecDeque::iter
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    i1: slice::Iter<'a, T>,
    i2: slice::Iter<'a, T>,
}

impl<'a, T> Iter<'a, T> {
    pub(super) fn new(i1: slice::Iter<'a, T>, i2: slice::Iter<'a, T>) -> Self {
        Self { i1, i2 }
    }
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Iter").field(&self.i1.as_slice()).field(&self.i2.as_slice()).finish()
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Iter { i1: self.i1.clone(), i2: self.i2.clone() }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        match self.i1.next() {
            Some(val) => Some(val),
            None => {
                // most of the time, the iterator will either always
                // call next(), or always call next_back(). By swapping
                // the iterators once the first one is empty, we ensure
                // that the first branch is taken as often as possible,
                // without sacrificing correctness, as i1 is empty anyways
                mem::swap(&mut self.i1, &mut self.i2);
                self.i1.next()
            }
        }
    }

    fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        let remaining = self.i1.advance_by(n);
        match remaining {
            Ok(()) => return Ok(()),
            Err(n) => {
                mem::swap(&mut self.i1, &mut self.i2);
                self.i1.advance_by(n.get())
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn fold<Acc, F>(self, accum: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let accum = self.i1.fold(accum, &mut f);
        self.i2.fold(accum, &mut f)
    }

    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        let acc = self.i1.try_fold(init, &mut f)?;
        self.i2.try_fold(acc, &mut f)
    }

    #[inline]
    fn last(mut self) -> Option<&'a T> {
        self.next_back()
    }

    #[inline]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        // Safety: The TrustedRandomAccess contract requires that callers only pass an index
        // that is in bounds.
        unsafe {
            let i1_len = self.i1.len();
            if idx < i1_len {
                self.i1.__iterator_get_unchecked(idx)
            } else {
                self.i2.__iterator_get_unchecked(idx - i1_len)
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        match self.i2.next_back() {
            Some(val) => Some(val),
            None => {
                // most of the time, the iterator will either always
                // call next(), or always call next_back(). By swapping
                // the iterators once the second one is empty, we ensure
                // that the first branch is taken as often as possible,
                // without sacrificing correctness, as i2 is empty anyways
                mem::swap(&mut self.i1, &mut self.i2);
                self.i2.next_back()
            }
        }
    }

    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        match self.i2.advance_back_by(n) {
            Ok(()) => return Ok(()),
            Err(n) => {
                mem::swap(&mut self.i1, &mut self.i2);
                self.i2.advance_back_by(n.get())
            }
        }
    }

    fn rfold<Acc, F>(self, accum: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let accum = self.i2.rfold(accum, &mut f);
        self.i1.rfold(accum, &mut f)
    }

    fn try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        let acc = self.i2.try_rfold(init, &mut f)?;
        self.i1.try_rfold(acc, &mut f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for Iter<'_, T> {
    fn len(&self) -> usize {
        self.i1.len() + self.i2.len()
    }

    fn is_empty(&self) -> bool {
        self.i1.is_empty() && self.i2.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for Iter<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for Iter<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<T> TrustedRandomAccess for Iter<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<T> TrustedRandomAccessNoCoerce for Iter<'_, T> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}
