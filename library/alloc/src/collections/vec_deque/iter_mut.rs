use core::iter::{FusedIterator, TrustedLen, TrustedRandomAccess, TrustedRandomAccessNoCoerce};
use core::num::NonZero;
use core::ops::Try;
use core::{fmt, mem, slice};

/// A mutable iterator over the elements of a `VecDeque`.
///
/// This `struct` is created by the [`iter_mut`] method on [`super::VecDeque`]. See its
/// documentation for more.
///
/// [`iter_mut`]: super::VecDeque::iter_mut
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, T: 'a> {
    i1: slice::IterMut<'a, T>,
    i2: slice::IterMut<'a, T>,
}

impl<'a, T> IterMut<'a, T> {
    pub(super) fn new(i1: slice::IterMut<'a, T>, i2: slice::IterMut<'a, T>) -> Self {
        Self { i1, i2 }
    }

    /// Views the underlying data as a pair of subslices of the original data.
    ///
    /// The slices contain, in order, the contents of the deque not yet yielded
    /// by the iterator.
    ///
    /// To avoid creating `&mut` references that alias, this is forced to
    /// consume the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(vec_deque_iter_as_slices)]
    ///
    /// use std::collections::VecDeque;
    ///
    /// let mut deque = VecDeque::new();
    /// deque.push_back(0);
    /// deque.push_back(1);
    /// deque.push_back(2);
    /// deque.push_front(10);
    /// deque.push_front(9);
    /// deque.push_front(8);
    ///
    /// let mut iter = deque.iter_mut();
    /// iter.next();
    /// iter.next_back();
    ///
    /// let slices = iter.into_slices();
    /// slices.0[0] = 42;
    /// slices.1[0] = 24;
    /// assert_eq!(deque.as_slices(), (&[8, 42, 10][..], &[24, 1, 2][..]));
    /// ```
    #[unstable(feature = "vec_deque_iter_as_slices", issue = "123947")]
    pub fn into_slices(self) -> (&'a mut [T], &'a mut [T]) {
        (self.i1.into_slice(), self.i2.into_slice())
    }

    /// Views the underlying data as a pair of subslices of the original data.
    ///
    /// The slices contain, in order, the contents of the deque not yet yielded
    /// by the iterator.
    ///
    /// To avoid creating `&mut [T]` references that alias, the returned slices
    /// borrow their lifetimes from the iterator the method is applied on.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(vec_deque_iter_as_slices)]
    ///
    /// use std::collections::VecDeque;
    ///
    /// let mut deque = VecDeque::new();
    /// deque.push_back(0);
    /// deque.push_back(1);
    /// deque.push_back(2);
    /// deque.push_front(10);
    /// deque.push_front(9);
    /// deque.push_front(8);
    ///
    /// let mut iter = deque.iter_mut();
    /// iter.next();
    /// iter.next_back();
    ///
    /// assert_eq!(iter.as_slices(), (&[9, 10][..], &[0, 1][..]));
    /// ```
    #[unstable(feature = "vec_deque_iter_as_slices", issue = "123947")]
    pub fn as_slices(&self) -> (&[T], &[T]) {
        (self.i1.as_slice(), self.i2.as_slice())
    }

    /// Views the underlying data as a pair of subslices of the original data.
    ///
    /// The slices contain, in order, the contents of the deque not yet yielded
    /// by the iterator.
    ///
    /// To avoid creating `&mut [T]` references that alias, the returned slices
    /// borrow their lifetimes from the iterator the method is applied on.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(vec_deque_iter_as_slices)]
    ///
    /// use std::collections::VecDeque;
    ///
    /// let mut deque = VecDeque::new();
    /// deque.push_back(0);
    /// deque.push_back(1);
    /// deque.push_back(2);
    /// deque.push_front(10);
    /// deque.push_front(9);
    /// deque.push_front(8);
    ///
    /// let mut iter = deque.iter_mut();
    /// iter.next();
    /// iter.next_back();
    ///
    /// iter.as_mut_slices().0[0] = 42;
    /// iter.as_mut_slices().1[0] = 24;
    /// assert_eq!(deque.as_slices(), (&[8, 42, 10][..], &[24, 1, 2][..]));
    /// ```
    #[unstable(feature = "vec_deque_iter_as_slices", issue = "123947")]
    pub fn as_mut_slices(&mut self) -> (&mut [T], &mut [T]) {
        (self.i1.as_mut_slice(), self.i2.as_mut_slice())
    }
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug> fmt::Debug for IterMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IterMut").field(&self.i1.as_slice()).field(&self.i2.as_slice()).finish()
    }
}

#[stable(feature = "default_iters_sequel", since = "1.82.0")]
impl<T> Default for IterMut<'_, T> {
    /// Creates an empty `vec_deque::IterMut`.
    ///
    /// ```
    /// # use std::collections::vec_deque;
    /// let iter: vec_deque::IterMut<'_, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        IterMut { i1: Default::default(), i2: Default::default() }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
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

    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        match self.i1.advance_by(n) {
            Ok(()) => Ok(()),
            Err(remaining) => {
                mem::swap(&mut self.i1, &mut self.i2);
                self.i1.advance_by(remaining.get())
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
    fn last(mut self) -> Option<&'a mut T> {
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
impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut T> {
        match self.i2.next_back() {
            Some(val) => Some(val),
            None => {
                // most of the time, the iterator will either always
                // call next(), or always call next_back(). By swapping
                // the iterators once the first one is empty, we ensure
                // that the first branch is taken as often as possible,
                // without sacrificing correctness, as i2 is empty anyways
                mem::swap(&mut self.i1, &mut self.i2);
                self.i2.next_back()
            }
        }
    }

    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        match self.i2.advance_back_by(n) {
            Ok(()) => Ok(()),
            Err(remaining) => {
                mem::swap(&mut self.i1, &mut self.i2);
                self.i2.advance_back_by(remaining.get())
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
impl<T> ExactSizeIterator for IterMut<'_, T> {
    fn len(&self) -> usize {
        self.i1.len() + self.i2.len()
    }

    fn is_empty(&self) -> bool {
        self.i1.is_empty() && self.i2.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for IterMut<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for IterMut<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<T> TrustedRandomAccess for IterMut<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<T> TrustedRandomAccessNoCoerce for IterMut<'_, T> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}
