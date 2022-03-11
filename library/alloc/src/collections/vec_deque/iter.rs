use core::fmt;
use core::iter::{FusedIterator, TrustedLen, TrustedRandomAccess, TrustedRandomAccessNoCoerce};
use core::mem::MaybeUninit;
use core::ops::Try;

use super::{count, wrap_index, RingSlices};

/// An iterator over the elements of a `VecDeque`.
///
/// This `struct` is created by the [`iter`] method on [`super::VecDeque`]. See its
/// documentation for more.
///
/// [`iter`]: super::VecDeque::iter
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    pub(crate) ring: &'a [MaybeUninit<T>],
    pub(crate) tail: usize,
    pub(crate) head: usize,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (front, back) = RingSlices::ring_slices(self.ring, self.head, self.tail);
        // Safety:
        // - `self.head` and `self.tail` in a ring buffer are always valid indices.
        // - `RingSlices::ring_slices` guarantees that the slices split according to `self.head` and `self.tail` are initialized.
        unsafe {
            f.debug_tuple("Iter")
                .field(&MaybeUninit::slice_assume_init_ref(front))
                .field(&MaybeUninit::slice_assume_init_ref(back))
                .finish()
        }
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Iter { ring: self.ring, tail: self.tail, head: self.head }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if self.tail == self.head {
            return None;
        }
        let tail = self.tail;
        self.tail = wrap_index(self.tail.wrapping_add(1), self.ring.len());
        // Safety:
        // - `self.tail` in a ring buffer is always a valid index.
        // - `self.head` and `self.tail` equality is checked above.
        unsafe { Some(self.ring.get_unchecked(tail).assume_init_ref()) }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = count(self.tail, self.head, self.ring.len());
        (len, Some(len))
    }

    fn fold<Acc, F>(self, mut accum: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let (front, back) = RingSlices::ring_slices(self.ring, self.head, self.tail);
        // Safety:
        // - `self.head` and `self.tail` in a ring buffer are always valid indices.
        // - `RingSlices::ring_slices` guarantees that the slices split according to `self.head` and `self.tail` are initialized.
        unsafe {
            accum = MaybeUninit::slice_assume_init_ref(front).iter().fold(accum, &mut f);
            MaybeUninit::slice_assume_init_ref(back).iter().fold(accum, &mut f)
        }
    }

    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        let (mut iter, final_res);
        if self.tail <= self.head {
            // Safety: single slice self.ring[self.tail..self.head] is initialized.
            iter = unsafe { MaybeUninit::slice_assume_init_ref(&self.ring[self.tail..self.head]) }
                .iter();
            final_res = iter.try_fold(init, &mut f);
        } else {
            // Safety: two slices: self.ring[self.tail..], self.ring[..self.head] both are initialized.
            let (front, back) = self.ring.split_at(self.tail);

            let mut back_iter = unsafe { MaybeUninit::slice_assume_init_ref(back).iter() };
            let res = back_iter.try_fold(init, &mut f);
            let len = self.ring.len();
            self.tail = (self.ring.len() - back_iter.len()) & (len - 1);
            iter = unsafe { MaybeUninit::slice_assume_init_ref(&front[..self.head]).iter() };
            final_res = iter.try_fold(res?, &mut f);
        }
        self.tail = self.head - iter.len();
        final_res
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n >= count(self.tail, self.head, self.ring.len()) {
            self.tail = self.head;
            None
        } else {
            self.tail = wrap_index(self.tail.wrapping_add(n), self.ring.len());
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<&'a T> {
        self.next_back()
    }

    #[inline]
    #[doc(hidden)]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        // Safety: The TrustedRandomAccess contract requires that callers only pass an index
        // that is in bounds.
        unsafe {
            let idx = wrap_index(self.tail.wrapping_add(idx), self.ring.len());
            self.ring.get_unchecked(idx).assume_init_ref()
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        if self.tail == self.head {
            return None;
        }
        self.head = wrap_index(self.head.wrapping_sub(1), self.ring.len());
        // Safety:
        // - `self.head` in a ring buffer is always a valid index.
        // - `self.head` and `self.tail` equality is checked above.
        unsafe { Some(self.ring.get_unchecked(self.head).assume_init_ref()) }
    }

    fn rfold<Acc, F>(self, mut accum: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let (front, back) = RingSlices::ring_slices(self.ring, self.head, self.tail);
        // Safety:
        // - `self.head` and `self.tail` in a ring buffer are always valid indices.
        // - `RingSlices::ring_slices` guarantees that the slices split according to `self.head` and `self.tail` are initialized.
        unsafe {
            accum = MaybeUninit::slice_assume_init_ref(back).iter().rfold(accum, &mut f);
            MaybeUninit::slice_assume_init_ref(front).iter().rfold(accum, &mut f)
        }
    }

    fn try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        let (mut iter, final_res);
        if self.tail <= self.head {
            // Safety: single slice self.ring[self.tail..self.head] is initialized.
            iter = unsafe {
                MaybeUninit::slice_assume_init_ref(&self.ring[self.tail..self.head]).iter()
            };
            final_res = iter.try_rfold(init, &mut f);
        } else {
            // Safety: two slices: self.ring[self.tail..], self.ring[..self.head] both are initialized.
            let (front, back) = self.ring.split_at(self.tail);

            let mut front_iter =
                unsafe { MaybeUninit::slice_assume_init_ref(&front[..self.head]).iter() };
            let res = front_iter.try_rfold(init, &mut f);
            self.head = front_iter.len();
            iter = unsafe { MaybeUninit::slice_assume_init_ref(back).iter() };
            final_res = iter.try_rfold(res?, &mut f);
        }
        self.head = self.tail + iter.len();
        final_res
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for Iter<'_, T> {
    fn is_empty(&self) -> bool {
        self.head == self.tail
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
