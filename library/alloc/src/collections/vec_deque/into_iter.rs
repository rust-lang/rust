use core::iter::{FusedIterator, TrustedLen};
use core::mem::MaybeUninit;
use core::num::NonZero;
use core::ops::Try;
use core::{array, fmt, ptr};

use super::VecDeque;
use crate::alloc::{Allocator, Global};

/// An owning iterator over the elements of a `VecDeque`.
///
/// This `struct` is created by the [`into_iter`] method on [`VecDeque`]
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
///
/// [`into_iter`]: VecDeque::into_iter
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<
    T,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    inner: VecDeque<T, A>,
}

impl<T, A: Allocator> IntoIter<T, A> {
    pub(super) fn new(inner: VecDeque<T, A>) -> Self {
        IntoIter { inner }
    }

    pub(super) fn into_vecdeque(self) -> VecDeque<T, A> {
        self.inner
    }
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug, A: Allocator> fmt::Debug for IntoIter<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoIter").field(&self.inner).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, A: Allocator> Iterator for IntoIter<T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.pop_front()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.len();
        (len, Some(len))
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        let len = self.inner.len;
        let rem = if len < n {
            self.inner.clear();
            n - len
        } else {
            self.inner.drain(..n);
            0
        };
        NonZero::new(rem).map_or(Ok(()), Err)
    }

    #[inline]
    fn count(self) -> usize {
        self.inner.len
    }

    fn try_fold<B, F, R>(&mut self, mut init: B, mut f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        struct Guard<'a, T, A: Allocator> {
            deque: &'a mut VecDeque<T, A>,
            // `consumed <= deque.len` always holds.
            consumed: usize,
        }

        impl<'a, T, A: Allocator> Drop for Guard<'a, T, A> {
            fn drop(&mut self) {
                self.deque.len -= self.consumed;
                self.deque.head = self.deque.to_physical_idx(self.consumed);
            }
        }

        let mut guard = Guard { deque: &mut self.inner, consumed: 0 };

        let (head, tail) = guard.deque.as_slices();

        init = head
            .iter()
            .map(|elem| {
                guard.consumed += 1;
                // SAFETY: Because we incremented `guard.consumed`, the
                // deque effectively forgot the element, so we can take
                // ownership
                unsafe { ptr::read(elem) }
            })
            .try_fold(init, &mut f)?;

        tail.iter()
            .map(|elem| {
                guard.consumed += 1;
                // SAFETY: Same as above.
                unsafe { ptr::read(elem) }
            })
            .try_fold(init, &mut f)
    }

    #[inline]
    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        match self.try_fold(init, |b, item| Ok::<B, !>(f(b, item))) {
            Ok(b) => b,
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.inner.pop_back()
    }

    fn next_chunk<const N: usize>(
        &mut self,
    ) -> Result<[Self::Item; N], array::IntoIter<Self::Item, N>> {
        let mut raw_arr = [const { MaybeUninit::uninit() }; N];
        let raw_arr_ptr = raw_arr.as_mut_ptr().cast();
        let (head, tail) = self.inner.as_slices();

        if head.len() >= N {
            // SAFETY: By manually adjusting the head and length of the deque, we effectively
            // make it forget the first `N` elements, so taking ownership of them is safe.
            unsafe { ptr::copy_nonoverlapping(head.as_ptr(), raw_arr_ptr, N) };
            self.inner.head = self.inner.to_physical_idx(N);
            self.inner.len -= N;
            // SAFETY: We initialized the entire array with items from `head`
            return Ok(unsafe { raw_arr.transpose().assume_init() });
        }

        // SAFETY: Same argument as above.
        unsafe { ptr::copy_nonoverlapping(head.as_ptr(), raw_arr_ptr, head.len()) };
        let remaining = N - head.len();

        if tail.len() >= remaining {
            // SAFETY: Same argument as above.
            unsafe {
                ptr::copy_nonoverlapping(tail.as_ptr(), raw_arr_ptr.add(head.len()), remaining)
            };
            self.inner.head = self.inner.to_physical_idx(N);
            self.inner.len -= N;
            // SAFETY: We initialized the entire array with items from `head` and `tail`
            Ok(unsafe { raw_arr.transpose().assume_init() })
        } else {
            // SAFETY: Same argument as above.
            unsafe {
                ptr::copy_nonoverlapping(tail.as_ptr(), raw_arr_ptr.add(head.len()), tail.len())
            };
            let init = head.len() + tail.len();
            // We completely drained all the deques elements.
            self.inner.head = 0;
            self.inner.len = 0;
            // SAFETY: We copied all elements from both slices to the beginning of the array, so
            // the given range is initialized.
            Err(unsafe { array::IntoIter::new_unchecked(raw_arr, 0..init) })
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, A: Allocator> DoubleEndedIterator for IntoIter<T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.inner.pop_back()
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        let len = self.inner.len;
        let rem = if len < n {
            self.inner.clear();
            n - len
        } else {
            self.inner.truncate(len - n);
            0
        };
        NonZero::new(rem).map_or(Ok(()), Err)
    }

    fn try_rfold<B, F, R>(&mut self, mut init: B, mut f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        struct Guard<'a, T, A: Allocator> {
            deque: &'a mut VecDeque<T, A>,
            // `consumed <= deque.len` always holds.
            consumed: usize,
        }

        impl<'a, T, A: Allocator> Drop for Guard<'a, T, A> {
            fn drop(&mut self) {
                self.deque.len -= self.consumed;
            }
        }

        let mut guard = Guard { deque: &mut self.inner, consumed: 0 };

        let (head, tail) = guard.deque.as_slices();

        init = tail
            .iter()
            .map(|elem| {
                guard.consumed += 1;
                // SAFETY: See `try_fold`'s safety comment.
                unsafe { ptr::read(elem) }
            })
            .try_rfold(init, &mut f)?;

        head.iter()
            .map(|elem| {
                guard.consumed += 1;
                // SAFETY: Same as above.
                unsafe { ptr::read(elem) }
            })
            .try_rfold(init, &mut f)
    }

    #[inline]
    fn rfold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        match self.try_rfold(init, |b, item| Ok::<B, !>(f(b, item))) {
            Ok(b) => b,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, A: Allocator> ExactSizeIterator for IntoIter<T, A> {
    #[inline]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, A: Allocator> FusedIterator for IntoIter<T, A> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T, A: Allocator> TrustedLen for IntoIter<T, A> {}
