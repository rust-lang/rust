use crate::co_alloc::CoAllocPref;
use core::iter::{FusedIterator, TrustedLen};
use core::num::NonZeroUsize;
use core::{array, fmt, mem::MaybeUninit, ops::Try, ptr};

use crate::alloc::{Allocator, Global};

use super::VecDeque;

/// An owning iterator over the elements of a `VecDeque`.
///
/// This `struct` is created by the [`into_iter`] method on [`VecDeque`]
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
///
/// [`into_iter`]: VecDeque::into_iter
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(unused_braces)]
pub struct IntoIter<
    T,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
    const CO_ALLOC_PREF: CoAllocPref = { CO_ALLOC_PREF_DEFAULT!() },
> where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    inner: VecDeque<T, A, CO_ALLOC_PREF>,
}

#[allow(unused_braces)]
impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> IntoIter<T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    pub(super) fn new(inner: VecDeque<T, A, CO_ALLOC_PREF>) -> Self {
        IntoIter { inner }
    }

    pub(super) fn into_vecdeque(self) -> VecDeque<T, A, CO_ALLOC_PREF> {
        self.inner
    }
}

#[stable(feature = "collection_debug", since = "1.17.0")]
#[allow(unused_braces)]
impl<T: fmt::Debug, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> fmt::Debug
    for IntoIter<T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoIter").field(&self.inner).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(unused_braces)]
impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> Iterator for IntoIter<T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
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
    fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        let len = self.inner.len;
        let rem = if len < n {
            self.inner.clear();
            n - len
        } else {
            self.inner.drain(..n);
            0
        };
        NonZeroUsize::new(rem).map_or(Ok(()), Err)
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
        struct Guard<'a, T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref>
        where
            [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
        {
            deque: &'a mut VecDeque<T, A, CO_ALLOC_PREF>,
            // `consumed <= deque.len` always holds.
            consumed: usize,
        }

        impl<'a, T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> Drop for Guard<'a, T, A, CO_ALLOC_PREF>
        where
            [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
        {
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
            Err(e) => match e {},
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.inner.pop_back()
    }

    fn next_chunk<const N: usize>(
        &mut self,
    ) -> Result<[Self::Item; N], array::IntoIter<Self::Item, N>> {
        let mut raw_arr = MaybeUninit::uninit_array();
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
#[allow(unused_braces)]
impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> DoubleEndedIterator
    for IntoIter<T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.inner.pop_back()
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        let len = self.inner.len;
        let rem = if len < n {
            self.inner.clear();
            n - len
        } else {
            self.inner.truncate(len - n);
            0
        };
        NonZeroUsize::new(rem).map_or(Ok(()), Err)
    }

    fn try_rfold<B, F, R>(&mut self, mut init: B, mut f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        struct Guard<'a, T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref>
        where
            [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
        {
            deque: &'a mut VecDeque<T, A, CO_ALLOC_PREF>,
            // `consumed <= deque.len` always holds.
            consumed: usize,
        }

        impl<'a, T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> Drop for Guard<'a, T, A, CO_ALLOC_PREF>
        where
            [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
        {
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
            Err(e) => match e {},
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(unused_braces)]
impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> ExactSizeIterator
    for IntoIter<T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    #[inline]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
#[allow(unused_braces)]
impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> FusedIterator
    for IntoIter<T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
}

#[unstable(feature = "trusted_len", issue = "37572")]
#[allow(unused_braces)]
unsafe impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> TrustedLen
    for IntoIter<T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
}
