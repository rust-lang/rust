use crate::co_alloc::CoAllocPref;
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::{self, SizedTypeProperties};
use core::ptr::NonNull;
use core::{fmt, ptr};

use crate::alloc::{Allocator, Global};

use super::VecDeque;

/// A draining iterator over the elements of a `VecDeque`.
///
/// This `struct` is created by the [`drain`] method on [`VecDeque`]. See its
/// documentation for more.
///
/// [`drain`]: VecDeque::drain
#[stable(feature = "drain", since = "1.6.0")]
#[allow(unused_braces)]
pub struct Drain<
    'a,
    T: 'a,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
    const CO_ALLOC_PREF: CoAllocPref = { SHORT_TERM_VEC_CO_ALLOC_PREF!() },
> where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    deque: NonNull<VecDeque<T, A, CO_ALLOC_PREF>>,
    // drain_start is stored in deque.len
    drain_len: usize,
    // index into the logical array, not the physical one (always lies in [0..deque.len))
    idx: usize,
    // number of elements after the drain range
    tail_len: usize,
    remaining: usize,
    // Needed to make Drain covariant over T
    _marker: PhantomData<&'a T>,
}

#[allow(unused_braces)]
impl<'a, T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> Drain<'a, T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    pub(super) unsafe fn new(
        deque: &'a mut VecDeque<T, A, CO_ALLOC_PREF>,
        drain_start: usize,
        drain_len: usize,
    ) -> Self {
        let orig_len = mem::replace(&mut deque.len, drain_start);
        let tail_len = orig_len - drain_start - drain_len;
        Drain {
            deque: NonNull::from(deque),
            drain_len,
            idx: drain_start,
            tail_len,
            remaining: drain_len,
            _marker: PhantomData,
        }
    }

    // Only returns pointers to the slices, as that's all we need
    // to drop them. May only be called if `self.remaining != 0`.
    unsafe fn as_slices(&self) -> (*mut [T], *mut [T]) {
        unsafe {
            let deque = self.deque.as_ref();

            // We know that `self.idx + self.remaining <= deque.len <= usize::MAX`, so this won't overflow.
            let logical_remaining_range = self.idx..self.idx + self.remaining;

            // SAFETY: `logical_remaining_range` represents the
            // range into the logical buffer of elements that
            // haven't been drained yet, so they're all initialized,
            // and `slice::range(start..end, end) == start..end`,
            // so the preconditions for `slice_ranges` are met.
            let (a_range, b_range) =
                deque.slice_ranges(logical_remaining_range.clone(), logical_remaining_range.end);
            (deque.buffer_range(a_range), deque.buffer_range(b_range))
        }
    }
}

#[stable(feature = "collection_debug", since = "1.17.0")]
#[allow(unused_braces)]
impl<T: fmt::Debug, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> fmt::Debug
    for Drain<'_, T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Drain")
            .field(&self.drain_len)
            .field(&self.idx)
            .field(&self.tail_len)
            .field(&self.remaining)
            .finish()
    }
}

#[stable(feature = "drain", since = "1.6.0")]
#[allow(unused_braces)]
unsafe impl<T: Sync, A: Allocator + Sync, const CO_ALLOC_PREF: CoAllocPref> Sync
    for Drain<'_, T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
}
#[stable(feature = "drain", since = "1.6.0")]
#[allow(unused_braces)]
unsafe impl<T: Send, A: Allocator + Send, const CO_ALLOC_PREF: CoAllocPref> Send
    for Drain<'_, T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
}

#[stable(feature = "drain", since = "1.6.0")]
#[allow(unused_braces)]
impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> Drop for Drain<'_, T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    fn drop(&mut self) {
        struct DropGuard<'r, 'a, T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref>(
            &'r mut Drain<'a, T, A, CO_ALLOC_PREF>,
        )
        where
            [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:;

        impl<'r, 'a, T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> Drop
            for DropGuard<'r, 'a, T, A, CO_ALLOC_PREF>
        where
            [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
        {
            fn drop(&mut self) {
                if self.0.remaining != 0 {
                    unsafe {
                        // SAFETY: We just checked that `self.remaining != 0`.
                        let (front, back) = self.0.as_slices();
                        ptr::drop_in_place(front);
                        ptr::drop_in_place(back);
                    }
                }

                let source_deque = unsafe { self.0.deque.as_mut() };

                let drain_start = source_deque.len();
                let drain_len = self.0.drain_len;
                let drain_end = drain_start + drain_len;

                let orig_len = self.0.tail_len + drain_end;

                if T::IS_ZST {
                    // no need to copy around any memory if T is a ZST
                    source_deque.len = orig_len - drain_len;
                    return;
                }

                let head_len = drain_start;
                let tail_len = self.0.tail_len;

                match (head_len, tail_len) {
                    (0, 0) => {
                        source_deque.head = 0;
                        source_deque.len = 0;
                    }
                    (0, _) => {
                        source_deque.head = source_deque.to_physical_idx(drain_len);
                        source_deque.len = orig_len - drain_len;
                    }
                    (_, 0) => {
                        source_deque.len = orig_len - drain_len;
                    }
                    _ => unsafe {
                        if head_len <= tail_len {
                            source_deque.wrap_copy(
                                source_deque.head,
                                source_deque.to_physical_idx(drain_len),
                                head_len,
                            );
                            source_deque.head = source_deque.to_physical_idx(drain_len);
                            source_deque.len = orig_len - drain_len;
                        } else {
                            source_deque.wrap_copy(
                                source_deque.to_physical_idx(head_len + drain_len),
                                source_deque.to_physical_idx(head_len),
                                tail_len,
                            );
                            source_deque.len = orig_len - drain_len;
                        }
                    },
                }
            }
        }

        let guard = DropGuard(self);
        if guard.0.remaining != 0 {
            unsafe {
                // SAFETY: We just checked that `self.remaining != 0`.
                let (front, back) = guard.0.as_slices();
                // since idx is a logical index, we don't need to worry about wrapping.
                guard.0.idx += front.len();
                guard.0.remaining -= front.len();
                ptr::drop_in_place(front);
                guard.0.remaining = 0;
                ptr::drop_in_place(back);
            }
        }

        // Dropping `guard` handles moving the remaining elements into place.
    }
}

#[stable(feature = "drain", since = "1.6.0")]
#[allow(unused_braces)]
impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> Iterator for Drain<'_, T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.remaining == 0 {
            return None;
        }
        let wrapped_idx = unsafe { self.deque.as_ref().to_physical_idx(self.idx) };
        self.idx += 1;
        self.remaining -= 1;
        Some(unsafe { self.deque.as_mut().buffer_read(wrapped_idx) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remaining;
        (len, Some(len))
    }
}

#[stable(feature = "drain", since = "1.6.0")]
#[allow(unused_braces)]
impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> DoubleEndedIterator
    for Drain<'_, T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let wrapped_idx = unsafe { self.deque.as_ref().to_physical_idx(self.idx + self.remaining) };
        Some(unsafe { self.deque.as_mut().buffer_read(wrapped_idx) })
    }
}

#[stable(feature = "drain", since = "1.6.0")]
#[allow(unused_braces)]
impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> ExactSizeIterator
    for Drain<'_, T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
}

#[stable(feature = "fused", since = "1.26.0")]
#[allow(unused_braces)]
impl<T, A: Allocator, const CO_ALLOC_PREF: CoAllocPref> FusedIterator
    for Drain<'_, T, A, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots!(A, CO_ALLOC_PREF) }]:,
{
}
