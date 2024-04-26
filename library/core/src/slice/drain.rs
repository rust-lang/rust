use crate::array;
use crate::fmt;
use crate::iter::{
    FusedIterator, TrustedFused, TrustedLen, TrustedRandomAccessNoCoerce, UncheckedIterator,
};
use crate::mem::MaybeUninit;
use crate::num::NonZero;
use crate::ptr::NonNull;
use crate::slice::NonNullIter;

/// An iterator which takes ownership of items out of a slice, dropping any
/// remaining items when the iterator drops.
///
/// Note that, like a raw pointer, it's **up to you** to get the lifetime right.
/// In some ways it's actually harder to get right, as the iterator interface
/// appears safe, but as you promise when creating one of these, you still must
/// ensure that the mentioned memory is usable the whole time this lives.
///
/// Ideally you won't be using this directly, but rather a version encapsulated
/// in a safer interface, like `vec::IntoIter`.
///
/// This raw version may be removed in favour of a future language feature,
/// such as using `unsafe<'a> Drain<'a, T>` instead of `DrainRaw<T>`.
#[unstable(feature = "slice_drain_raw_iter", issue = "none")]
pub struct DrainRaw<T>(NonNullIter<T>);

#[unstable(feature = "slice_drain_raw_iter", issue = "none")]
// `may_dangle` is needed for compatibility with `vec::IntoIter`
unsafe impl<#[may_dangle] T> Drop for DrainRaw<T> {
    fn drop(&mut self) {
        // When used in things like `vec::IntoIter`, the memory over which we're
        // iterating might have been deallocated once we're running this drop.
        // At the time of writing, Miri doesn't like `sub_ptr` between pointers
        // into a deallocated allocation.  So checking empty first -- which just
        // needs pointer equality -- avoids that issue.
        if !self.is_empty() {
            let slice = self.as_nonnull_slice();
            // SAFETY: By type invariant, we're allowed to drop the rest of the items.
            unsafe { slice.drop_in_place() };
        }
    }
}

#[unstable(feature = "slice_drain_raw_iter", issue = "none")]
impl<T: fmt::Debug> fmt::Debug for DrainRaw<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DrainRaw").field(&self.0.make_shortlived_slice()).finish()
    }
}

impl<T> DrainRaw<T> {
    /// Creates a new iterator which moves the `len` items starting at `ptr`
    /// while it's iterated, or drops them when the iterator is dropped.
    ///
    /// # Safety
    ///
    /// - `ptr` through `ptr.add(len)` must be a single allocated object
    ///   such that that it's sound to `offset` through it.
    /// - All those elements must be readable, including being sufficiently aligned.
    /// - All those elements are valid for dropping.
    #[unstable(feature = "slice_drain_raw_iter", issue = "none")]
    #[inline]
    pub unsafe fn from_parts(ptr: NonNull<T>, len: usize) -> Self {
        // SAFETY: this function's safety conditions are stricter than NonNullIter,
        // and include allowing the type to drop the items in `Drop`.
        Self(unsafe { NonNullIter::from_parts(ptr, len) })
    }

    /// Returns a pointer to the remaining elements of the iterator
    #[unstable(feature = "slice_drain_raw_iter", issue = "none")]
    #[inline]
    pub fn as_nonnull_slice(&self) -> NonNull<[T]> {
        self.0.make_nonnull_slice()
    }

    /// Equivalent to exhausting the iterator normally, but faster.
    #[unstable(feature = "slice_drain_raw_iter", issue = "none")]
    #[inline]
    pub fn drop_remaining(&mut self) {
        let all = self.forget_remaining();
        // SAFETY: We "forgot" these elements so our `Drop` won't drop them,
        // so it's ok to drop them here without risking double-frees.
        unsafe { all.drop_in_place() }
    }

    /// Exhaust the iterator without actually dropping the rest of the items.
    ///
    /// Returns the forgotten items.
    #[unstable(feature = "slice_drain_raw_iter", issue = "none")]
    #[inline]
    pub fn forget_remaining(&mut self) -> NonNull<[T]> {
        let all = self.as_nonnull_slice();
        self.0.exhaust();
        all
    }
}

impl<T> UncheckedIterator for DrainRaw<T> {
    #[inline]
    unsafe fn next_unchecked(&mut self) -> T {
        // SAFETY: we're a 1:1 mapping of the inner iterator, so if the caller
        // proved we have another item, the inner iterator has another one too.
        // Also, the `next_unchecked` means the returned item is no longer part
        // of the inner iterator, and thus `read`ing it here -- and giving it
        // to the caller who will (probably) drop  it -- is ok.
        unsafe { self.0.next_unchecked().read() }
    }
}

#[unstable(feature = "slice_drain_raw_iter", issue = "none")]
impl<T> Iterator for DrainRaw<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        match self.0.next() {
            // SAFETY: The `next` means the returned item is no longer part of
            // the inner iterator, and thus `read`ing it here -- and giving it
            // to the caller who will (probably) drop it -- is ok.
            Some(ptr) => Some(unsafe { ptr.read() }),
            None => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        let clamped = self.len().min(n);
        // SAFETY: By construction, `clamped` is always in-bounds.
        // The skipped elements are removed from the inner iterator so won't be
        // dropped in `Drop`, so dropping there here is fine.
        unsafe {
            let to_drop = self.0.skip_forward_unchecked(clamped);
            to_drop.drop_in_place();
        }
        NonZero::new(n - clamped).map_or(Ok(()), Err)
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn next_chunk<const N: usize>(&mut self) -> Result<[T; N], core::array::IntoIter<T, N>> {
        let len = self.len();
        let clamped = len.min(N);

        // SAFETY: By construction, `clamped` is always in-bounds.
        let to_copy = unsafe { self.0.skip_forward_unchecked(clamped) };
        if len >= N {
            // SAFETY: If we have more elements than were requested, they can be
            // read directly because arrays need no extra alignment.
            Ok(unsafe { to_copy.cast::<[T; N]>().read() })
        } else {
            let mut raw_ary = MaybeUninit::uninit_array();
            // SAFETY: If we don't have enough elements left, then copy all the
            // ones we do have into the local array, which cannot overlap because
            // new locals are always distinct storage.
            Err(unsafe {
                MaybeUninit::<T>::slice_as_mut_ptr(&mut raw_ary)
                    .copy_from_nonoverlapping(to_copy.as_mut_ptr(), len);
                array::IntoIter::new_unchecked(raw_ary, 0..len)
            })
        }
    }

    unsafe fn __iterator_get_unchecked(&mut self, i: usize) -> Self::Item
    where
        Self: TrustedRandomAccessNoCoerce,
    {
        // SAFETY: the caller must guarantee that `i` is in bounds of the slice,
        // so the `get_unchecked_mut(i)` is guaranteed to pointer to an element
        // and thus guaranteed to be valid to dereference.
        //
        // Also note the implementation of `Self: TrustedRandomAccess` requires
        // that `T: Copy` so reading elements from the buffer doesn't invalidate
        // them for `Drop`.
        unsafe { self.as_nonnull_slice().get_unchecked_mut(i).read() }
    }
}

#[unstable(feature = "slice_drain_raw_iter", issue = "none")]
impl<T> DoubleEndedIterator for DrainRaw<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        match self.0.next_back() {
            // SAFETY: The `next_back` means the returned item is no longer part of
            // the inner iterator, and thus `read`ing it here -- and giving it
            // to the caller who will (probably) drop  it -- is ok.
            Some(ptr) => Some(unsafe { ptr.read() }),
            None => None,
        }
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        let clamped = self.len().min(n);
        // SAFETY: By construction, `clamped` is always in-bounds.
        // The skipped elements are removed from the inner iterator so won't be
        // dropped in `Drop`, so dropping there here is fine.
        unsafe {
            let to_drop = self.0.skip_backward_unchecked(clamped);
            to_drop.drop_in_place();
        }
        NonZero::new(n - clamped).map_or(Ok(()), Err)
    }
}

#[unstable(feature = "slice_drain_raw_iter", issue = "none")]
impl<T> ExactSizeIterator for DrainRaw<T> {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

#[unstable(feature = "slice_drain_raw_iter", issue = "none")]
impl<T> FusedIterator for DrainRaw<T> {}

#[unstable(feature = "slice_drain_raw_iter", issue = "none")]
#[doc(hidden)]
unsafe impl<T> TrustedFused for DrainRaw<T> {}

#[unstable(feature = "slice_drain_raw_iter", issue = "none")]
unsafe impl<T> TrustedLen for DrainRaw<T> {}

#[doc(hidden)]
#[unstable(issue = "none", feature = "std_internals")]
#[rustc_unsafe_specialization_marker]
pub trait NonDrop {}

// T: Copy as approximation for !Drop since get_unchecked does not advance self.ptr
// and thus we can't implement drop-handling
#[unstable(issue = "none", feature = "std_internals")]
impl<T: Copy> NonDrop for T {}

// TrustedRandomAccess (without NoCoerce) must not be implemented because
// subtypes/supertypes of `T` might not be `NonDrop`
#[unstable(feature = "slice_drain_raw_iter", issue = "none")]
unsafe impl<T: NonDrop> TrustedRandomAccessNoCoerce for DrainRaw<T> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}
