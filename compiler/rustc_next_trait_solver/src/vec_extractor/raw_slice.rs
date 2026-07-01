use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};

/// A version of [`core::slice::Iter`] that works on raw pointers.
/// This should really be part of `core`...
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct RawSliceIter<'a, T> {
    /// The pointer to the next element to return, or the past-the-end location
    /// if the iterator is empty.
    ///
    /// This address will be used for all ZST elements, never changed.
    ptr: NonNull<T>,
    /// For non-ZSTs, the non-null pointer to the past-the-end element.
    ///
    /// For ZSTs, this is `ptr::without_provenance_mut(len)`.
    end_or_len: *const T,

    /// <https://bsky.app/profile/did:plc:yood7rhvorqjgyvlileb5jco/post/3mnctzdqffs2o>
    _of_the_opera: PhantomData<&'a [T]>,
}

impl<'a, T> RawSliceIter<'a, T> {
    /// Helper for checking if `T` is a `ZST`, similar to how `core` has `SizedTypeProperties`.
    const ZST: bool = size_of::<T>() == 0;

    /// Creates a new slice iter from a pointer to a slice
    ///
    /// # Safety
    ///
    /// The pointer and its [length][NonNull::len] must satisfy the safety guarantees of [`NonNull::add`].
    ///
    /// In particular, if any of the following conditions are violated, the result is Undefined Behavior:
    /// - The computed offset, `len * size_of::<T>()` bytes, must not overflow `isize`.
    /// - If the computed offset is non-zero, then self must be derived from a pointer to some allocation
    ///   and the entire memory range between self and the result must be in bounds of that allocation.
    ///   In particular, this range must not “wrap around” the edge of the address space.
    ///
    /// Allocations can never be larger than `isize::MAX` bytes, so if the computed offset stays in bounds
    /// of the allocation, it is guaranteed to satisfy the first requirement.
    ///
    /// **Additionally** it is undefined behavior if any of the methods (even safe ones!) are called
    /// after the allocation that `slice` points to has been deallocated. That is, the above safety
    /// requirements needs to be upheld while `RawSliceIter` is in use.
    ///
    /// The lifetime of the `RawSliceIter` might help with that, although note that this method
    /// does not bound the lifetime in anyway, so the caller is responsible to make sure it's correct.
    #[inline]
    pub const unsafe fn new(slice: NonNull<[T]>) -> Self {
        let ptr: NonNull<T> = slice.cast();
        let len = slice.len();

        unsafe {
            let end_or_len =
                if Self::ZST { ptr::without_provenance(len) } else { ptr.as_ptr().add(len) };

            Self { ptr, end_or_len, _of_the_opera: PhantomData }
        }
    }

    pub const fn from_ref(slice: &'a [T]) -> Self {
        // This should just be `NonNull::from(slice)`, but it's not const stable :(
        //
        // Safety:
        //
        // - The pointer & length come from a slice, guaranteeing that everything is in bounds of
        //   an allocation
        // - The input slice has the same lifetime as the output iterator, so it's guaranteed that
        //   the allocation will live long enough for any call to a method on an iterator
        unsafe { Self::new(NonNull::new(ptr::from_ref(slice).cast_mut()).unwrap()) }
    }

    pub const fn from_mut(slice: &'a mut [T]) -> Self {
        // This should just be `NonNull::from(slice)`, but it's not const stable :(
        //
        // Safety:
        //
        // - The pointer & length come from a slice, guaranteeing that everything is in bounds of
        //   an allocation
        // - The input slice has the same lifetime as the output iterator, so it's guaranteed that
        //   the allocation will live long enough for any call to a method on an iterator
        unsafe { Self::new(NonNull::new(ptr::from_mut(slice)).unwrap()) }
    }

    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> NonNull<[T]> {
        let len = unsafe {
            if Self::ZST {
                self.end_or_len.addr()
            } else {
                self.end_or_len.offset_from_unsigned(self.ptr.as_ptr())
            }
        };

        NonNull::slice_from_raw_parts(self.ptr, len)
    }

    #[inline(always)]
    fn zst<R>(
        &self,
        zst: impl FnOnce(NonNull<T>, usize) -> R,
        non_zst: impl FnOnce(NonNull<T>, NonNull<T>) -> R,
    ) -> R {
        if Self::ZST {
            zst(self.ptr, self.end_or_len.addr())
        } else {
            non_zst(self.ptr, unsafe { NonNull::new_unchecked(self.end_or_len.cast_mut()) })
        }
    }

    #[inline(always)]
    fn zst_mut<R>(
        &mut self,
        zst: impl FnOnce(&mut NonNull<T>, &mut usize) -> R,
        non_zst: impl FnOnce(&mut NonNull<T>, &mut NonNull<T>) -> R,
    ) -> R {
        if Self::ZST {
            zst(&mut self.ptr, unsafe { &mut *ptr::from_mut(&mut self.end_or_len).cast::<usize>() })
        } else {
            non_zst(&mut self.ptr, unsafe {
                &mut *ptr::from_mut(&mut self.end_or_len).cast::<NonNull<T>>()
            })
        }
    }
}

impl<T> Clone for RawSliceIter<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        RawSliceIter { ..*self }
    }
}

impl<'a, T> RawSliceIter<'a, T> {
    /// Returns the last element and moves the end of the iterator backwards by 1.
    ///
    /// # Safety
    ///
    /// The iterator must not be empty
    #[inline]
    unsafe fn next_back_unchecked(&mut self) -> NonNull<T> {
        unsafe { self.pre_dec_end(1) }
    }

    /// Helper function for moving the start of the iterator forwards by `offset` elements,
    /// returning the old start.
    /// Unsafe because the offset must not exceed `self.len()`.
    #[inline(always)]
    unsafe fn post_inc_start(&mut self, offset: usize) -> NonNull<T> {
        let old = self.ptr;

        // SAFETY: the caller guarantees that `offset` doesn't exceed `self.len()`,
        // so this new pointer is inside `self` and thus guaranteed to be non-null.
        unsafe {
            self.zst_mut(
                |_, len| *len = len.unchecked_sub(offset),
                |start, _| *start = start.add(offset),
            );
        }

        old
    }

    /// Helper function for moving the end of the iterator backwards by `offset` elements,
    /// returning the new end.
    /// Unsafe because the offset must not exceed `self.len()`.
    #[inline(always)]
    unsafe fn pre_dec_end(&mut self, offset: usize) -> NonNull<T> {
        self.zst_mut(
            |ptr, len| unsafe {
                *len = len.unchecked_sub(offset);
                *ptr
            },
            |_, end| unsafe {
                *end = end.sub(offset);
                *end
            },
        )
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.zst(|_, len| len == 0, |start, end| start == end)
    }
}

impl<T> ExactSizeIterator for RawSliceIter<'_, T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.zst(|_, len| len, |start, end| unsafe { end.offset_from_unsigned(start) })
    }
}

impl<'a, T> Iterator for RawSliceIter<'a, T> {
    type Item = NonNull<T>;

    #[inline]
    fn next(&mut self) -> Option<NonNull<T>> {
        // intentionally not using the helpers because this is
        // one of the most mono'd things in the library.

        let ptr = self.ptr;
        let end_or_len = self.end_or_len;

        // SAFETY: See inner comments. (For some reason having multiple
        // block breaks inlining this -- if you can fix that please do!)
        unsafe {
            if Self::ZST {
                let len = end_or_len.addr();
                if len == 0 {
                    return None;
                }
                // SAFETY: just checked that it's not zero, so subtracting one
                // cannot wrap.  (Ideally this would be `checked_sub`, which
                // does the same thing internally, but as of 2025-02 that
                // doesn't optimize quite as small in MIR.)
                self.end_or_len = ptr::without_provenance(len.unchecked_sub(1));
            } else {
                // SAFETY: by type invariant, the `end_or_len` field is always
                // non-null for a non-ZST pointee.  (This transmute ensures we
                // get `!nonnull` metadata on the load of the field.)
                if ptr == core::mem::transmute::<*const T, NonNull<T>>(end_or_len) {
                    return None;
                }
                // SAFETY: since it's not empty, per the check above, moving
                // forward one keeps us inside the slice, and this is valid.
                self.ptr = ptr.add(1);
            }

            Some(ptr)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.len();
        (exact, Some(exact))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }
}

impl<'a, T> DoubleEndedIterator for RawSliceIter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<NonNull<T>> {
        // SAFETY: The call to `next_back_unchecked`
        // is safe since we check if the iterator is empty first.
        #[expect(unstable_name_collisions)]
        unsafe {
            if self.is_empty() { None } else { Some(self.next_back_unchecked()) }
        }
    }
}

impl<T> FusedIterator for RawSliceIter<'_, T> {}

impl<T> Default for RawSliceIter<'_, T> {
    /// Creates an empty raw slice iterator.
    fn default() -> Self {
        Self::from_ref(&[])
    }
}
