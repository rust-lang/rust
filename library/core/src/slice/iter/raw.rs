#![unstable(feature = "raw_slice_iter", issue = "none")]
#![expect(missing_docs)]
use crate::hint::assert_unchecked;
use crate::iter::{FusedIterator, TrustedLen};
use crate::marker::PhantomData;
use crate::mem::SizedTypeProperties;
use crate::num::NonZero;
use crate::ptr::NonNull;
use crate::{cmp, fmt};

mod end_or_len;
use end_or_len::{End, EndOrLenRepr, Len};

/// A version of [`core::slice::Iter`] that works on raw pointers.
/// This should really be part of `core`...
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct IterRaw<'a, T> {
    /// The pointer to the next element to return, or the past-the-end location
    /// if the iterator is empty.
    ///
    /// This address will be used for all ZST elements, never changed.
    ptr: NonNull<T>,
    /// For non-ZSTs, the non-null pointer to the past-the-end element.
    ///
    /// For ZSTs, this is `ptr::without_provenance_mut(len)`.
    end_or_len: EndOrLenRepr<T>,

    /// <https://bsky.app/profile/did:plc:yood7rhvorqjgyvlileb5jco/post/3mnctzdqffs2o>
    _of_the_opera: PhantomData<&'a [T]>,
}

impl<'a, T> IterRaw<'a, T> {
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
    #[rustc_force_inline]
    #[rustc_const_unstable(feature = "raw_slice_iter", issue = "none")]
    #[rustc_const_stable_indirect]
    pub const unsafe fn new(slice: NonNull<[T]>) -> IterRaw<'a, T> {
        let ptr: NonNull<T> = slice.cast();
        let len = slice.len();

        unsafe {
            let end_or_len = EndOrLenRepr::new(len, ptr.add(len));
            IterRaw { ptr, end_or_len, _of_the_opera: PhantomData }
        }
    }

    #[rustc_const_unstable(feature = "raw_slice_iter", issue = "none")]
    #[rustc_const_stable_indirect]
    pub const fn from_ref(slice: &'a [T]) -> IterRaw<'a, T> {
        // Safety:
        //
        // - The pointer & length come from a slice, guaranteeing that everything is in bounds of
        //   an allocation
        // - The input slice has the same lifetime as the output iterator, so it's guaranteed that
        //   the allocation will live long enough for any call to a method on an iterator
        unsafe { IterRaw::new(NonNull::from_ref(slice)) }
    }

    #[rustc_const_unstable(feature = "raw_slice_iter", issue = "none")]
    #[rustc_const_stable_indirect]
    pub const fn from_mut(slice: &'a mut [T]) -> IterRaw<'a, T> {
        // Safety:
        //
        // - The pointer & length come from a slice, guaranteeing that everything is in bounds of
        //   an allocation
        // - The input slice has the same lifetime as the output iterator, so it's guaranteed that
        //   the allocation will live long enough for any call to a method on an iterator
        unsafe { IterRaw::new(NonNull::from_mut(slice)) }
    }

    #[must_use]
    #[rustc_force_inline]
    pub fn as_slice(&self) -> NonNull<[T]> {
        NonNull::slice_from_raw_parts(self.ptr, self.len())
    }
}

impl<T> Clone for IterRaw<'_, T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        IterRaw { ..*self }
    }
}

impl<'a, T> IterRaw<'a, T> {
    /// Returns the last element and moves the end of the iterator backwards by 1.
    ///
    /// # Safety
    ///
    /// The iterator must not be empty
    #[rustc_force_inline]
    unsafe fn next_back_unchecked(&mut self) -> NonNull<T> {
        unsafe { self.pre_dec_end(1) }
    }

    /// Helper function for moving the start of the iterator forwards by `offset` elements,
    /// returning the old start.
    /// Unsafe because the offset must not exceed `self.len()`.
    #[rustc_force_inline]
    unsafe fn post_inc_start(&mut self, offset: usize) -> NonNull<T> {
        let old = self.ptr;

        // SAFETY: the caller guarantees that `offset` doesn't exceed `self.len()`,
        // so this new pointer is inside `self` and thus guaranteed to be non-null.
        unsafe {
            match self.end_or_len.view_mut() {
                End(_) => self.ptr = self.ptr.add(offset),
                Len(len) => *len = len.unchecked_sub(offset),
            }
        }

        old
    }

    /// Helper function for moving the end of the iterator backwards by `offset` elements,
    /// returning the new end.
    /// Unsafe because the offset must not exceed `self.len()`.
    #[rustc_force_inline]
    unsafe fn pre_dec_end(&mut self, offset: usize) -> NonNull<T> {
        match self.end_or_len.view_mut() {
            End(end) => unsafe {
                *end = end.sub(offset);
                *end
            },
            Len(len) => unsafe {
                *len = len.unchecked_sub(offset);
                self.ptr
            },
        }
    }
}

impl<T> ExactSizeIterator for IterRaw<'_, T> {
    #[inline(always)]
    fn len(&self) -> usize {
        match self.end_or_len.view() {
            End(end) => unsafe { end.offset_from_unsigned(self.ptr) },
            Len(len) => len,
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        match self.end_or_len.view() {
            End(end) => self.ptr == end,
            Len(len) => len == 0,
        }
    }
}

impl<'a, T> IterRaw<'a, T> {
    #[rustc_force_inline]
    pub(crate) fn next(&mut self) -> Option<NonNull<T>> {
        let ptr = self.ptr;

        // SAFETY: See inner comments. (For some reason having multiple
        // block breaks inlining this -- if you can fix that please do!)
        unsafe {
            match self.end_or_len.view_mut() {
                End(end) => {
                    if ptr == *end {
                        return None;
                    }

                    // SAFETY: since it's not empty, per the check above, moving
                    // forward one keeps us inside the slice, and this is valid.
                    self.ptr = ptr.add(1);
                }
                Len(len) => {
                    if *len == 0 {
                        return None;
                    }

                    // SAFETY: just checked that it's not zero, so subtracting one
                    // cannot wrap.  (Ideally this would be `checked_sub`, which
                    // does the same thing internally, but as of 2025-02 that
                    // doesn't optimize quite as small in MIR.)
                    *len = len.unchecked_sub(1);
                }
            }

            Some(ptr)
        }
    }

    #[rustc_force_inline]
    pub(crate) fn next_chunk<const N: usize>(
        &mut self,
    ) -> Result<[NonNull<T>; N], crate::array::IntoIter<NonNull<T>, N>> {
        if T::IS_ZST || self.len() < N {
            return crate::array::iter_next_chunk(self);
        }

        unsafe {
            let r = self
                // SAFETY: the check above ensures len >= N
                .post_inc_start(N)
                .cast_array::<N>() // NonNull<T> -> NonNull<[T; N]>
                .each_nonnull(); // NonNull<[T; N]> -> [NonNull<T>; N]

            Ok(r)
        }
    }

    #[rustc_force_inline]
    pub(crate) fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.len();
        (exact, Some(exact))
    }

    #[rustc_force_inline]
    pub(crate) fn count(self) -> usize {
        self.len()
    }

    #[rustc_force_inline]
    pub(crate) fn nth(&mut self, n: usize) -> Option<NonNull<T>> {
        if n >= self.len() {
            match self.end_or_len.view_mut() {
                End(end) => self.ptr = *end,
                Len(len) => *len = 0,
            }
            return None;
        }

        // SAFETY: We are in bounds. `post_inc_start` does the right thing even for ZSTs.
        unsafe {
            self.post_inc_start(n);
            Some(self.post_inc_start(1))
        }
    }

    #[rustc_force_inline]
    pub(crate) fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        let advance = cmp::min(self.len(), n);
        // SAFETY: By construction, `advance` does not exceed `self.len()`.
        unsafe { self.post_inc_start(advance) };
        NonZero::new(n - advance).map_or(Ok(()), Err)
    }

    #[rustc_force_inline]
    pub(crate) fn last(mut self) -> Option<NonNull<T>> {
        self.next_back()
    }

    #[rustc_force_inline]
    pub(crate) fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, NonNull<T>) -> B,
    {
        // this implementation consists of the following optimizations compared to the
        // default implementation:
        // - do-while loop, as is llvm's preferred loop shape,
        //   see https://releases.llvm.org/16.0.0/docs/LoopTerminology.html#more-canonical-loops
        // - bumps an index instead of a pointer since the latter case inhibits
        //   some optimizations, see #111603
        // - avoids Option wrapping/matching
        if self.is_empty() {
            return init;
        }
        let mut acc = init;
        let mut i = 0usize;
        let len = self.len();
        loop {
            // SAFETY: the loop iterates `i in 0..len`, which always is in bounds of
            // the slice allocation
            acc = f(acc, unsafe { self.ptr.add(i) });
            // SAFETY: `i` can't overflow since it'll only reach usize::MAX if the
            // slice had that length, in which case we'll break out of the loop
            // after the increment
            i = unsafe { i.unchecked_add(1) };
            if i == len {
                break;
            }
        }
        acc
    }

    // We override the default implementation, which uses `try_fold`,
    // because this simple implementation generates less LLVM IR and is
    // faster to compile.
    #[rustc_force_inline]
    pub(crate) fn for_each<F>(mut self, mut f: F)
    where
        Self: Sized,
        F: FnMut(NonNull<T>),
    {
        while let Some(x) = self.next() {
            f(x);
        }
    }

    // We override the default implementation, which uses `try_fold`,
    // because this simple implementation generates less LLVM IR and is
    // faster to compile.
    #[rustc_force_inline]
    pub(crate) fn all<F>(&mut self, mut f: F) -> bool
    where
        Self: Sized,
        F: FnMut(NonNull<T>) -> bool,
    {
        while let Some(x) = self.next() {
            if !f(x) {
                return false;
            }
        }
        true
    }

    // We override the default implementation, which uses `try_fold`,
    // because this simple implementation generates less LLVM IR and is
    // faster to compile.
    #[rustc_force_inline]
    pub(crate) fn any<F>(&mut self, mut f: F) -> bool
    where
        Self: Sized,
        F: FnMut(NonNull<T>) -> bool,
    {
        while let Some(x) = self.next() {
            if f(x) {
                return true;
            }
        }
        false
    }

    // We override the default implementation, which uses `try_fold`,
    // because this simple implementation generates less LLVM IR and is
    // faster to compile.
    #[rustc_force_inline]
    pub(crate) fn find<P>(&mut self, mut predicate: P) -> Option<NonNull<T>>
    where
        Self: Sized,
        P: FnMut(&NonNull<T>) -> bool,
    {
        while let Some(x) = self.next() {
            if predicate(&x) {
                return Some(x);
            }
        }
        None
    }

    // We override the default implementation, which uses `try_fold`,
    // because this simple implementation generates less LLVM IR and is
    // faster to compile.
    #[rustc_force_inline]
    pub(crate) fn find_map<B, F>(&mut self, mut f: F) -> Option<B>
    where
        Self: Sized,
        F: FnMut(NonNull<T>) -> Option<B>,
    {
        while let Some(x) = self.next() {
            if let Some(y) = f(x) {
                return Some(y);
            }
        }
        None
    }

    // We override the default implementation, which uses `try_fold`,
    // because this simple implementation generates less LLVM IR and is
    // faster to compile. Also, the `assume` avoids a bounds check.
    #[rustc_force_inline]
    pub(crate) fn position<P>(&mut self, mut predicate: P) -> Option<usize>
    where
        Self: Sized,
        P: FnMut(NonNull<T>) -> bool,
    {
        let n = self.len();
        let mut i = 0;
        while let Some(x) = self.next() {
            if predicate(x) {
                // SAFETY: we are guaranteed to be in bounds by the loop invariant:
                // when `i >= n`, `self.next()` returns `None` and the loop breaks.
                unsafe { assert_unchecked(i < n) };
                return Some(i);
            }
            i += 1;
        }
        None
    }

    // We override the default implementation, which uses `try_fold`,
    // because this simple implementation generates less LLVM IR and is
    // faster to compile. Also, the `assume` avoids a bounds check.
    #[rustc_force_inline]
    pub(crate) fn rposition<P>(&mut self, mut predicate: P) -> Option<usize>
    where
        P: FnMut(NonNull<T>) -> bool,
        // Self: Sized + ExactSizeIterator + DoubleEndedIterator,
    {
        let n = self.len();
        let mut i = n;
        while let Some(x) = self.next_back() {
            i -= 1;
            if predicate(x) {
                // SAFETY: `i` must be lower than `n` since it starts at `n`
                // and is only decreasing.
                unsafe { assert_unchecked(i < n) };
                return Some(i);
            }
        }
        None
    }

    #[rustc_force_inline]
    pub(crate) unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> NonNull<T> {
        // SAFETY: the caller must guarantee that `i` is in bounds of
        // the underlying slice, so `i` cannot overflow an `isize`, and
        // the returned references is guaranteed to refer to an element
        // of the slice and thus guaranteed to be valid.
        //
        // Also note that the caller also guarantees that we're never
        // called with the same index again, and that no other methods
        // that will access this subslice are called, so it is valid
        // for the returned reference to be mutable in the case of
        // `IterMut`
        unsafe { self.ptr.add(idx) }
    }
}

impl<'a, T> Iterator for IterRaw<'a, T> {
    type Item = NonNull<T>;

    #[inline(always)]
    fn next(&mut self) -> Option<NonNull<T>> {
        let ptr = self.ptr;

        // SAFETY: See inner comments. (For some reason having multiple
        // block breaks inlining this -- if you can fix that please do!)
        unsafe {
            match self.end_or_len.view_mut() {
                End(end) => {
                    if ptr == *end {
                        return None;
                    }

                    // SAFETY: since it's not empty, per the check above, moving
                    // forward one keeps us inside the slice, and this is valid.
                    self.ptr = ptr.add(1);
                }
                Len(len) => {
                    if *len == 0 {
                        return None;
                    }

                    // SAFETY: just checked that it's not zero, so subtracting one
                    // cannot wrap.  (Ideally this would be `checked_sub`, which
                    // does the same thing internally, but as of 2025-02 that
                    // doesn't optimize quite as small in MIR.)
                    *len = len.unchecked_sub(1);
                }
            }

            Some(ptr)
        }
    }

    fn next_chunk<const N: usize>(
        &mut self,
    ) -> Result<[NonNull<T>; N], crate::array::IntoIter<NonNull<T>, N>> {
        if T::IS_ZST || self.len() < N {
            return crate::array::iter_next_chunk(self);
        }

        unsafe {
            let r = self
                // SAFETY: the check above ensures len >= N
                .post_inc_start(N)
                .cast_array::<N>() // NonNull<T> -> NonNull<[T; N]>
                .each_nonnull(); // NonNull<[T; N]> -> [NonNull<T>; N]

            Ok(r)
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.len();
        (exact, Some(exact))
    }

    #[inline(always)]
    fn count(self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn nth(&mut self, n: usize) -> Option<NonNull<T>> {
        if n >= self.len() {
            match self.end_or_len.view_mut() {
                End(end) => self.ptr = *end,
                Len(len) => *len = 0,
            }
            return None;
        }

        // SAFETY: We are in bounds. `post_inc_start` does the right thing even for ZSTs.
        unsafe {
            self.post_inc_start(n);
            Some(self.post_inc_start(1))
        }
    }

    #[inline(always)]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        let advance = cmp::min(self.len(), n);
        // SAFETY: By construction, `advance` does not exceed `self.len()`.
        unsafe { self.post_inc_start(advance) };
        NonZero::new(n - advance).map_or(Ok(()), Err)
    }

    #[inline(always)]
    fn last(mut self) -> Option<NonNull<T>> {
        self.next_back()
    }

    #[inline(always)]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        // this implementation consists of the following optimizations compared to the
        // default implementation:
        // - do-while loop, as is llvm's preferred loop shape,
        //   see https://releases.llvm.org/16.0.0/docs/LoopTerminology.html#more-canonical-loops
        // - bumps an index instead of a pointer since the latter case inhibits
        //   some optimizations, see #111603
        // - avoids Option wrapping/matching
        if self.is_empty() {
            return init;
        }
        let mut acc = init;
        let mut i = 0usize;
        let len = self.len();
        loop {
            // SAFETY: the loop iterates `i in 0..len`, which always is in bounds of
            // the slice allocation
            acc = f(acc, unsafe { self.ptr.add(i) });
            // SAFETY: `i` can't overflow since it'll only reach usize::MAX if the
            // slice had that length, in which case we'll break out of the loop
            // after the increment
            i = unsafe { i.unchecked_add(1) };
            if i == len {
                break;
            }
        }
        acc
    }

    // We override the default implementation, which uses `try_fold`,
    // because this simple implementation generates less LLVM IR and is
    // faster to compile.
    #[inline(always)]
    fn for_each<F>(mut self, mut f: F)
    where
        Self: Sized,
        F: FnMut(Self::Item),
    {
        while let Some(x) = self.next() {
            f(x);
        }
    }

    // We override the default implementation, which uses `try_fold`,
    // because this simple implementation generates less LLVM IR and is
    // faster to compile.
    #[inline(always)]
    fn all<F>(&mut self, mut f: F) -> bool
    where
        Self: Sized,
        F: FnMut(Self::Item) -> bool,
    {
        while let Some(x) = self.next() {
            if !f(x) {
                return false;
            }
        }
        true
    }

    // We override the default implementation, which uses `try_fold`,
    // because this simple implementation generates less LLVM IR and is
    // faster to compile.
    #[inline(always)]
    fn any<F>(&mut self, mut f: F) -> bool
    where
        Self: Sized,
        F: FnMut(Self::Item) -> bool,
    {
        while let Some(x) = self.next() {
            if f(x) {
                return true;
            }
        }
        false
    }

    // We override the default implementation, which uses `try_fold`,
    // because this simple implementation generates less LLVM IR and is
    // faster to compile.
    #[inline(always)]
    fn find<P>(&mut self, mut predicate: P) -> Option<Self::Item>
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        while let Some(x) = self.next() {
            if predicate(&x) {
                return Some(x);
            }
        }
        None
    }

    // We override the default implementation, which uses `try_fold`,
    // because this simple implementation generates less LLVM IR and is
    // faster to compile.
    #[inline(always)]
    fn find_map<B, F>(&mut self, mut f: F) -> Option<B>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> Option<B>,
    {
        while let Some(x) = self.next() {
            if let Some(y) = f(x) {
                return Some(y);
            }
        }
        None
    }

    // We override the default implementation, which uses `try_fold`,
    // because this simple implementation generates less LLVM IR and is
    // faster to compile. Also, the `assume` avoids a bounds check.
    #[inline(always)]
    fn position<P>(&mut self, mut predicate: P) -> Option<usize>
    where
        Self: Sized,
        P: FnMut(Self::Item) -> bool,
    {
        let n = self.len();
        let mut i = 0;
        while let Some(x) = self.next() {
            if predicate(x) {
                // SAFETY: we are guaranteed to be in bounds by the loop invariant:
                // when `i >= n`, `self.next()` returns `None` and the loop breaks.
                unsafe { assert_unchecked(i < n) };
                return Some(i);
            }
            i += 1;
        }
        None
    }

    // We override the default implementation, which uses `try_fold`,
    // because this simple implementation generates less LLVM IR and is
    // faster to compile. Also, the `assume` avoids a bounds check.
    #[inline(always)]
    fn rposition<P>(&mut self, mut predicate: P) -> Option<usize>
    where
        P: FnMut(Self::Item) -> bool,
        Self: Sized + ExactSizeIterator + DoubleEndedIterator,
    {
        let n = self.len();
        let mut i = n;
        while let Some(x) = self.next_back() {
            i -= 1;
            if predicate(x) {
                // SAFETY: `i` must be lower than `n` since it starts at `n`
                // and is only decreasing.
                unsafe { assert_unchecked(i < n) };
                return Some(i);
            }
        }
        None
    }

    #[inline(always)]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        // SAFETY: the caller must guarantee that `i` is in bounds of
        // the underlying slice, so `i` cannot overflow an `isize`, and
        // the returned references is guaranteed to refer to an element
        // of the slice and thus guaranteed to be valid.
        //
        // Also note that the caller also guarantees that we're never
        // called with the same index again, and that no other methods
        // that will access this subslice are called, so it is valid
        // for the returned reference to be mutable in the case of
        // `IterMut`
        unsafe { self.ptr.add(idx) }
    }

    // FIXME: override is_sorted_by?
}

impl<'a, T> DoubleEndedIterator for IterRaw<'a, T> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<NonNull<T>> {
        // SAFETY: The call to `next_back_unchecked`
        // is safe since we check if the iterator is empty first.
        unsafe { if self.is_empty() { None } else { Some(self.next_back_unchecked()) } }
    }

    #[inline(always)]
    fn nth_back(&mut self, n: usize) -> Option<NonNull<T>> {
        if n >= self.len() {
            match self.end_or_len.view_mut() {
                End(end) => self.ptr = *end,
                Len(len) => *len = 0,
            }
            return None;
        }

        // SAFETY: We are in bounds. `pre_dec_end` does the right thing even for ZSTs.
        unsafe {
            self.pre_dec_end(n);
            Some(self.next_back_unchecked())
        }
    }

    #[inline(always)]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        let advance = cmp::min(self.len(), n);
        // SAFETY: By construction, `advance` does not exceed `self.len()`.
        unsafe { self.pre_dec_end(advance) };
        NonZero::new(n - advance).map_or(Ok(()), Err)
    }
}

impl<T> FusedIterator for IterRaw<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for IterRaw<'_, T> {}

impl<T> Default for IterRaw<'_, T> {
    /// Creates an empty raw slice iterator.
    fn default() -> Self {
        IterRaw::from_ref(&[])
    }
}

impl<T> fmt::Debug for IterRaw<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("slice::IterRaw");

        match self.end_or_len.view() {
            End(end) => s.field("start", &self.ptr).field("end", &end),
            Len(len) => s.field("ptr", &self.ptr).field("len", &len),
        };

        s.finish()
    }
}
