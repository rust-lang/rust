//! Macros used by iterators of slice.

/// Convenience & performance macro for consuming the `end_or_len` field, by
/// giving a `(&mut) usize` or `(&mut) NonNull<T>` depending whether `T` is
/// or is not a ZST respectively.
///
/// Internally, this reads the `end` through a pointer-to-`NonNull` so that
/// it'll get the appropriate non-null metadata in the backend without needing
/// to call `assume` manually.
macro_rules! if_zst {
    (mut $this:ident, $len:ident => $zst_body:expr, $end:ident => $other_body:expr,) => {{
        #![allow(unused_unsafe)] // we're sometimes used within an unsafe block

        if T::IS_ZST {
            // SAFETY: for ZSTs, the pointer is storing a provenance-free length,
            // so consuming and updating it as a `usize` is fine.
            let $len = unsafe { &mut *ptr::addr_of_mut!($this.end_or_len).cast::<usize>() };
            $zst_body
        } else {
            // SAFETY: for non-ZSTs, the type invariant ensures it cannot be null
            let $end = unsafe { &mut *ptr::addr_of_mut!($this.end_or_len).cast::<NonNull<T>>() };
            $other_body
        }
    }};
    ($this:ident, $len:ident => $zst_body:expr, $end:ident => $other_body:expr,) => {{
        #![allow(unused_unsafe)] // we're sometimes used within an unsafe block

        if T::IS_ZST {
            let $len = $this.end_or_len.addr();
            $zst_body
        } else {
            // SAFETY: for non-ZSTs, the type invariant ensures it cannot be null
            let $end = unsafe { *ptr::addr_of!($this.end_or_len).cast::<NonNull<T>>() };
            $other_body
        }
    }};
}

// Inlining is_empty and len makes a huge performance difference
macro_rules! is_empty {
    ($self: ident) => {
        if_zst!($self,
            len => len == 0,
            end => $self.ptr == end,
        )
    };
}

macro_rules! len {
    ($self: ident) => {{
        if_zst!($self,
            len => len,
            end => {
                // To get rid of some bounds checks (see `position`), we use ptr_sub instead of
                // offset_from (Tested by `codegen/slice-position-bounds-check`.)
                // SAFETY: by the type invariant pointers are aligned and `start <= end`
                unsafe { end.sub_ptr($self.ptr) }
            },
        )
    }};
}

// The shared definition of the `Iter` and `IterMut` iterators
macro_rules! iterator {
    (
        struct $name:ty => $ptr:ty,
        $elem:ty,
        {$($extra:tt)*}
    ) => {
        #[allow(unused_lifetimes)]
        impl<'a, T> $name {
            /// Returns the last element and moves the end of the iterator backwards by 1.
            ///
            /// # Safety
            ///
            /// The iterator must not be empty
            #[inline]
            unsafe fn next_back_unchecked(&mut self) -> $elem {
                // SAFETY: the caller promised it's not empty, so
                // the offsetting is in-bounds and there's an element to return.
                unsafe { Self::non_null_to_item(self.pre_dec_end(1)) }
            }

            // Helper function for creating a slice from the iterator.
            #[inline]
            pub(crate) fn make_nonnull_slice(&self) -> NonNull<[T]> {
                NonNull::slice_from_raw_parts(self.ptr, len!(self))
            }

            #[inline]
            pub(crate) fn make_shortlived_slice<'b>(&'b self) -> &'b [T] {
                // SAFETY: Everything expanded with this macro is readable while
                // the iterator exists and is unchanged, so by tying this to the
                // shorter-than-`'a` self borrow we can make this safe to call.
                // (Elision would be fine here, but using `'b` for emphasis.)
                unsafe { self.make_nonnull_slice().as_ref() }
            }

            // Helper function for moving the start of the iterator forwards by `offset` elements,
            // returning the old start.
            // Unsafe because the offset must not exceed `self.len()`.
            #[inline(always)]
            unsafe fn post_inc_start(&mut self, offset: usize) -> NonNull<T> {
                let old = self.ptr;

                // SAFETY: the caller guarantees that `offset` doesn't exceed `self.len()`,
                // so this new pointer is inside `self` and thus guaranteed to be non-null.
                unsafe {
                    if_zst!(mut self,
                        // Using the intrinsic directly avoids emitting a UbCheck
                        len => *len = crate::intrinsics::unchecked_sub(*len, offset),
                        _end => self.ptr = self.ptr.add(offset),
                    );
                }
                old
            }

            // Helper function for moving the end of the iterator backwards by `offset` elements,
            // returning the new end.
            // Unsafe because the offset must not exceed `self.len()`.
            #[inline(always)]
            unsafe fn pre_dec_end(&mut self, offset: usize) -> NonNull<T> {
                if_zst!(mut self,
                    // SAFETY: By our precondition, `offset` can be at most the
                    // current length, so the subtraction can never overflow.
                    len => unsafe {
                        // Using the intrinsic directly avoids emitting a UbCheck
                        *len = crate::intrinsics::unchecked_sub(*len, offset);
                        self.ptr
                    },
                    // SAFETY: the caller guarantees that `offset` doesn't exceed `self.len()`,
                    // which is guaranteed to not overflow an `isize`. Also, the resulting pointer
                    // is in bounds of `slice`, which fulfills the other requirements for `offset`.
                    end => unsafe {
                        *end = end.sub(offset);
                        *end
                    },
                )
            }

            // This is not used on every type that uses this macro, but is more
            // convenient to implement here so it can use `post_inc_start`.
            #[allow(dead_code)]
            #[inline]
            pub(crate) unsafe fn skip_forward_unchecked(&mut self, offset: usize) -> NonNull<[T]> {
                // SAFETY: The caller guarantees the provided offset is in-bounds.
                let old_begin = unsafe { self.post_inc_start(offset) };
                NonNull::slice_from_raw_parts(old_begin, offset)
            }

            // This is not used on every type that uses this macro, but is more
            // convenient to implement here so it can use `pre_dec_end`.
            #[allow(dead_code)]
            #[inline]
            pub(crate) unsafe fn skip_backward_unchecked(&mut self, offset: usize) -> NonNull<[T]> {
                // SAFETY: The caller guarantees the provided offset is in-bounds.
                let new_end = unsafe { self.pre_dec_end(offset) };
                NonNull::slice_from_raw_parts(new_end, offset)
            }
        }

        #[allow(unused_lifetimes)]
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, T> ExactSizeIterator for $name {
            #[inline(always)]
            fn len(&self) -> usize {
                len!(self)
            }

            #[inline(always)]
            fn is_empty(&self) -> bool {
                is_empty!(self)
            }
        }

        #[allow(unused_lifetimes)]
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, T> Iterator for $name {
            type Item = $elem;

            #[inline]
            fn next(&mut self) -> Option<$elem> {
                // could be implemented with slices, but this avoids bounds checks

                // SAFETY: The call to `next_unchecked` is
                // safe since we check if the iterator is empty first.
                unsafe {
                    if is_empty!(self) {
                        None
                    } else {
                        Some(self.next_unchecked())
                    }
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let exact = len!(self);
                (exact, Some(exact))
            }

            #[inline]
            fn count(self) -> usize {
                len!(self)
            }

            #[inline]
            fn nth(&mut self, n: usize) -> Option<$elem> {
                if n >= len!(self) {
                    // This iterator is now empty.
                    if_zst!(mut self,
                        len => *len = 0,
                        end => self.ptr = *end,
                    );
                    return None;
                }
                // SAFETY: We are in bounds. `post_inc_start` does the right thing even for ZSTs.
                unsafe {
                    self.post_inc_start(n);
                    Some(self.next_unchecked())
                }
            }

            #[inline]
            fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
                let advance = cmp::min(len!(self), n);
                // SAFETY: By construction, `advance` does not exceed `self.len()`.
                unsafe { self.post_inc_start(advance) };
                NonZero::new(n - advance).map_or(Ok(()), Err)
            }

            #[inline]
            fn last(mut self) -> Option<$elem> {
                self.next_back()
            }

            #[inline]
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
                if is_empty!(self) {
                    return init;
                }
                let mut acc = init;
                let mut i = 0;
                let len = len!(self);
                loop {
                    // SAFETY: the loop iterates `i in 0..len`, which always is in bounds of
                    // the slice allocation
                    acc = f(acc, unsafe { Self::non_null_to_item(self.ptr.add(i)) });
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
            #[inline]
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
            #[inline]
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
            #[inline]
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
            #[inline]
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
            #[inline]
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
            #[inline]
            #[rustc_inherit_overflow_checks]
            fn position<P>(&mut self, mut predicate: P) -> Option<usize> where
                Self: Sized,
                P: FnMut(Self::Item) -> bool,
            {
                let n = len!(self);
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
            #[inline]
            fn rposition<P>(&mut self, mut predicate: P) -> Option<usize> where
                P: FnMut(Self::Item) -> bool,
                Self: Sized + ExactSizeIterator + DoubleEndedIterator
            {
                let n = len!(self);
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

            #[inline]
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
                unsafe { Self::non_null_to_item(self.ptr.add(idx)) }
            }

            $($extra)*
        }

        #[allow(unused_lifetimes)]
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, T> DoubleEndedIterator for $name {
            #[inline]
            fn next_back(&mut self) -> Option<$elem> {
                // could be implemented with slices, but this avoids bounds checks

                // SAFETY: The call to `next_back_unchecked`
                // is safe since we check if the iterator is empty first.
                unsafe {
                    if is_empty!(self) {
                        None
                    } else {
                        Some(self.next_back_unchecked())
                    }
                }
            }

            #[inline]
            fn nth_back(&mut self, n: usize) -> Option<$elem> {
                if n >= len!(self) {
                    // This iterator is now empty.
                    if_zst!(mut self,
                        len => *len = 0,
                        end => *end = self.ptr,
                    );
                    return None;
                }
                // SAFETY: We are in bounds. `pre_dec_end` does the right thing even for ZSTs.
                unsafe {
                    self.pre_dec_end(n);
                    Some(self.next_back_unchecked())
                }
            }

            #[inline]
            fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
                let advance = cmp::min(len!(self), n);
                // SAFETY: By construction, `advance` does not exceed `self.len()`.
                unsafe { self.pre_dec_end(advance) };
                NonZero::new(n - advance).map_or(Ok(()), Err)
            }
        }

        #[allow(unused_lifetimes)]
        #[stable(feature = "fused", since = "1.26.0")]
        impl<'a, T> FusedIterator for $name {}

        #[allow(unused_lifetimes)]
        #[unstable(feature = "trusted_len", issue = "37572")]
        unsafe impl<'a, T> TrustedLen for $name {}

        #[allow(unused_lifetimes)]
        impl<'a, T> UncheckedIterator for $name {
            #[inline]
            unsafe fn next_unchecked(&mut self) -> $elem {
                // SAFETY: The caller promised there's at least one more item.
                unsafe {
                    Self::non_null_to_item(self.post_inc_start(1))
                }
            }
        }
    }
}

macro_rules! forward_iterator {
    ($name:ident: $elem:ident, $iter_of:ty) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, $elem, P> Iterator for $name<'a, $elem, P>
        where
            P: FnMut(&T) -> bool,
        {
            type Item = $iter_of;

            #[inline]
            fn next(&mut self) -> Option<$iter_of> {
                self.inner.next()
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.inner.size_hint()
            }
        }

        #[stable(feature = "fused", since = "1.26.0")]
        impl<'a, $elem, P> FusedIterator for $name<'a, $elem, P> where P: FnMut(&T) -> bool {}
    };
}
