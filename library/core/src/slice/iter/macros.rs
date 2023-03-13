//! Macros used by iterators of slice.

// Inlining is_empty and len makes a huge performance difference
macro_rules! is_empty {
    // The way we encode the length of a ZST iterator, this works both for ZST
    // and non-ZST.
    ($self: ident) => {
        $self.ptr.as_ptr() as *const T == $self.end
    };
}

macro_rules! len {
    ($self: ident) => {{
        #![allow(unused_unsafe)] // we're sometimes used within an unsafe block

        let start = $self.ptr;
        if T::IS_ZST {
            // This _cannot_ use `ptr_sub` because we depend on wrapping
            // to represent the length of long ZST slice iterators.
            $self.end.addr().wrapping_sub(start.as_ptr().addr())
        } else {
            // To get rid of some bounds checks (see `position`), we use ptr_sub instead of
            // offset_from (Tested by `codegen/slice-position-bounds-check`.)
            // SAFETY: by the type invariant pointers are aligned and `start <= end`
            unsafe { $self.end.sub_ptr(start.as_ptr()) }
        }
    }};
}

// The shared definition of the `Iter` and `IterMut` iterators
macro_rules! iterator {
    (
        struct $name:ident -> $ptr:ty,
        $elem:ty,
        $raw_mut:tt,
        {$( $mut_:tt )?},
        {$($extra:tt)*}
    ) => {
        // Returns the first element and moves the start of the iterator forwards by 1.
        // Greatly improves performance compared to an inlined function. The iterator
        // must not be empty.
        macro_rules! next_unchecked {
            ($self: ident) => {& $( $mut_ )? *$self.post_inc_start(1)}
        }

        // Returns the last element and moves the end of the iterator backwards by 1.
        // Greatly improves performance compared to an inlined function. The iterator
        // must not be empty.
        macro_rules! next_back_unchecked {
            ($self: ident) => {& $( $mut_ )? *$self.pre_dec_end(1)}
        }

        // Shrinks the iterator when T is a ZST, by moving the end of the iterator
        // backwards by `n`. `n` must not exceed `self.len()`.
        macro_rules! zst_shrink {
            ($self: ident, $n: ident) => {
                $self.end = $self.end.wrapping_byte_sub($n);
            }
        }

        impl<'a, T> $name<'a, T> {
            // Helper function for creating a slice from the iterator.
            #[inline(always)]
            fn make_slice(&self) -> &'a [T] {
                // SAFETY: the iterator was created from a slice with pointer
                // `self.ptr` and length `len!(self)`. This guarantees that all
                // the prerequisites for `from_raw_parts` are fulfilled.
                unsafe { from_raw_parts(self.ptr.as_ptr(), len!(self)) }
            }

            // Helper function for moving the start of the iterator forwards by `offset` elements,
            // returning the old start.
            // Unsafe because the offset must not exceed `self.len()`.
            #[inline(always)]
            unsafe fn post_inc_start(&mut self, offset: usize) -> * $raw_mut T {
                if mem::size_of::<T>() == 0 {
                    zst_shrink!(self, offset);
                    self.ptr.as_ptr()
                } else {
                    let old = self.ptr.as_ptr();
                    // SAFETY: the caller guarantees that `offset` doesn't exceed `self.len()`,
                    // so this new pointer is inside `self` and thus guaranteed to be non-null.
                    self.ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(offset)) };
                    old
                }
            }

            // Helper function for moving the end of the iterator backwards by `offset` elements,
            // returning the new end.
            // Unsafe because the offset must not exceed `self.len()`.
            #[inline(always)]
            unsafe fn pre_dec_end(&mut self, offset: usize) -> * $raw_mut T {
                if T::IS_ZST {
                    zst_shrink!(self, offset);
                    self.ptr.as_ptr()
                } else {
                    // SAFETY: the caller guarantees that `offset` doesn't exceed `self.len()`,
                    // which is guaranteed to not overflow an `isize`. Also, the resulting pointer
                    // is in bounds of `slice`, which fulfills the other requirements for `offset`.
                    self.end = unsafe { self.end.sub(offset) };
                    self.end
                }
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<T> ExactSizeIterator for $name<'_, T> {
            #[inline(always)]
            fn len(&self) -> usize {
                len!(self)
            }

            #[inline(always)]
            fn is_empty(&self) -> bool {
                is_empty!(self)
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, T> Iterator for $name<'a, T> {
            type Item = $elem;

            #[inline]
            fn next(&mut self) -> Option<$elem> {
                // could be implemented with slices, but this avoids bounds checks

                // SAFETY: `assume` calls are safe since a slice's start pointer
                // must be non-null, and slices over non-ZSTs must also have a
                // non-null end pointer. The call to `next_unchecked!` is safe
                // since we check if the iterator is empty first.
                unsafe {
                    assume(!self.ptr.as_ptr().is_null());
                    if !<T>::IS_ZST {
                        assume(!self.end.is_null());
                    }
                    if is_empty!(self) {
                        None
                    } else {
                        Some(next_unchecked!(self))
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
                    if T::IS_ZST {
                        // We have to do it this way as `ptr` may never be 0, but `end`
                        // could be (due to wrapping).
                        self.end = self.ptr.as_ptr();
                    } else {
                        // SAFETY: end can't be 0 if T isn't ZST because ptr isn't 0 and end >= ptr
                        unsafe {
                            self.ptr = NonNull::new_unchecked(self.end as *mut T);
                        }
                    }
                    return None;
                }
                // SAFETY: We are in bounds. `post_inc_start` does the right thing even for ZSTs.
                unsafe {
                    self.post_inc_start(n);
                    Some(next_unchecked!(self))
                }
            }

            #[inline]
            fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
                let advance = cmp::min(len!(self), n);
                // SAFETY: By construction, `advance` does not exceed `self.len()`.
                unsafe { self.post_inc_start(advance) };
                NonZeroUsize::new(n - advance).map_or(Ok(()), Err)
            }

            #[inline]
            fn last(mut self) -> Option<$elem> {
                self.next_back()
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
                        unsafe { assume(i < n) };
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
                        unsafe { assume(i < n) };
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
                unsafe { & $( $mut_ )? * self.ptr.as_ptr().add(idx) }
            }

            $($extra)*
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, T> DoubleEndedIterator for $name<'a, T> {
            #[inline]
            fn next_back(&mut self) -> Option<$elem> {
                // could be implemented with slices, but this avoids bounds checks

                // SAFETY: `assume` calls are safe since a slice's start pointer must be non-null,
                // and slices over non-ZSTs must also have a non-null end pointer.
                // The call to `next_back_unchecked!` is safe since we check if the iterator is
                // empty first.
                unsafe {
                    assume(!self.ptr.as_ptr().is_null());
                    if !<T>::IS_ZST {
                        assume(!self.end.is_null());
                    }
                    if is_empty!(self) {
                        None
                    } else {
                        Some(next_back_unchecked!(self))
                    }
                }
            }

            #[inline]
            fn nth_back(&mut self, n: usize) -> Option<$elem> {
                if n >= len!(self) {
                    // This iterator is now empty.
                    self.end = self.ptr.as_ptr();
                    return None;
                }
                // SAFETY: We are in bounds. `pre_dec_end` does the right thing even for ZSTs.
                unsafe {
                    self.pre_dec_end(n);
                    Some(next_back_unchecked!(self))
                }
            }

            #[inline]
            fn advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
                let advance = cmp::min(len!(self), n);
                // SAFETY: By construction, `advance` does not exceed `self.len()`.
                unsafe { self.pre_dec_end(advance) };
                NonZeroUsize::new(n - advance).map_or(Ok(()), Err)
            }
        }

        #[stable(feature = "fused", since = "1.26.0")]
        impl<T> FusedIterator for $name<'_, T> {}

        #[unstable(feature = "trusted_len", issue = "37572")]
        unsafe impl<T> TrustedLen for $name<'_, T> {}

        impl<'a, T> UncheckedIterator for $name<'a, T> {
            unsafe fn next_unchecked(&mut self) -> $elem {
                // SAFETY: The caller promised there's at least one more item.
                unsafe {
                    next_unchecked!(self)
                }
            }
        }

        #[stable(feature = "default_iters", since = "CURRENT_RUSTC_VERSION")]
        impl<T> Default for $name<'_, T> {
            /// Creates an empty slice iterator.
            ///
            /// ```
            #[doc = concat!("# use core::slice::", stringify!($name), ";")]
            #[doc = concat!("let iter: ", stringify!($name<'_, u8>), " = Default::default();")]
            /// assert_eq!(iter.len(), 0);
            /// ```
            fn default() -> Self {
                (& $( $mut_ )? []).into_iter()
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
