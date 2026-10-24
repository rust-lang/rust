//! Macros used by iterators of slice.

// The shared definition of the `Iter` and `IterMut` iterators
macro_rules! iterator {
    (
        struct $name:ident -> $ptr:ty,
        $elem:ty,
        $raw_mut:tt,
        {$( $mut_:tt )?},
        $into_ref:ident,
        $array_ref:ident,
        {$($extra:tt)*}
    ) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<T> ExactSizeIterator for $name<'_, T> {
            #[inline(always)]
            fn len(&self) -> usize {
                self.inner.len()
            }

            #[inline(always)]
            fn is_empty(&self) -> bool {
                self.inner.is_empty()
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, T> Iterator for $name<'a, T> {
            type Item = $elem;

            #[inline]
            fn next(&mut self) -> Option<$elem> {
                self.inner.next().map(|ptr|unsafe { {ptr}.$into_ref() })
           }

            fn next_chunk<const N:usize>(&mut self) -> Result<[$elem; N], crate::array::IntoIter<$elem, N>> {
                self
                    .inner
                    .next_chunk::<N>()
                    .map(|chunk: [crate::ptr::NonNull<_>; N]| unsafe { crate::intrinsics::transmute_unchecked(chunk) })
                    .map_err(
                        |rest| {
                            let (data, alive) = rest.into_inner();
                            unsafe { crate::array::IntoIter::new_unchecked(crate::intrinsics::transmute_unchecked(data), alive) }
                        },
                    )
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.inner.size_hint()
            }

            #[inline]
            fn count(self) -> usize {
                self.inner.count()
            }

            #[inline]
            fn nth(&mut self, n: usize) -> Option<$elem> {
                self.inner.nth(n).map(|ptr| unsafe { {ptr}.$into_ref() })
            }

            #[inline]
            fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
                self.inner.advance_by(n)
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
self.inner.fold(init, |acc, x| f(acc, unsafe { {x}.$into_ref()}))
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
            fn position<P>(&mut self, mut predicate: P) -> Option<usize> where
                Self: Sized,
                P: FnMut(Self::Item) -> bool,
            {
                self.inner.position(|x| predicate(unsafe {{x}.$into_ref()}))
            }

            // We override the default implementation, which uses `try_fold`,
            // because this simple implementation generates less LLVM IR and is
            // faster to compile. Also, the `assume` avoids a bounds check.
            // FIXME: this crashes the compiler?...
            // #[inline]
            // fn rposition<P>(&mut self, mut predicate: P) -> Option<usize> where
            //     P: FnMut(Self::Item) -> bool,
            //     Self: Sized + ExactSizeIterator + DoubleEndedIterator
            // {
            //     self.inner.rposition(|x| predicate(x.$into_ref()))
            // }

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
                unsafe { self.inner.__iterator_get_unchecked(idx).$into_ref() }
            }

            $($extra)*
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, T> DoubleEndedIterator for $name<'a, T> {
            #[inline]
            fn next_back(&mut self) -> Option<$elem> {
                unsafe {
                    self.inner.next_back().map(|ptr| {ptr}.$into_ref())
                }
            }

            #[inline]
            fn nth_back(&mut self, n: usize) -> Option<$elem> {
                self.inner.nth_back(n).map(|ptr| unsafe{{ptr}.$into_ref()})
            }

            #[inline]
            fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
                self.inner.advance_back_by(n)
            }
        }

        #[stable(feature = "fused", since = "1.26.0")]
        impl<T> FusedIterator for $name<'_, T> {}

        #[unstable(feature = "trusted_len", issue = "37572")]
        unsafe impl<T> TrustedLen for $name<'_, T> {}

        #[stable(feature = "default_iters", since = "1.70.0")]
        impl<T> Default for $name<'_, T> {
            /// Creates an empty slice iterator.
            ///
            /// ```
            #[doc = concat!("# use core::slice::", stringify!($name), ";")]
            #[doc = concat!("let iter: ", stringify!($name<'_, u8>), " = Default::default();")]
            /// assert_eq!(iter.len(), 0);
            /// ```
            fn default() -> Self {
                Self { inner: <_>::default(), _marker: PhantomData }
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
