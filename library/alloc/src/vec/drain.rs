use crate::alloc::{Allocator, Global};
use core::fmt;
use core::iter::{FusedIterator, TrustedLen};
use core::marker::PhantomData;
use core::mem::{ManuallyDrop, SizedTypeProperties};
use core::ptr::{self, NonNull};
use core::slice::{self};

use super::Vec;

/// A draining iterator for `Vec<T>`.
///
/// This `struct` is created by [`Vec::drain`].
/// See its documentation for more.
///
/// # Example
///
/// ```
/// let mut v = vec![0, 1, 2];
/// let iter: std::vec::Drain<'_, _> = v.drain(..);
/// ```
#[stable(feature = "drain", since = "1.6.0")]
pub struct Drain<
    'a,
    T: 'a,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator + 'a = Global,
> {
    /// Index of tail to preserve
    pub(super) tail_start: usize,
    /// Length of tail
    pub(super) tail_len: usize,
    /// Current remaining range to remove
    pub(super) iter: slice::DrainRaw<T>,
    pub(super) vec: NonNull<Vec<T, A>>,
    pub(super) phantom: PhantomData<&'a [T]>,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug, A: Allocator> fmt::Debug for Drain<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Drain").field(&self.as_slice()).finish()
    }
}

impl<'a, T, A: Allocator> Drain<'a, T, A> {
    /// Returns the remaining items of this iterator as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec!['a', 'b', 'c'];
    /// let mut drain = vec.drain(..);
    /// assert_eq!(drain.as_slice(), &['a', 'b', 'c']);
    /// let _ = drain.next().unwrap();
    /// assert_eq!(drain.as_slice(), &['b', 'c']);
    /// ```
    #[must_use]
    #[stable(feature = "vec_drain_as_slice", since = "1.46.0")]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: Restricting the lifetime of the returned slice to the self
        // borrow keeps anything else from dropping them.
        unsafe { self.iter.as_nonnull_slice().as_ref() }
    }

    /// Returns a reference to the underlying allocator.
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[must_use]
    #[inline]
    pub fn allocator(&self) -> &A {
        unsafe { self.vec.as_ref().allocator() }
    }

    /// Keep unyielded elements in the source `Vec`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(drain_keep_rest)]
    ///
    /// let mut vec = vec!['a', 'b', 'c'];
    /// let mut drain = vec.drain(..);
    ///
    /// assert_eq!(drain.next().unwrap(), 'a');
    ///
    /// // This call keeps 'b' and 'c' in the vec.
    /// drain.keep_rest();
    ///
    /// // If we wouldn't call `keep_rest()`,
    /// // `vec` would be empty.
    /// assert_eq!(vec, ['b', 'c']);
    /// ```
    #[unstable(feature = "drain_keep_rest", issue = "101122")]
    pub fn keep_rest(self) {
        // At this moment layout looks like this:
        //
        // [head] [yielded by next] [unyielded] [yielded by next_back] [tail]
        //        ^-- start         \_________/-- unyielded_len        \____/-- self.tail_len
        //                          ^-- unyielded_ptr                  ^-- tail
        //
        // Normally `Drop` impl would drop [unyielded] and then move [tail] to the `start`.
        // Here we want to
        // 1. Move [unyielded] to `start`
        // 2. Move [tail] to a new start at `start + len(unyielded)`
        // 3. Update length of the original vec to `len(head) + len(unyielded) + len(tail)`
        //    a. In case of ZST, this is the only thing we want to do
        // 4. Do *not* drop self, as everything is put in a consistent state already, there is nothing to do
        let mut this = ManuallyDrop::new(self);

        unsafe {
            let source_vec = this.vec.as_mut();

            let start = source_vec.len();
            let tail = this.tail_start;

            let keep_items = this.iter.forget_remaining();
            let unyielded_len = keep_items.len();
            let unyielded_ptr = keep_items.as_mut_ptr();

            // ZSTs have no identity, so we don't need to move them around.
            if !T::IS_ZST {
                let start_ptr = source_vec.as_mut_ptr().add(start);

                // memmove back unyielded elements
                if unyielded_ptr != start_ptr {
                    let src = unyielded_ptr;
                    let dst = start_ptr;

                    ptr::copy(src, dst, unyielded_len);
                }

                // memmove back untouched tail
                if tail != (start + unyielded_len) {
                    let src = source_vec.as_ptr().add(tail);
                    let dst = start_ptr.add(unyielded_len);
                    ptr::copy(src, dst, this.tail_len);
                }
            }

            source_vec.set_len(start + unyielded_len + this.tail_len);
        }
    }
}

#[stable(feature = "vec_drain_as_slice", since = "1.46.0")]
impl<'a, T, A: Allocator> AsRef<[T]> for Drain<'a, T, A> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

#[stable(feature = "drain", since = "1.6.0")]
unsafe impl<T: Sync, A: Sync + Allocator> Sync for Drain<'_, T, A> {}
#[stable(feature = "drain", since = "1.6.0")]
unsafe impl<T: Send, A: Send + Allocator> Send for Drain<'_, T, A> {}

#[stable(feature = "drain", since = "1.6.0")]
impl<T, A: Allocator> Iterator for Drain<'_, T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<T, A: Allocator> DoubleEndedIterator for Drain<'_, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back()
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<T, A: Allocator> Drop for Drain<'_, T, A> {
    fn drop(&mut self) {
        /// Moves back the un-`Drain`ed elements to restore the original `Vec`.
        struct DropGuard<'r, 'a, T, A: Allocator>(&'r mut Drain<'a, T, A>);

        impl<'r, 'a, T, A: Allocator> Drop for DropGuard<'r, 'a, T, A> {
            fn drop(&mut self) {
                if self.0.tail_len > 0 {
                    unsafe {
                        let source_vec = self.0.vec.as_mut();
                        // memmove back untouched tail, update to new length
                        let start = source_vec.len();
                        let tail = self.0.tail_start;
                        if tail != start {
                            let src = source_vec.as_ptr().add(tail);
                            let dst = source_vec.as_mut_ptr().add(start);
                            ptr::copy(src, dst, self.0.tail_len);
                        }
                        source_vec.set_len(start + self.0.tail_len);
                    }
                }
            }
        }

        if T::IS_ZST {
            let drop_len = self.iter.forget_remaining().len();

            // ZSTs have no identity, so we don't need to move them around, we only need to drop the correct amount.
            // this can be achieved by manipulating the Vec length instead of moving values out from `iter`.
            unsafe {
                let vec = self.vec.as_mut();
                let old_len = vec.len();
                vec.set_len(old_len + drop_len + self.tail_len);
                vec.truncate(old_len + self.tail_len);
            }

            return;
        }

        // ensure elements are moved back into their appropriate places, even when drop_in_place panics
        let guard = DropGuard(self);

        guard.0.iter.drop_remaining();
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<T, A: Allocator> ExactSizeIterator for Drain<'_, T, A> {
    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T, A: Allocator> TrustedLen for Drain<'_, T, A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, A: Allocator> FusedIterator for Drain<'_, T, A> {}
