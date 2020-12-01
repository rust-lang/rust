//! Defines the `IntoIter` owned iterator for arrays.

use crate::{
    fmt,
    iter::{ExactSizeIterator, FusedIterator, TrustedLen},
    mem::{self, MaybeUninit},
    ops::Range,
    ptr,
};

/// A by-value [array] iterator.
///
/// [array]: ../../std/primitive.array.html
#[unstable(feature = "array_value_iter", issue = "65798")]
pub struct IntoIter<T, const N: usize> {
    /// This is the array we are iterating over.
    ///
    /// Elements with index `i` where `alive.start <= i < alive.end` have not
    /// been yielded yet and are valid array entries. Elements with indices `i
    /// < alive.start` or `i >= alive.end` have been yielded already and must
    /// not be accessed anymore! Those dead elements might even be in a
    /// completely uninitialized state!
    ///
    /// So the invariants are:
    /// - `data[alive]` is alive (i.e. contains valid elements)
    /// - `data[..alive.start]` and `data[alive.end..]` are dead (i.e. the
    ///   elements were already read and must not be touched anymore!)
    data: [MaybeUninit<T>; N],

    /// The elements in `data` that have not been yielded yet.
    ///
    /// Invariants:
    /// - `alive.start <= alive.end`
    /// - `alive.end <= N`
    alive: Range<usize>,
}

impl<T, const N: usize> IntoIter<T, N> {
    /// Creates a new iterator over the given `array`.
    ///
    /// *Note*: this method might never get stabilized and/or removed in the
    /// future as there will likely be another, preferred way of obtaining this
    /// iterator (either via `IntoIterator` for arrays or via another way).
    #[unstable(feature = "array_value_iter", issue = "65798")]
    pub fn new(array: [T; N]) -> Self {
        // SAFETY: The transmute here is actually safe. The docs of `MaybeUninit`
        // promise:
        //
        // > `MaybeUninit<T>` is guaranteed to have the same size and alignment
        // > as `T`.
        //
        // The docs even show a transmute from an array of `MaybeUninit<T>` to
        // an array of `T`.
        //
        // With that, this initialization satisfies the invariants.

        // FIXME(LukasKalbertodt): actually use `mem::transmute` here, once it
        // works with const generics:
        //     `mem::transmute::<[T; N], [MaybeUninit<T>; N]>(array)`
        //
        // Until then, we can use `mem::transmute_copy` to create a bitwise copy
        // as a different type, then forget `array` so that it is not dropped.
        unsafe {
            let iter = Self { data: mem::transmute_copy(&array), alive: 0..N };
            mem::forget(array);
            iter
        }
    }

    /// Returns an immutable slice of all elements that have not been yielded
    /// yet.
    #[unstable(feature = "array_value_iter_slice", issue = "65798")]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: We know that all elements within `alive` are properly initialized.
        unsafe {
            let slice = self.data.get_unchecked(self.alive.clone());
            MaybeUninit::slice_assume_init_ref(slice)
        }
    }

    /// Returns a mutable slice of all elements that have not been yielded yet.
    #[unstable(feature = "array_value_iter_slice", issue = "65798")]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: We know that all elements within `alive` are properly initialized.
        unsafe {
            let slice = self.data.get_unchecked_mut(self.alive.clone());
            MaybeUninit::slice_assume_init_mut(slice)
        }
    }
}

#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        // Get the next index from the front.
        //
        // Increasing `alive.start` by 1 maintains the invariant regarding
        // `alive`. However, due to this change, for a short time, the alive
        // zone is not `data[alive]` anymore, but `data[idx..alive.end]`.
        self.alive.next().map(|idx| {
            // Read the element from the array.
            // SAFETY: `idx` is an index into the former "alive" region of the
            // array. Reading this element means that `data[idx]` is regarded as
            // dead now (i.e. do not touch). As `idx` was the start of the
            // alive-zone, the alive zone is now `data[alive]` again, restoring
            // all invariants.
            unsafe { self.data.get_unchecked(idx).assume_init_read() }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        // Get the next index from the back.
        //
        // Decreasing `alive.end` by 1 maintains the invariant regarding
        // `alive`. However, due to this change, for a short time, the alive
        // zone is not `data[alive]` anymore, but `data[alive.start..=idx]`.
        self.alive.next_back().map(|idx| {
            // Read the element from the array.
            // SAFETY: `idx` is an index into the former "alive" region of the
            // array. Reading this element means that `data[idx]` is regarded as
            // dead now (i.e. do not touch). As `idx` was the end of the
            // alive-zone, the alive zone is now `data[alive]` again, restoring
            // all invariants.
            unsafe { self.data.get_unchecked(idx).assume_init_read() }
        })
    }
}

#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T, const N: usize> Drop for IntoIter<T, N> {
    fn drop(&mut self) {
        // SAFETY: This is safe: `as_mut_slice` returns exactly the sub-slice
        // of elements that have not been moved out yet and that remain
        // to be dropped.
        unsafe { ptr::drop_in_place(self.as_mut_slice()) }
    }
}

#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T, const N: usize> ExactSizeIterator for IntoIter<T, N> {
    fn len(&self) -> usize {
        // Will never underflow due to the invariant `alive.start <=
        // alive.end`.
        self.alive.end - self.alive.start
    }
    fn is_empty(&self) -> bool {
        self.alive.is_empty()
    }
}

#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T, const N: usize> FusedIterator for IntoIter<T, N> {}

// The iterator indeed reports the correct length. The number of "alive"
// elements (that will still be yielded) is the length of the range `alive`.
// This range is decremented in length in either `next` or `next_back`. It is
// always decremented by 1 in those methods, but only if `Some(_)` is returned.
#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
unsafe impl<T, const N: usize> TrustedLen for IntoIter<T, N> {}

#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T: Clone, const N: usize> Clone for IntoIter<T, N> {
    fn clone(&self) -> Self {
        // Note, we don't really need to match the exact same alive range, so
        // we can just clone into offset 0 regardless of where `self` is.
        let mut new = Self { data: MaybeUninit::uninit_array(), alive: 0..0 };

        // Clone all alive elements.
        for (src, dst) in self.as_slice().iter().zip(&mut new.data) {
            // Write a clone into the new array, then update its alive range.
            // If cloning panics, we'll correctly drop the previous items.
            dst.write(src.clone());
            new.alive.end += 1;
        }

        new
    }
}

#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T: fmt::Debug, const N: usize> fmt::Debug for IntoIter<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Only print the elements that were not yielded yet: we cannot
        // access the yielded elements anymore.
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}
