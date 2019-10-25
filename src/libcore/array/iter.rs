//! Defines the `IntoIter` owned iterator for arrays.

use crate::{
    fmt,
    iter::{ExactSizeIterator, FusedIterator, TrustedLen},
    mem::{self, MaybeUninit},
    ops::Range,
    ptr,
};
use super::LengthAtMost32;


/// A by-value [array] iterator.
///
/// [array]: ../../std/primitive.array.html
#[unstable(feature = "array_value_iter", issue = "65798")]
pub struct IntoIter<T, const N: usize>
where
    [T; N]: LengthAtMost32,
{
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

impl<T, const N: usize> IntoIter<T, {N}>
where
    [T; N]: LengthAtMost32,
{
    /// Creates a new iterator over the given `array`.
    ///
    /// *Note*: this method might never get stabilized and/or removed in the
    /// future as there will likely be another, preferred way of obtaining this
    /// iterator (either via `IntoIterator` for arrays or via another way).
    #[unstable(feature = "array_value_iter", issue = "65798")]
    pub fn new(array: [T; N]) -> Self {
        // The transmute here is actually safe. The docs of `MaybeUninit`
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
        //     `mem::transmute::<[T; {N}], [MaybeUninit<T>; {N}]>(array)`
        //
        // Until then, we do it manually here. We first create a bitwise copy
        // but cast the pointer so that it is treated as a different type. Then
        // we forget `array` so that it is not dropped.
        let data = unsafe {
            let data = ptr::read(&array as *const [T; N] as *const [MaybeUninit<T>; N]);
            mem::forget(array);
            data
        };

        Self {
            data,
            alive: 0..N,
        }
    }

    /// Returns an immutable slice of all elements that have not been yielded
    /// yet.
    fn as_slice(&self) -> &[T] {
        // This transmute is safe. As mentioned in `new`, `MaybeUninit` retains
        // the size and alignment of `T`. Furthermore, we know that all
        // elements within `alive` are properly initialized.
        let slice = &self.data[self.alive.clone()];
        unsafe {
            mem::transmute::<&[MaybeUninit<T>], &[T]>(slice)
        }
    }
}


#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T, const N: usize> Iterator for IntoIter<T, {N}>
where
    [T; N]: LengthAtMost32,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.alive.start == self.alive.end {
            return None;
        }

        // Bump start index.
        //
        // From the check above we know that `alive.start != alive.end`.
        // Combine this with the invariant `alive.start <= alive.end`, we know
        // that `alive.start < alive.end`. Increasing `alive.start` by 1
        // maintains the invariant regarding `alive`. However, due to this
        // change, for a short time, the alive zone is not `data[alive]`
        // anymore, but `data[idx..alive.end]`.
        let idx = self.alive.start;
        self.alive.start += 1;

        // Read the element from the array. This is safe: `idx` is an index
        // into the "alive" region of the array. Reading this element means
        // that `data[idx]` is regarded as dead now (i.e. do not touch). As
        // `idx` was the start of the alive-zone, the alive zone is now
        // `data[alive]` again, restoring all invariants.
        let out = unsafe { self.data.get_unchecked(idx).read() };

        Some(out)
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
impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, {N}>
where
    [T; N]: LengthAtMost32,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.alive.start == self.alive.end {
            return None;
        }

        // Decrease end index.
        //
        // From the check above we know that `alive.start != alive.end`.
        // Combine this with the invariant `alive.start <= alive.end`, we know
        // that `alive.start < alive.end`. As `alive.start` cannot be negative,
        // `alive.end` is at least 1, meaning that we can safely decrement it
        // by one. This also maintains the invariant `alive.start <=
        // alive.end`. However, due to this change, for a short time, the alive
        // zone is not `data[alive]` anymore, but `data[alive.start..alive.end
        // + 1]`.
        self.alive.end -= 1;

        // Read the element from the array. This is safe: `alive.end` is an
        // index into the "alive" region of the array. Compare the previous
        // comment that states that the alive region is
        // `data[alive.start..alive.end + 1]`. Reading this element means that
        // `data[alive.end]` is regarded as dead now (i.e. do not touch). As
        // `alive.end` was the end of the alive-zone, the alive zone is now
        // `data[alive]` again, restoring all invariants.
        let out = unsafe { self.data.get_unchecked(self.alive.end).read() };

        Some(out)
    }
}

#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T, const N: usize> Drop for IntoIter<T, {N}>
where
    [T; N]: LengthAtMost32,
{
    fn drop(&mut self) {
        // We simply drop each element via `for_each`. This should not incur
        // any significant runtime overhead and avoids adding another `unsafe`
        // block.
        self.by_ref().for_each(drop);
    }
}

#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T, const N: usize> ExactSizeIterator for IntoIter<T, {N}>
where
    [T; N]: LengthAtMost32,
{
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
impl<T, const N: usize> FusedIterator for IntoIter<T, {N}>
where
    [T; N]: LengthAtMost32,
{}

// The iterator indeed reports the correct length. The number of "alive"
// elements (that will still be yielded) is the length of the range `alive`.
// This range is decremented in length in either `next` or `next_back`. It is
// always decremented by 1 in those methods, but only if `Some(_)` is returned.
#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
unsafe impl<T, const N: usize> TrustedLen for IntoIter<T, {N}>
where
    [T; N]: LengthAtMost32,
{}

#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T: Clone, const N: usize> Clone for IntoIter<T, {N}>
where
    [T; N]: LengthAtMost32,
{
    fn clone(&self) -> Self {
        unsafe {
            // This creates a new uninitialized array. Note that the `assume_init`
            // refers to the array, not the individual elements. And it is Ok if
            // the array is in an uninitialized state as all elements may be
            // uninitialized (all bit patterns are valid). Compare the
            // `MaybeUninit` docs for more information.
            let mut new_data: [MaybeUninit<T>; N] = MaybeUninit::uninit().assume_init();

            // Clone all alive elements.
            for idx in self.alive.clone() {
                // The element at `idx` in the old array is alive, so we can
                // safely call `get_ref()`. We then clone it, and write the
                // clone into the new array.
                let clone = self.data.get_unchecked(idx).get_ref().clone();
                new_data.get_unchecked_mut(idx).write(clone);
            }

            Self {
                data: new_data,
                alive: self.alive.clone(),
            }
        }
    }
}

#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T: fmt::Debug, const N: usize> fmt::Debug for IntoIter<T, {N}>
where
    [T; N]: LengthAtMost32,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Only print the elements that were not yielded yet: we cannot
        // access the yielded elements anymore.
        f.debug_tuple("IntoIter")
            .field(&self.as_slice())
            .finish()
    }
}
