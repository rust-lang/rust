//! Defines the `IntoIter` owned iterator for arrays.

use crate::intrinsics::transmute_unchecked;
use crate::iter::{self, FusedIterator, TrustedLen, TrustedRandomAccessNoCoerce};
use crate::mem::MaybeUninit;
use crate::num::NonZero;
use crate::ops::{IndexRange, Range};
use crate::{fmt, ptr};

/// A by-value [array] iterator.
#[stable(feature = "array_value_iter", since = "1.51.0")]
#[rustc_insignificant_dtor]
#[rustc_diagnostic_item = "ArrayIntoIter"]
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
    /// - `alive.end <= N`
    ///
    /// (And the `IndexRange` type requires `alive.start <= alive.end`.)
    alive: IndexRange,
}

// Note: the `#[rustc_skip_during_method_dispatch(array)]` on `trait IntoIterator`
// hides this implementation from explicit `.into_iter()` calls on editions < 2021,
// so those calls will still resolve to the slice implementation, by reference.
#[stable(feature = "array_into_iter_impl", since = "1.53.0")]
impl<T, const N: usize> IntoIterator for [T; N] {
    type Item = T;
    type IntoIter = IntoIter<T, N>;

    /// Creates a consuming iterator, that is, one that moves each value out of
    /// the array (from start to end).
    ///
    /// The array cannot be used after calling this unless `T` implements
    /// `Copy`, so the whole array is copied.
    ///
    /// Arrays have special behavior when calling `.into_iter()` prior to the
    /// 2021 edition -- see the [array] Editions section for more information.
    ///
    /// [array]: prim@array
    fn into_iter(self) -> Self::IntoIter {
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
        //
        // FIXME: If normal `transmute` ever gets smart enough to allow this
        // directly, use it instead of `transmute_unchecked`.
        let data: [MaybeUninit<T>; N] = unsafe { transmute_unchecked(self) };
        IntoIter { data, alive: IndexRange::zero_to(N) }
    }
}

impl<T, const N: usize> IntoIter<T, N> {
    /// Creates a new iterator over the given `array`.
    #[stable(feature = "array_value_iter", since = "1.51.0")]
    #[deprecated(since = "1.59.0", note = "use `IntoIterator::into_iter` instead")]
    pub fn new(array: [T; N]) -> Self {
        IntoIterator::into_iter(array)
    }

    /// Creates an iterator over the elements in a partially-initialized buffer.
    ///
    /// If you have a fully-initialized array, then use [`IntoIterator`].
    /// But this is useful for returning partial results from unsafe code.
    ///
    /// # Safety
    ///
    /// - The `buffer[initialized]` elements must all be initialized.
    /// - The range must be canonical, with `initialized.start <= initialized.end`.
    /// - The range must be in-bounds for the buffer, with `initialized.end <= N`.
    ///   (Like how indexing `[0][100..100]` fails despite the range being empty.)
    ///
    /// It's sound to have more elements initialized than mentioned, though that
    /// will most likely result in them being leaked.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(array_into_iter_constructors)]
    /// #![feature(maybe_uninit_uninit_array_transpose)]
    /// use std::array::IntoIter;
    /// use std::mem::MaybeUninit;
    ///
    /// # // Hi!  Thanks for reading the code. This is restricted to `Copy` because
    /// # // otherwise it could leak. A fully-general version this would need a drop
    /// # // guard to handle panics from the iterator, but this works for an example.
    /// fn next_chunk<T: Copy, const N: usize>(
    ///     it: &mut impl Iterator<Item = T>,
    /// ) -> Result<[T; N], IntoIter<T, N>> {
    ///     let mut buffer = [const { MaybeUninit::uninit() }; N];
    ///     let mut i = 0;
    ///     while i < N {
    ///         match it.next() {
    ///             Some(x) => {
    ///                 buffer[i].write(x);
    ///                 i += 1;
    ///             }
    ///             None => {
    ///                 // SAFETY: We've initialized the first `i` items
    ///                 unsafe {
    ///                     return Err(IntoIter::new_unchecked(buffer, 0..i));
    ///                 }
    ///             }
    ///         }
    ///     }
    ///
    ///     // SAFETY: We've initialized all N items
    ///     unsafe { Ok(buffer.transpose().assume_init()) }
    /// }
    ///
    /// let r: [_; 4] = next_chunk(&mut (10..16)).unwrap();
    /// assert_eq!(r, [10, 11, 12, 13]);
    /// let r: IntoIter<_, 40> = next_chunk(&mut (10..16)).unwrap_err();
    /// assert_eq!(r.collect::<Vec<_>>(), vec![10, 11, 12, 13, 14, 15]);
    /// ```
    #[unstable(feature = "array_into_iter_constructors", issue = "91583")]
    pub const unsafe fn new_unchecked(
        buffer: [MaybeUninit<T>; N],
        initialized: Range<usize>,
    ) -> Self {
        // SAFETY: one of our safety conditions is that the range is canonical.
        let alive = unsafe { IndexRange::new_unchecked(initialized.start, initialized.end) };
        Self { data: buffer, alive }
    }

    /// Creates an iterator over `T` which returns no elements.
    ///
    /// If you just need an empty iterator, then use
    /// [`iter::empty()`](crate::iter::empty) instead.
    /// And if you need an empty array, use `[]`.
    ///
    /// But this is useful when you need an `array::IntoIter<T, N>` *specifically*.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(array_into_iter_constructors)]
    /// use std::array::IntoIter;
    ///
    /// let empty = IntoIter::<i32, 3>::empty();
    /// assert_eq!(empty.len(), 0);
    /// assert_eq!(empty.as_slice(), &[]);
    ///
    /// let empty = IntoIter::<std::convert::Infallible, 200>::empty();
    /// assert_eq!(empty.len(), 0);
    /// ```
    ///
    /// `[1, 2].into_iter()` and `[].into_iter()` have different types
    /// ```should_fail,edition2021
    /// #![feature(array_into_iter_constructors)]
    /// use std::array::IntoIter;
    ///
    /// pub fn get_bytes(b: bool) -> IntoIter<i8, 4> {
    ///     if b {
    ///         [1, 2, 3, 4].into_iter()
    ///     } else {
    ///         [].into_iter() // error[E0308]: mismatched types
    ///     }
    /// }
    /// ```
    ///
    /// But using this method you can get an empty iterator of appropriate size:
    /// ```edition2021
    /// #![feature(array_into_iter_constructors)]
    /// use std::array::IntoIter;
    ///
    /// pub fn get_bytes(b: bool) -> IntoIter<i8, 4> {
    ///     if b {
    ///         [1, 2, 3, 4].into_iter()
    ///     } else {
    ///         IntoIter::empty()
    ///     }
    /// }
    ///
    /// assert_eq!(get_bytes(true).collect::<Vec<_>>(), vec![1, 2, 3, 4]);
    /// assert_eq!(get_bytes(false).collect::<Vec<_>>(), vec![]);
    /// ```
    #[unstable(feature = "array_into_iter_constructors", issue = "91583")]
    pub const fn empty() -> Self {
        let buffer = [const { MaybeUninit::uninit() }; N];
        let initialized = 0..0;

        // SAFETY: We're telling it that none of the elements are initialized,
        // which is trivially true. And ∀N: usize, 0 <= N.
        unsafe { Self::new_unchecked(buffer, initialized) }
    }

    /// Returns an immutable slice of all elements that have not been yielded
    /// yet.
    #[stable(feature = "array_value_iter", since = "1.51.0")]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: We know that all elements within `alive` are properly initialized.
        unsafe {
            let slice = self.data.get_unchecked(self.alive.clone());
            slice.assume_init_ref()
        }
    }

    /// Returns a mutable slice of all elements that have not been yielded yet.
    #[stable(feature = "array_value_iter", since = "1.51.0")]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: We know that all elements within `alive` are properly initialized.
        unsafe {
            let slice = self.data.get_unchecked_mut(self.alive.clone());
            slice.assume_init_mut()
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

    #[inline]
    fn fold<Acc, Fold>(mut self, init: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        let data = &mut self.data;
        iter::ByRefSized(&mut self.alive).fold(init, |acc, idx| {
            // SAFETY: idx is obtained by folding over the `alive` range, which implies the
            // value is currently considered alive but as the range is being consumed each value
            // we read here will only be read once and then considered dead.
            fold(acc, unsafe { data.get_unchecked(idx).assume_init_read() })
        })
    }

    fn count(self) -> usize {
        self.len()
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        // This also moves the start, which marks them as conceptually "dropped",
        // so if anything goes bad then our drop impl won't double-free them.
        let range_to_drop = self.alive.take_prefix(n);
        let remaining = n - range_to_drop.len();

        // SAFETY: These elements are currently initialized, so it's fine to drop them.
        unsafe {
            let slice = self.data.get_unchecked_mut(range_to_drop);
            slice.assume_init_drop();
        }

        NonZero::new(remaining).map_or(Ok(()), Err)
    }

    #[inline]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        // SAFETY: The caller must provide an idx that is in bound of the remainder.
        unsafe { self.data.as_ptr().add(self.alive.start()).add(idx).cast::<T>().read() }
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

    #[inline]
    fn rfold<Acc, Fold>(mut self, init: Acc, mut rfold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        let data = &mut self.data;
        iter::ByRefSized(&mut self.alive).rfold(init, |acc, idx| {
            // SAFETY: idx is obtained by folding over the `alive` range, which implies the
            // value is currently considered alive but as the range is being consumed each value
            // we read here will only be read once and then considered dead.
            rfold(acc, unsafe { data.get_unchecked(idx).assume_init_read() })
        })
    }

    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        // This also moves the end, which marks them as conceptually "dropped",
        // so if anything goes bad then our drop impl won't double-free them.
        let range_to_drop = self.alive.take_suffix(n);
        let remaining = n - range_to_drop.len();

        // SAFETY: These elements are currently initialized, so it's fine to drop them.
        unsafe {
            let slice = self.data.get_unchecked_mut(range_to_drop);
            slice.assume_init_drop();
        }

        NonZero::new(remaining).map_or(Ok(()), Err)
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
        self.alive.len()
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

#[doc(hidden)]
#[unstable(issue = "none", feature = "std_internals")]
#[rustc_unsafe_specialization_marker]
pub trait NonDrop {}

// T: Copy as approximation for !Drop since get_unchecked does not advance self.alive
// and thus we can't implement drop-handling
#[unstable(issue = "none", feature = "std_internals")]
impl<T: Copy> NonDrop for T {}

#[doc(hidden)]
#[unstable(issue = "none", feature = "std_internals")]
unsafe impl<T, const N: usize> TrustedRandomAccessNoCoerce for IntoIter<T, N>
where
    T: NonDrop,
{
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T: Clone, const N: usize> Clone for IntoIter<T, N> {
    fn clone(&self) -> Self {
        // Note, we don't really need to match the exact same alive range, so
        // we can just clone into offset 0 regardless of where `self` is.
        let mut new =
            Self { data: [const { MaybeUninit::uninit() }; N], alive: IndexRange::zero_to(0) };

        // Clone all alive elements.
        for (src, dst) in iter::zip(self.as_slice(), &mut new.data) {
            // Write a clone into the new array, then update its alive range.
            // If cloning panics, we'll correctly drop the previous items.
            dst.write(src.clone());
            // This addition cannot overflow as we're iterating a slice
            new.alive = IndexRange::zero_to(new.alive.end() + 1);
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
