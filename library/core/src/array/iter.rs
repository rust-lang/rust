//! Defines the `IntoIter` owned iterator for arrays.

use crate::intrinsics::transmute_unchecked;
use crate::iter::{FusedIterator, TrustedLen, TrustedRandomAccessNoCoerce};
use crate::mem::MaybeUninit;
use crate::num::NonZero;
use crate::ops::{IndexRange, Range, Try};
use crate::{fmt, ptr};

mod iter_inner;

type InnerSized<T, const N: usize> = iter_inner::PolymorphicIter<[MaybeUninit<T>; N]>;
type InnerUnsized<T> = iter_inner::PolymorphicIter<[MaybeUninit<T>]>;

/// A by-value [array] iterator.
#[stable(feature = "array_value_iter", since = "1.51.0")]
#[rustc_insignificant_dtor]
#[rustc_diagnostic_item = "ArrayIntoIter"]
#[derive(Clone)]
pub struct IntoIter<T, const N: usize> {
    inner: InnerSized<T, N>,
}

impl<T, const N: usize> IntoIter<T, N> {
    #[inline]
    fn unsize(&self) -> &InnerUnsized<T> {
        &self.inner
    }
    #[inline]
    fn unsize_mut(&mut self) -> &mut InnerUnsized<T> {
        &mut self.inner
    }
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
    #[inline]
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
        // SAFETY: The original array was entirely initialized and the the alive
        // range we're passing here represents that fact.
        let inner = unsafe { InnerSized::new_unchecked(IndexRange::zero_to(N), data) };
        IntoIter { inner }
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
    #[inline]
    pub const unsafe fn new_unchecked(
        buffer: [MaybeUninit<T>; N],
        initialized: Range<usize>,
    ) -> Self {
        // SAFETY: one of our safety conditions is that the range is canonical.
        let alive = unsafe { IndexRange::new_unchecked(initialized.start, initialized.end) };
        // SAFETY: one of our safety condition is that these items are initialized.
        let inner = unsafe { InnerSized::new_unchecked(alive, buffer) };
        IntoIter { inner }
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
    #[inline]
    pub const fn empty() -> Self {
        let inner = InnerSized::empty();
        IntoIter { inner }
    }

    /// Returns an immutable slice of all elements that have not been yielded
    /// yet.
    #[stable(feature = "array_value_iter", since = "1.51.0")]
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.unsize().as_slice()
    }

    /// Returns a mutable slice of all elements that have not been yielded yet.
    #[stable(feature = "array_value_iter", since = "1.51.0")]
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.unsize_mut().as_mut_slice()
    }
}

#[stable(feature = "array_value_iter_default", since = "CURRENT_RUSTC_VERSION")]
impl<T, const N: usize> Default for IntoIter<T, N> {
    fn default() -> Self {
        IntoIter::empty()
    }
}

#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.unsize_mut().next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.unsize().size_hint()
    }

    #[inline]
    fn fold<Acc, Fold>(mut self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.unsize_mut().fold(init, fold)
    }

    #[inline]
    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        self.unsize_mut().try_fold(init, f)
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.unsize_mut().advance_by(n)
    }

    #[inline]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        // SAFETY: The caller must provide an idx that is in bound of the remainder.
        let elem_ref = unsafe { self.as_mut_slice().get_unchecked_mut(idx) };
        // SAFETY: We only implement `TrustedRandomAccessNoCoerce` for types
        // which are actually `Copy`, so cannot have multiple-drop issues.
        unsafe { ptr::read(elem_ref) }
    }
}

#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.unsize_mut().next_back()
    }

    #[inline]
    fn rfold<Acc, Fold>(mut self, init: Acc, rfold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.unsize_mut().rfold(init, rfold)
    }

    #[inline]
    fn try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        self.unsize_mut().try_rfold(init, f)
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.unsize_mut().advance_back_by(n)
    }
}

#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T, const N: usize> Drop for IntoIter<T, N> {
    #[inline]
    fn drop(&mut self) {
        // `inner` now handles this, but it'd technically be a breaking change
        // to remove this `impl`, even though it's useless.
    }
}

#[stable(feature = "array_value_iter_impls", since = "1.40.0")]
impl<T, const N: usize> ExactSizeIterator for IntoIter<T, N> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
    #[inline]
    fn is_empty(&self) -> bool {
        self.inner.len() == 0
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
impl<T: fmt::Debug, const N: usize> fmt::Debug for IntoIter<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.unsize().fmt(f)
    }
}
