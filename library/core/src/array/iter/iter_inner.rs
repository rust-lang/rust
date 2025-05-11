//! Defines the `IntoIter` owned iterator for arrays.

use crate::mem::MaybeUninit;
use crate::num::NonZero;
use crate::ops::{IndexRange, NeverShortCircuit, Try};
use crate::{fmt, iter};

#[allow(private_bounds)]
trait PartialDrop {
    /// # Safety
    /// `self[alive]` are all initialized before the call,
    /// then are never used (without reinitializing them) after it.
    unsafe fn partial_drop(&mut self, alive: IndexRange);
}
impl<T> PartialDrop for [MaybeUninit<T>] {
    unsafe fn partial_drop(&mut self, alive: IndexRange) {
        // SAFETY: We know that all elements within `alive` are properly initialized.
        unsafe { self.get_unchecked_mut(alive).assume_init_drop() }
    }
}
impl<T, const N: usize> PartialDrop for [MaybeUninit<T>; N] {
    unsafe fn partial_drop(&mut self, alive: IndexRange) {
        let slice: &mut [MaybeUninit<T>] = self;
        // SAFETY: Initialized elements in the array are also initialized in the slice.
        unsafe { slice.partial_drop(alive) }
    }
}

/// The internals of a by-value array iterator.
///
/// The real `array::IntoIter<T, N>` stores a `PolymorphicIter<[MaybeUninit<T>, N]>`
/// which it unsizes to `PolymorphicIter<[MaybeUninit<T>]>` to iterate.
#[allow(private_bounds)]
pub(super) struct PolymorphicIter<DATA: ?Sized>
where
    DATA: PartialDrop,
{
    /// The elements in `data` that have not been yielded yet.
    ///
    /// Invariants:
    /// - `alive.end <= N`
    ///
    /// (And the `IndexRange` type requires `alive.start <= alive.end`.)
    alive: IndexRange,

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
    data: DATA,
}

#[allow(private_bounds)]
impl<DATA: ?Sized> PolymorphicIter<DATA>
where
    DATA: PartialDrop,
{
    #[inline]
    pub(super) const fn len(&self) -> usize {
        self.alive.len()
    }
}

#[allow(private_bounds)]
impl<DATA: ?Sized> Drop for PolymorphicIter<DATA>
where
    DATA: PartialDrop,
{
    #[inline]
    fn drop(&mut self) {
        // SAFETY: by our type invariant `self.alive` is exactly the initialized
        // items, and this is drop so nothing can use the items afterwards.
        unsafe { self.data.partial_drop(self.alive.clone()) }
    }
}

impl<T, const N: usize> PolymorphicIter<[MaybeUninit<T>; N]> {
    #[inline]
    pub(super) const fn empty() -> Self {
        Self { alive: IndexRange::zero_to(0), data: [const { MaybeUninit::uninit() }; N] }
    }

    /// # Safety
    /// `data[alive]` are all initialized.
    #[inline]
    pub(super) const unsafe fn new_unchecked(alive: IndexRange, data: [MaybeUninit<T>; N]) -> Self {
        Self { alive, data }
    }
}

impl<T: Clone, const N: usize> Clone for PolymorphicIter<[MaybeUninit<T>; N]> {
    #[inline]
    fn clone(&self) -> Self {
        // Note, we don't really need to match the exact same alive range, so
        // we can just clone into offset 0 regardless of where `self` is.
        let mut new = Self::empty();

        fn clone_into_new<U: Clone>(
            source: &PolymorphicIter<[MaybeUninit<U>]>,
            target: &mut PolymorphicIter<[MaybeUninit<U>]>,
        ) {
            // Clone all alive elements.
            for (src, dst) in iter::zip(source.as_slice(), &mut target.data) {
                // Write a clone into the new array, then update its alive range.
                // If cloning panics, we'll correctly drop the previous items.
                dst.write(src.clone());
                // This addition cannot overflow as we're iterating a slice,
                // the length of which always fits in usize.
                target.alive = IndexRange::zero_to(target.alive.end() + 1);
            }
        }

        clone_into_new(self, &mut new);
        new
    }
}

impl<T> PolymorphicIter<[MaybeUninit<T>]> {
    #[inline]
    pub(super) fn as_slice(&self) -> &[T] {
        // SAFETY: We know that all elements within `alive` are properly initialized.
        unsafe {
            let slice = self.data.get_unchecked(self.alive.clone());
            slice.assume_init_ref()
        }
    }

    #[inline]
    pub(super) fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: We know that all elements within `alive` are properly initialized.
        unsafe {
            let slice = self.data.get_unchecked_mut(self.alive.clone());
            slice.assume_init_mut()
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for PolymorphicIter<[MaybeUninit<T>]> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Only print the elements that were not yielded yet: we cannot
        // access the yielded elements anymore.
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}

/// Iterator-equivalent methods.
///
/// We don't implement the actual iterator traits because we want to implement
/// things like `try_fold` that require `Self: Sized` (which we're not).
impl<T> PolymorphicIter<[MaybeUninit<T>]> {
    #[inline]
    pub(super) fn next(&mut self) -> Option<T> {
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

    #[inline]
    pub(super) fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    pub(super) fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
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
    pub(super) fn fold<B>(&mut self, init: B, f: impl FnMut(B, T) -> B) -> B {
        self.try_fold(init, NeverShortCircuit::wrap_mut_2(f)).0
    }

    #[inline]
    pub(super) fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        F: FnMut(B, T) -> R,
        R: Try<Output = B>,
    {
        // `alive` is an `IndexRange`, not an arbitrary iterator, so we can
        // trust that its `try_fold` isn't going to do something weird like
        // call the fold-er multiple times for the same index.
        let data = &mut self.data;
        self.alive.try_fold(init, move |accum, idx| {
            // SAFETY: `idx` has been removed from the alive range, so we're not
            // going to drop it (even if `f` panics) and thus its ok to give
            // out ownership of that item to `f` to handle.
            let elem = unsafe { data.get_unchecked(idx).assume_init_read() };
            f(accum, elem)
        })
    }

    #[inline]
    pub(super) fn next_back(&mut self) -> Option<T> {
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
    pub(super) fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
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

    #[inline]
    pub(super) fn rfold<B>(&mut self, init: B, f: impl FnMut(B, T) -> B) -> B {
        self.try_rfold(init, NeverShortCircuit::wrap_mut_2(f)).0
    }

    #[inline]
    pub(super) fn try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        F: FnMut(B, T) -> R,
        R: Try<Output = B>,
    {
        // `alive` is an `IndexRange`, not an arbitrary iterator, so we can
        // trust that its `try_rfold` isn't going to do something weird like
        // call the fold-er multiple times for the same index.
        let data = &mut self.data;
        self.alive.try_rfold(init, move |accum, idx| {
            // SAFETY: `idx` has been removed from the alive range, so we're not
            // going to drop it (even if `f` panics) and thus its ok to give
            // out ownership of that item to `f` to handle.
            let elem = unsafe { data.get_unchecked(idx).assume_init_read() };
            f(accum, elem)
        })
    }
}
