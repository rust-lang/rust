use core::async_iter::AsyncIterator;
use core::iter::{self, FusedIterator, TrustedLen, TrustedRandomAccessNoCoerce};
use core::mem::MaybeUninit;
use core::num::NonZero;
use core::ops::IndexRange;
use core::pin::Pin;
use core::task::{Context, Poll};
use core::{ptr, slice};

use crate::alloc::{Allocator, Global};
#[cfg(not(no_global_oom_handling))]
use crate::borrow::Cow;
use crate::boxed::Box;
#[cfg(not(no_global_oom_handling))]
use crate::string::String;
#[cfg(not(no_global_oom_handling))]
use crate::vec::Vec;
use crate::{fmt, vec};

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator + ?Sized, A: Allocator> Iterator for Box<I, A> {
    type Item = I::Item;
    fn next(&mut self) -> Option<I::Item> {
        (**self).next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        (**self).nth(n)
    }
    fn last(self) -> Option<I::Item> {
        BoxIter::last(self)
    }
}

trait BoxIter {
    type Item;
    fn last(self) -> Option<Self::Item>;
}

impl<I: Iterator + ?Sized, A: Allocator> BoxIter for Box<I, A> {
    type Item = I::Item;
    default fn last(self) -> Option<I::Item> {
        #[inline]
        fn some<T>(_: Option<T>, x: T) -> Option<T> {
            Some(x)
        }

        self.fold(None, some)
    }
}

/// Specialization for sized `I`s that uses `I`s implementation of `last()`
/// instead of the default.
#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator, A: Allocator> BoxIter for Box<I, A> {
    fn last(self) -> Option<I::Item> {
        (*self).last()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: DoubleEndedIterator + ?Sized, A: Allocator> DoubleEndedIterator for Box<I, A> {
    fn next_back(&mut self) -> Option<I::Item> {
        (**self).next_back()
    }
    fn nth_back(&mut self, n: usize) -> Option<I::Item> {
        (**self).nth_back(n)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<I: ExactSizeIterator + ?Sized, A: Allocator> ExactSizeIterator for Box<I, A> {
    fn len(&self) -> usize {
        (**self).len()
    }
    fn is_empty(&self) -> bool {
        (**self).is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I: FusedIterator + ?Sized, A: Allocator> FusedIterator for Box<I, A> {}

#[unstable(feature = "async_iterator", issue = "79024")]
impl<S: ?Sized + AsyncIterator + Unpin> AsyncIterator for Box<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut **self).poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }
}

/// This implementation is required to make sure that the `Box<[I]>: IntoIterator`
/// implementation doesn't overlap with `IntoIterator for T where T: Iterator` blanket.
#[stable(feature = "boxed_slice_into_iter", since = "1.80.0")]
impl<I, A: Allocator> !Iterator for Box<[I], A> {}

/// This implementation is required to make sure that the `&Box<[I]>: IntoIterator`
/// implementation doesn't overlap with `IntoIterator for T where T: Iterator` blanket.
#[stable(feature = "boxed_slice_into_iter", since = "1.80.0")]
impl<'a, I, A: Allocator> !Iterator for &'a Box<[I], A> {}

/// This implementation is required to make sure that the `&mut Box<[I]>: IntoIterator`
/// implementation doesn't overlap with `IntoIterator for T where T: Iterator` blanket.
#[stable(feature = "boxed_slice_into_iter", since = "1.80.0")]
impl<'a, I, A: Allocator> !Iterator for &'a mut Box<[I], A> {}

// Note: the `#[rustc_skip_during_method_dispatch(boxed_slice)]` on `trait IntoIterator`
// hides this implementation from explicit `.into_iter()` calls on editions < 2024,
// so those calls will still resolve to the slice implementation, by reference.
#[stable(feature = "boxed_slice_into_iter", since = "1.80.0")]
impl<I, A: Allocator> IntoIterator for Box<[I], A> {
    type IntoIter = vec::IntoIter<I, A>;
    type Item = I;
    fn into_iter(self) -> vec::IntoIter<I, A> {
        self.into_vec().into_iter()
    }
}

#[stable(feature = "boxed_slice_into_iter", since = "1.80.0")]
impl<'a, I, A: Allocator> IntoIterator for &'a Box<[I], A> {
    type IntoIter = slice::Iter<'a, I>;
    type Item = &'a I;
    fn into_iter(self) -> slice::Iter<'a, I> {
        self.iter()
    }
}

#[stable(feature = "boxed_slice_into_iter", since = "1.80.0")]
impl<'a, I, A: Allocator> IntoIterator for &'a mut Box<[I], A> {
    type IntoIter = slice::IterMut<'a, I>;
    type Item = &'a mut I;
    fn into_iter(self) -> slice::IterMut<'a, I> {
        self.iter_mut()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "boxed_slice_from_iter", since = "1.32.0")]
impl<I> FromIterator<I> for Box<[I]> {
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        iter.into_iter().collect::<Vec<_>>().into_boxed_slice()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "boxed_str_from_iter", since = "1.80.0")]
impl FromIterator<char> for Box<str> {
    fn from_iter<T: IntoIterator<Item = char>>(iter: T) -> Self {
        String::from_iter(iter).into_boxed_str()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "boxed_str_from_iter", since = "1.80.0")]
impl<'a> FromIterator<&'a char> for Box<str> {
    fn from_iter<T: IntoIterator<Item = &'a char>>(iter: T) -> Self {
        String::from_iter(iter).into_boxed_str()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "boxed_str_from_iter", since = "1.80.0")]
impl<'a> FromIterator<&'a str> for Box<str> {
    fn from_iter<T: IntoIterator<Item = &'a str>>(iter: T) -> Self {
        String::from_iter(iter).into_boxed_str()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "boxed_str_from_iter", since = "1.80.0")]
impl FromIterator<String> for Box<str> {
    fn from_iter<T: IntoIterator<Item = String>>(iter: T) -> Self {
        String::from_iter(iter).into_boxed_str()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "boxed_str_from_iter", since = "1.80.0")]
impl<A: Allocator> FromIterator<Box<str, A>> for Box<str> {
    fn from_iter<T: IntoIterator<Item = Box<str, A>>>(iter: T) -> Self {
        String::from_iter(iter).into_boxed_str()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "boxed_str_from_iter", since = "1.80.0")]
impl<'a> FromIterator<Cow<'a, str>> for Box<str> {
    fn from_iter<T: IntoIterator<Item = Cow<'a, str>>>(iter: T) -> Self {
        String::from_iter(iter).into_boxed_str()
    }
}

/// This implementation is required to make sure that the `Box<[I; N]>: IntoIterator`
/// implementation doesn't overlap with `IntoIterator for T where T: Iterator` blanket.
#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<I, const N: usize, A: Allocator> !Iterator for Box<[I; N], A> {}

/// This implementation is required to make sure that the `&Box<[I; N]>: IntoIterator`
/// implementation doesn't overlap with `IntoIterator for T where T: Iterator` blanket.
#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<'a, const N: usize, I, A: Allocator> !Iterator for &'a Box<[I; N], A> {}

/// This implementation is required to make sure that the `&mut Box<[I; N]>: IntoIterator`
/// implementation doesn't overlap with `IntoIterator for T where T: Iterator` blanket.
#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<'a, const N: usize, I, A: Allocator> !Iterator for &'a mut Box<[I; N], A> {}

/// A by-value `Box<[T; N]>` iterator.
#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
#[rustc_insignificant_dtor]
pub struct BoxedArrayIntoIter<T, const N: usize, A: Allocator = Global> {
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
    data: Box<[MaybeUninit<T>; N], A>,

    /// The elements in `data` that have not been yielded yet.
    ///
    /// Invariants:
    /// - `alive.end <= N`
    ///
    /// (And the `IndexRange` type requires `alive.start <= alive.end`.)
    alive: IndexRange,
}

impl<T, const N: usize, A: Allocator> BoxedArrayIntoIter<T, N, A> {
    /// Returns an immutable slice of all elements that have not been yielded
    /// yet.
    #[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: We know that all elements within `alive` are properly initialized.
        unsafe {
            let slice = self.data.get_unchecked(self.alive.clone());
            MaybeUninit::slice_assume_init_ref(slice)
        }
    }

    /// Returns a mutable slice of all elements that have not been yielded yet.
    #[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: We know that all elements within `alive` are properly initialized.
        unsafe {
            let slice = self.data.get_unchecked_mut(self.alive.clone());
            MaybeUninit::slice_assume_init_mut(slice)
        }
    }
}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<T, const N: usize, A: Allocator> Iterator for BoxedArrayIntoIter<T, N, A> {
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
            ptr::drop_in_place(MaybeUninit::slice_assume_init_mut(slice));
        }

        NonZero::new(remaining).map_or(Ok(()), Err)
    }

    #[inline]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        // SAFETY: The caller must provide an idx that is in bound of the remainder.
        unsafe { self.data.as_ptr().add(self.alive.start()).add(idx).cast::<T>().read() }
    }
}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<T, const N: usize, A: Allocator> DoubleEndedIterator for BoxedArrayIntoIter<T, N, A> {
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
            ptr::drop_in_place(MaybeUninit::slice_assume_init_mut(slice));
        }

        NonZero::new(remaining).map_or(Ok(()), Err)
    }
}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<T, const N: usize, A: Allocator> Drop for BoxedArrayIntoIter<T, N, A> {
    fn drop(&mut self) {
        // SAFETY: This is safe: `as_mut_slice` returns exactly the sub-slice
        // of elements that have not been moved out yet and that remain
        // to be dropped.
        unsafe { ptr::drop_in_place(self.as_mut_slice()) }
    }
}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<T, const N: usize, A: Allocator> ExactSizeIterator for BoxedArrayIntoIter<T, N, A> {
    fn len(&self) -> usize {
        self.alive.len()
    }
    fn is_empty(&self) -> bool {
        self.alive.is_empty()
    }
}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<T, const N: usize, A: Allocator> FusedIterator for BoxedArrayIntoIter<T, N, A> {}

// The iterator indeed reports the correct length. The number of "alive"
// elements (that will still be yielded) is the length of the range `alive`.
// This range is decremented in length in either `next` or `next_back`. It is
// always decremented by 1 in those methods, but only if `Some(_)` is returned.
#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
unsafe impl<T, const N: usize, A: Allocator> TrustedLen for BoxedArrayIntoIter<T, N, A> {}

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
unsafe impl<T, const N: usize, A: Allocator> TrustedRandomAccessNoCoerce
    for BoxedArrayIntoIter<T, N, A>
where
    T: NonDrop,
{
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<T: Clone, const N: usize, A: Clone + Allocator> Clone for BoxedArrayIntoIter<T, N, A> {
    fn clone(&self) -> Self {
        // Note, we don't really need to match the exact same alive range, so
        // we can just clone into offset 0 regardless of where `self` is.

        let mut new = Self {
            data: Box::new_in(
                [const { MaybeUninit::uninit() }; N],
                Box::allocator(&self.data).clone(),
            ),
            alive: IndexRange::zero_to(0),
        };

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

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<T: fmt::Debug, const N: usize, A: Allocator> fmt::Debug for BoxedArrayIntoIter<T, N, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Only print the elements that were not yielded yet: we cannot
        // access the yielded elements anymore.
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<T, const N: usize, A: Allocator> IntoIterator for Box<[T; N], A> {
    type IntoIter = BoxedArrayIntoIter<T, N, A>;
    type Item = T;
    fn into_iter(self) -> BoxedArrayIntoIter<T, N, A> {
        // SAFETY: This essentially does a transmute of `[T; N]` -> `[MaybeUninit<T>; N]`,
        // this is explicitly allowed as by the `MaybeUninit` docs.
        let data = unsafe {
            let (ptr, alloc) = Box::into_non_null_with_allocator(self);
            Box::from_non_null_in(ptr.cast(), alloc)
        };
        BoxedArrayIntoIter { data, alive: IndexRange::zero_to(N) }
    }
}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<'a, T, const N: usize, A: Allocator> IntoIterator for &'a Box<[T; N], A> {
    type IntoIter = slice::Iter<'a, T>;
    type Item = &'a T;
    fn into_iter(self) -> slice::Iter<'a, T> {
        self.iter()
    }
}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<'a, T, const N: usize, A: Allocator> IntoIterator for &'a mut Box<[T; N], A> {
    type IntoIter = slice::IterMut<'a, T>;
    type Item = &'a mut T;
    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.iter_mut()
    }
}
