//! Slice management and manipulation.
//!
//! For more details see [`std::slice`].
//!
//! [`std::slice`]: ../../std/slice/index.html

#![stable(feature = "rust1", since = "1.0.0")]

use crate::cmp::Ordering::{self, Equal, Greater, Less};
use crate::intrinsics::{exact_div, select_unpredictable, unchecked_sub};
use crate::mem::{self, SizedTypeProperties};
use crate::num::NonZero;
use crate::ops::{Bound, OneSidedRange, Range, RangeBounds, RangeInclusive};
use crate::panic::const_panic;
use crate::simd::{self, Simd};
use crate::ub_checks::assert_unsafe_precondition;
use crate::{fmt, hint, ptr, range, slice};

#[unstable(
    feature = "slice_internals",
    issue = "none",
    reason = "exposed from core to be reused in std; use the memchr crate"
)]
/// Pure Rust memchr implementation, taken from rust-memchr
pub mod memchr;

#[unstable(
    feature = "slice_internals",
    issue = "none",
    reason = "exposed from core to be reused in std;"
)]
#[doc(hidden)]
pub mod sort;

mod ascii;
mod cmp;
pub(crate) mod index;
mod iter;
mod raw;
mod rotate;
mod specialize;

#[stable(feature = "inherent_ascii_escape", since = "1.60.0")]
pub use ascii::EscapeAscii;
#[unstable(feature = "str_internals", issue = "none")]
#[doc(hidden)]
pub use ascii::is_ascii_simple;
#[stable(feature = "slice_get_slice", since = "1.28.0")]
pub use index::SliceIndex;
#[unstable(feature = "slice_range", issue = "76393")]
pub use index::{range, try_range};
#[unstable(feature = "array_windows", issue = "75027")]
pub use iter::ArrayWindows;
#[unstable(feature = "array_chunks", issue = "74985")]
pub use iter::{ArrayChunks, ArrayChunksMut};
#[stable(feature = "slice_group_by", since = "1.77.0")]
pub use iter::{ChunkBy, ChunkByMut};
#[stable(feature = "rust1", since = "1.0.0")]
pub use iter::{Chunks, ChunksMut, Windows};
#[stable(feature = "chunks_exact", since = "1.31.0")]
pub use iter::{ChunksExact, ChunksExactMut};
#[stable(feature = "rust1", since = "1.0.0")]
pub use iter::{Iter, IterMut};
#[stable(feature = "rchunks", since = "1.31.0")]
pub use iter::{RChunks, RChunksExact, RChunksExactMut, RChunksMut};
#[stable(feature = "slice_rsplit", since = "1.27.0")]
pub use iter::{RSplit, RSplitMut};
#[stable(feature = "rust1", since = "1.0.0")]
pub use iter::{RSplitN, RSplitNMut, Split, SplitMut, SplitN, SplitNMut};
#[stable(feature = "split_inclusive", since = "1.51.0")]
pub use iter::{SplitInclusive, SplitInclusiveMut};
#[stable(feature = "from_ref", since = "1.28.0")]
pub use raw::{from_mut, from_ref};
#[unstable(feature = "slice_from_ptr_range", issue = "89792")]
pub use raw::{from_mut_ptr_range, from_ptr_range};
#[stable(feature = "rust1", since = "1.0.0")]
pub use raw::{from_raw_parts, from_raw_parts_mut};

/// Calculates the direction and split point of a one-sided range.
///
/// This is a helper function for `take` and `take_mut` that returns
/// the direction of the split (front or back) as well as the index at
/// which to split. Returns `None` if the split index would overflow.
#[inline]
fn split_point_of(range: impl OneSidedRange<usize>) -> Option<(Direction, usize)> {
    use Bound::*;

    Some(match (range.start_bound(), range.end_bound()) {
        (Unbounded, Excluded(i)) => (Direction::Front, *i),
        (Unbounded, Included(i)) => (Direction::Front, i.checked_add(1)?),
        (Excluded(i), Unbounded) => (Direction::Back, i.checked_add(1)?),
        (Included(i), Unbounded) => (Direction::Back, *i),
        _ => unreachable!(),
    })
}

enum Direction {
    Front,
    Back,
}

#[cfg(not(test))]
impl<T> [T] {
    /// Returns the number of elements in the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert_eq!(a.len(), 3);
    /// ```
    #[lang = "slice_len_fn"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_slice_len", since = "1.39.0")]
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        ptr::metadata(self)
    }

    /// Returns `true` if the slice has a length of 0.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert!(!a.is_empty());
    ///
    /// let b: &[i32] = &[];
    /// assert!(b.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_slice_is_empty", since = "1.39.0")]
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the first element of the slice, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert_eq!(Some(&10), v.first());
    ///
    /// let w: &[i32] = &[];
    /// assert_eq!(None, w.first());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_slice_first_last_not_mut", since = "1.56.0")]
    #[inline]
    #[must_use]
    pub const fn first(&self) -> Option<&T> {
        if let [first, ..] = self { Some(first) } else { None }
    }

    /// Returns a mutable reference to the first element of the slice, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [0, 1, 2];
    ///
    /// if let Some(first) = x.first_mut() {
    ///     *first = 5;
    /// }
    /// assert_eq!(x, &[5, 1, 2]);
    ///
    /// let y: &mut [i32] = &mut [];
    /// assert_eq!(None, y.first_mut());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_slice_first_last", since = "1.83.0")]
    #[inline]
    #[must_use]
    pub const fn first_mut(&mut self) -> Option<&mut T> {
        if let [first, ..] = self { Some(first) } else { None }
    }

    /// Returns the first and all the rest of the elements of the slice, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &[0, 1, 2];
    ///
    /// if let Some((first, elements)) = x.split_first() {
    ///     assert_eq!(first, &0);
    ///     assert_eq!(elements, &[1, 2]);
    /// }
    /// ```
    #[stable(feature = "slice_splits", since = "1.5.0")]
    #[rustc_const_stable(feature = "const_slice_first_last_not_mut", since = "1.56.0")]
    #[inline]
    #[must_use]
    pub const fn split_first(&self) -> Option<(&T, &[T])> {
        if let [first, tail @ ..] = self { Some((first, tail)) } else { None }
    }

    /// Returns the first and all the rest of the elements of the slice, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [0, 1, 2];
    ///
    /// if let Some((first, elements)) = x.split_first_mut() {
    ///     *first = 3;
    ///     elements[0] = 4;
    ///     elements[1] = 5;
    /// }
    /// assert_eq!(x, &[3, 4, 5]);
    /// ```
    #[stable(feature = "slice_splits", since = "1.5.0")]
    #[rustc_const_stable(feature = "const_slice_first_last", since = "1.83.0")]
    #[inline]
    #[must_use]
    pub const fn split_first_mut(&mut self) -> Option<(&mut T, &mut [T])> {
        if let [first, tail @ ..] = self { Some((first, tail)) } else { None }
    }

    /// Returns the last and all the rest of the elements of the slice, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &[0, 1, 2];
    ///
    /// if let Some((last, elements)) = x.split_last() {
    ///     assert_eq!(last, &2);
    ///     assert_eq!(elements, &[0, 1]);
    /// }
    /// ```
    #[stable(feature = "slice_splits", since = "1.5.0")]
    #[rustc_const_stable(feature = "const_slice_first_last_not_mut", since = "1.56.0")]
    #[inline]
    #[must_use]
    pub const fn split_last(&self) -> Option<(&T, &[T])> {
        if let [init @ .., last] = self { Some((last, init)) } else { None }
    }

    /// Returns the last and all the rest of the elements of the slice, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [0, 1, 2];
    ///
    /// if let Some((last, elements)) = x.split_last_mut() {
    ///     *last = 3;
    ///     elements[0] = 4;
    ///     elements[1] = 5;
    /// }
    /// assert_eq!(x, &[4, 5, 3]);
    /// ```
    #[stable(feature = "slice_splits", since = "1.5.0")]
    #[rustc_const_stable(feature = "const_slice_first_last", since = "1.83.0")]
    #[inline]
    #[must_use]
    pub const fn split_last_mut(&mut self) -> Option<(&mut T, &mut [T])> {
        if let [init @ .., last] = self { Some((last, init)) } else { None }
    }

    /// Returns the last element of the slice, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert_eq!(Some(&30), v.last());
    ///
    /// let w: &[i32] = &[];
    /// assert_eq!(None, w.last());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_slice_first_last_not_mut", since = "1.56.0")]
    #[inline]
    #[must_use]
    pub const fn last(&self) -> Option<&T> {
        if let [.., last] = self { Some(last) } else { None }
    }

    /// Returns a mutable reference to the last item in the slice, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [0, 1, 2];
    ///
    /// if let Some(last) = x.last_mut() {
    ///     *last = 10;
    /// }
    /// assert_eq!(x, &[0, 1, 10]);
    ///
    /// let y: &mut [i32] = &mut [];
    /// assert_eq!(None, y.last_mut());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_slice_first_last", since = "1.83.0")]
    #[inline]
    #[must_use]
    pub const fn last_mut(&mut self) -> Option<&mut T> {
        if let [.., last] = self { Some(last) } else { None }
    }

    /// Returns an array reference to the first `N` items in the slice.
    ///
    /// If the slice is not at least `N` in length, this will return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// let u = [10, 40, 30];
    /// assert_eq!(Some(&[10, 40]), u.first_chunk::<2>());
    ///
    /// let v: &[i32] = &[10];
    /// assert_eq!(None, v.first_chunk::<2>());
    ///
    /// let w: &[i32] = &[];
    /// assert_eq!(Some(&[]), w.first_chunk::<0>());
    /// ```
    #[inline]
    #[stable(feature = "slice_first_last_chunk", since = "1.77.0")]
    #[rustc_const_stable(feature = "slice_first_last_chunk", since = "1.77.0")]
    pub const fn first_chunk<const N: usize>(&self) -> Option<&[T; N]> {
        if self.len() < N {
            None
        } else {
            // SAFETY: We explicitly check for the correct number of elements,
            //   and do not let the reference outlive the slice.
            Some(unsafe { &*(self.as_ptr().cast::<[T; N]>()) })
        }
    }

    /// Returns a mutable array reference to the first `N` items in the slice.
    ///
    /// If the slice is not at least `N` in length, this will return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [0, 1, 2];
    ///
    /// if let Some(first) = x.first_chunk_mut::<2>() {
    ///     first[0] = 5;
    ///     first[1] = 4;
    /// }
    /// assert_eq!(x, &[5, 4, 2]);
    ///
    /// assert_eq!(None, x.first_chunk_mut::<4>());
    /// ```
    #[inline]
    #[stable(feature = "slice_first_last_chunk", since = "1.77.0")]
    #[rustc_const_stable(feature = "const_slice_first_last_chunk", since = "1.83.0")]
    pub const fn first_chunk_mut<const N: usize>(&mut self) -> Option<&mut [T; N]> {
        if self.len() < N {
            None
        } else {
            // SAFETY: We explicitly check for the correct number of elements,
            //   do not let the reference outlive the slice,
            //   and require exclusive access to the entire slice to mutate the chunk.
            Some(unsafe { &mut *(self.as_mut_ptr().cast::<[T; N]>()) })
        }
    }

    /// Returns an array reference to the first `N` items in the slice and the remaining slice.
    ///
    /// If the slice is not at least `N` in length, this will return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &[0, 1, 2];
    ///
    /// if let Some((first, elements)) = x.split_first_chunk::<2>() {
    ///     assert_eq!(first, &[0, 1]);
    ///     assert_eq!(elements, &[2]);
    /// }
    ///
    /// assert_eq!(None, x.split_first_chunk::<4>());
    /// ```
    #[inline]
    #[stable(feature = "slice_first_last_chunk", since = "1.77.0")]
    #[rustc_const_stable(feature = "slice_first_last_chunk", since = "1.77.0")]
    pub const fn split_first_chunk<const N: usize>(&self) -> Option<(&[T; N], &[T])> {
        if self.len() < N {
            None
        } else {
            // SAFETY: We manually verified the bounds of the split.
            let (first, tail) = unsafe { self.split_at_unchecked(N) };

            // SAFETY: We explicitly check for the correct number of elements,
            //   and do not let the references outlive the slice.
            Some((unsafe { &*(first.as_ptr().cast::<[T; N]>()) }, tail))
        }
    }

    /// Returns a mutable array reference to the first `N` items in the slice and the remaining
    /// slice.
    ///
    /// If the slice is not at least `N` in length, this will return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [0, 1, 2];
    ///
    /// if let Some((first, elements)) = x.split_first_chunk_mut::<2>() {
    ///     first[0] = 3;
    ///     first[1] = 4;
    ///     elements[0] = 5;
    /// }
    /// assert_eq!(x, &[3, 4, 5]);
    ///
    /// assert_eq!(None, x.split_first_chunk_mut::<4>());
    /// ```
    #[inline]
    #[stable(feature = "slice_first_last_chunk", since = "1.77.0")]
    #[rustc_const_stable(feature = "const_slice_first_last_chunk", since = "1.83.0")]
    pub const fn split_first_chunk_mut<const N: usize>(
        &mut self,
    ) -> Option<(&mut [T; N], &mut [T])> {
        if self.len() < N {
            None
        } else {
            // SAFETY: We manually verified the bounds of the split.
            let (first, tail) = unsafe { self.split_at_mut_unchecked(N) };

            // SAFETY: We explicitly check for the correct number of elements,
            //   do not let the reference outlive the slice,
            //   and enforce exclusive mutability of the chunk by the split.
            Some((unsafe { &mut *(first.as_mut_ptr().cast::<[T; N]>()) }, tail))
        }
    }

    /// Returns an array reference to the last `N` items in the slice and the remaining slice.
    ///
    /// If the slice is not at least `N` in length, this will return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &[0, 1, 2];
    ///
    /// if let Some((elements, last)) = x.split_last_chunk::<2>() {
    ///     assert_eq!(elements, &[0]);
    ///     assert_eq!(last, &[1, 2]);
    /// }
    ///
    /// assert_eq!(None, x.split_last_chunk::<4>());
    /// ```
    #[inline]
    #[stable(feature = "slice_first_last_chunk", since = "1.77.0")]
    #[rustc_const_stable(feature = "slice_first_last_chunk", since = "1.77.0")]
    pub const fn split_last_chunk<const N: usize>(&self) -> Option<(&[T], &[T; N])> {
        if self.len() < N {
            None
        } else {
            // SAFETY: We manually verified the bounds of the split.
            let (init, last) = unsafe { self.split_at_unchecked(self.len() - N) };

            // SAFETY: We explicitly check for the correct number of elements,
            //   and do not let the references outlive the slice.
            Some((init, unsafe { &*(last.as_ptr().cast::<[T; N]>()) }))
        }
    }

    /// Returns a mutable array reference to the last `N` items in the slice and the remaining
    /// slice.
    ///
    /// If the slice is not at least `N` in length, this will return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [0, 1, 2];
    ///
    /// if let Some((elements, last)) = x.split_last_chunk_mut::<2>() {
    ///     last[0] = 3;
    ///     last[1] = 4;
    ///     elements[0] = 5;
    /// }
    /// assert_eq!(x, &[5, 3, 4]);
    ///
    /// assert_eq!(None, x.split_last_chunk_mut::<4>());
    /// ```
    #[inline]
    #[stable(feature = "slice_first_last_chunk", since = "1.77.0")]
    #[rustc_const_stable(feature = "const_slice_first_last_chunk", since = "1.83.0")]
    pub const fn split_last_chunk_mut<const N: usize>(
        &mut self,
    ) -> Option<(&mut [T], &mut [T; N])> {
        if self.len() < N {
            None
        } else {
            // SAFETY: We manually verified the bounds of the split.
            let (init, last) = unsafe { self.split_at_mut_unchecked(self.len() - N) };

            // SAFETY: We explicitly check for the correct number of elements,
            //   do not let the reference outlive the slice,
            //   and enforce exclusive mutability of the chunk by the split.
            Some((init, unsafe { &mut *(last.as_mut_ptr().cast::<[T; N]>()) }))
        }
    }

    /// Returns an array reference to the last `N` items in the slice.
    ///
    /// If the slice is not at least `N` in length, this will return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// let u = [10, 40, 30];
    /// assert_eq!(Some(&[40, 30]), u.last_chunk::<2>());
    ///
    /// let v: &[i32] = &[10];
    /// assert_eq!(None, v.last_chunk::<2>());
    ///
    /// let w: &[i32] = &[];
    /// assert_eq!(Some(&[]), w.last_chunk::<0>());
    /// ```
    #[inline]
    #[stable(feature = "slice_first_last_chunk", since = "1.77.0")]
    #[rustc_const_stable(feature = "const_slice_last_chunk", since = "1.80.0")]
    pub const fn last_chunk<const N: usize>(&self) -> Option<&[T; N]> {
        if self.len() < N {
            None
        } else {
            // SAFETY: We manually verified the bounds of the slice.
            // FIXME(const-hack): Without const traits, we need this instead of `get_unchecked`.
            let last = unsafe { self.split_at_unchecked(self.len() - N).1 };

            // SAFETY: We explicitly check for the correct number of elements,
            //   and do not let the references outlive the slice.
            Some(unsafe { &*(last.as_ptr().cast::<[T; N]>()) })
        }
    }

    /// Returns a mutable array reference to the last `N` items in the slice.
    ///
    /// If the slice is not at least `N` in length, this will return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [0, 1, 2];
    ///
    /// if let Some(last) = x.last_chunk_mut::<2>() {
    ///     last[0] = 10;
    ///     last[1] = 20;
    /// }
    /// assert_eq!(x, &[0, 10, 20]);
    ///
    /// assert_eq!(None, x.last_chunk_mut::<4>());
    /// ```
    #[inline]
    #[stable(feature = "slice_first_last_chunk", since = "1.77.0")]
    #[rustc_const_stable(feature = "const_slice_first_last_chunk", since = "1.83.0")]
    pub const fn last_chunk_mut<const N: usize>(&mut self) -> Option<&mut [T; N]> {
        if self.len() < N {
            None
        } else {
            // SAFETY: We manually verified the bounds of the slice.
            // FIXME(const-hack): Without const traits, we need this instead of `get_unchecked`.
            let last = unsafe { self.split_at_mut_unchecked(self.len() - N).1 };

            // SAFETY: We explicitly check for the correct number of elements,
            //   do not let the reference outlive the slice,
            //   and require exclusive access to the entire slice to mutate the chunk.
            Some(unsafe { &mut *(last.as_mut_ptr().cast::<[T; N]>()) })
        }
    }

    /// Returns a reference to an element or subslice depending on the type of
    /// index.
    ///
    /// - If given a position, returns a reference to the element at that
    ///   position or `None` if out of bounds.
    /// - If given a range, returns the subslice corresponding to that range,
    ///   or `None` if out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert_eq!(Some(&40), v.get(1));
    /// assert_eq!(Some(&[10, 40][..]), v.get(0..2));
    /// assert_eq!(None, v.get(3));
    /// assert_eq!(None, v.get(0..4));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    #[must_use]
    pub fn get<I>(&self, index: I) -> Option<&I::Output>
    where
        I: SliceIndex<Self>,
    {
        index.get(self)
    }

    /// Returns a mutable reference to an element or subslice depending on the
    /// type of index (see [`get`]) or `None` if the index is out of bounds.
    ///
    /// [`get`]: slice::get
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [0, 1, 2];
    ///
    /// if let Some(elem) = x.get_mut(1) {
    ///     *elem = 42;
    /// }
    /// assert_eq!(x, &[0, 42, 2]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    #[must_use]
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut I::Output>
    where
        I: SliceIndex<Self>,
    {
        index.get_mut(self)
    }

    /// Returns a reference to an element or subslice, without doing bounds
    /// checking.
    ///
    /// For a safe alternative see [`get`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    ///
    /// You can think of this like `.get(index).unwrap_unchecked()`.  It's UB
    /// to call `.get_unchecked(len)`, even if you immediately convert to a
    /// pointer.  And it's UB to call `.get_unchecked(..len + 1)`,
    /// `.get_unchecked(..=len)`, or similar.
    ///
    /// [`get`]: slice::get
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &[1, 2, 4];
    ///
    /// unsafe {
    ///     assert_eq!(x.get_unchecked(1), &2);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    #[must_use]
    pub unsafe fn get_unchecked<I>(&self, index: I) -> &I::Output
    where
        I: SliceIndex<Self>,
    {
        // SAFETY: the caller must uphold most of the safety requirements for `get_unchecked`;
        // the slice is dereferenceable because `self` is a safe reference.
        // The returned pointer is safe because impls of `SliceIndex` have to guarantee that it is.
        unsafe { &*index.get_unchecked(self) }
    }

    /// Returns a mutable reference to an element or subslice, without doing
    /// bounds checking.
    ///
    /// For a safe alternative see [`get_mut`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    ///
    /// You can think of this like `.get_mut(index).unwrap_unchecked()`.  It's
    /// UB to call `.get_unchecked_mut(len)`, even if you immediately convert
    /// to a pointer.  And it's UB to call `.get_unchecked_mut(..len + 1)`,
    /// `.get_unchecked_mut(..=len)`, or similar.
    ///
    /// [`get_mut`]: slice::get_mut
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [1, 2, 4];
    ///
    /// unsafe {
    ///     let elem = x.get_unchecked_mut(1);
    ///     *elem = 13;
    /// }
    /// assert_eq!(x, &[1, 13, 4]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    #[must_use]
    pub unsafe fn get_unchecked_mut<I>(&mut self, index: I) -> &mut I::Output
    where
        I: SliceIndex<Self>,
    {
        // SAFETY: the caller must uphold the safety requirements for `get_unchecked_mut`;
        // the slice is dereferenceable because `self` is a safe reference.
        // The returned pointer is safe because impls of `SliceIndex` have to guarantee that it is.
        unsafe { &mut *index.get_unchecked_mut(self) }
    }

    /// Returns a raw pointer to the slice's buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this
    /// function returns, or else it will end up dangling.
    ///
    /// The caller must also ensure that the memory the pointer (non-transitively) points to
    /// is never written to (except inside an `UnsafeCell`) using this pointer or any pointer
    /// derived from it. If you need to mutate the contents of the slice, use [`as_mut_ptr`].
    ///
    /// Modifying the container referenced by this slice may cause its buffer
    /// to be reallocated, which would also make any pointers to it invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &[1, 2, 4];
    /// let x_ptr = x.as_ptr();
    ///
    /// unsafe {
    ///     for i in 0..x.len() {
    ///         assert_eq!(x.get_unchecked(i), &*x_ptr.add(i));
    ///     }
    /// }
    /// ```
    ///
    /// [`as_mut_ptr`]: slice::as_mut_ptr
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_slice_as_ptr", since = "1.32.0")]
    #[rustc_never_returns_null_ptr]
    #[rustc_as_ptr]
    #[inline(always)]
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self as *const [T] as *const T
    }

    /// Returns an unsafe mutable pointer to the slice's buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this
    /// function returns, or else it will end up dangling.
    ///
    /// Modifying the container referenced by this slice may cause its buffer
    /// to be reallocated, which would also make any pointers to it invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [1, 2, 4];
    /// let x_ptr = x.as_mut_ptr();
    ///
    /// unsafe {
    ///     for i in 0..x.len() {
    ///         *x_ptr.add(i) += 2;
    ///     }
    /// }
    /// assert_eq!(x, &[3, 4, 6]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_ptr_offset", since = "1.61.0")]
    #[rustc_never_returns_null_ptr]
    #[rustc_as_ptr]
    #[inline(always)]
    #[must_use]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self as *mut [T] as *mut T
    }

    /// Returns the two raw pointers spanning the slice.
    ///
    /// The returned range is half-open, which means that the end pointer
    /// points *one past* the last element of the slice. This way, an empty
    /// slice is represented by two equal pointers, and the difference between
    /// the two pointers represents the size of the slice.
    ///
    /// See [`as_ptr`] for warnings on using these pointers. The end pointer
    /// requires extra caution, as it does not point to a valid element in the
    /// slice.
    ///
    /// This function is useful for interacting with foreign interfaces which
    /// use two pointers to refer to a range of elements in memory, as is
    /// common in C++.
    ///
    /// It can also be useful to check if a pointer to an element refers to an
    /// element of this slice:
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// let x = &a[1] as *const _;
    /// let y = &5 as *const _;
    ///
    /// assert!(a.as_ptr_range().contains(&x));
    /// assert!(!a.as_ptr_range().contains(&y));
    /// ```
    ///
    /// [`as_ptr`]: slice::as_ptr
    #[stable(feature = "slice_ptr_range", since = "1.48.0")]
    #[rustc_const_stable(feature = "const_ptr_offset", since = "1.61.0")]
    #[inline]
    #[must_use]
    pub const fn as_ptr_range(&self) -> Range<*const T> {
        let start = self.as_ptr();
        // SAFETY: The `add` here is safe, because:
        //
        //   - Both pointers are part of the same object, as pointing directly
        //     past the object also counts.
        //
        //   - The size of the slice is never larger than `isize::MAX` bytes, as
        //     noted here:
        //       - https://github.com/rust-lang/unsafe-code-guidelines/issues/102#issuecomment-473340447
        //       - https://doc.rust-lang.org/reference/behavior-considered-undefined.html
        //       - https://doc.rust-lang.org/core/slice/fn.from_raw_parts.html#safety
        //     (This doesn't seem normative yet, but the very same assumption is
        //     made in many places, including the Index implementation of slices.)
        //
        //   - There is no wrapping around involved, as slices do not wrap past
        //     the end of the address space.
        //
        // See the documentation of [`pointer::add`].
        let end = unsafe { start.add(self.len()) };
        start..end
    }

    /// Returns the two unsafe mutable pointers spanning the slice.
    ///
    /// The returned range is half-open, which means that the end pointer
    /// points *one past* the last element of the slice. This way, an empty
    /// slice is represented by two equal pointers, and the difference between
    /// the two pointers represents the size of the slice.
    ///
    /// See [`as_mut_ptr`] for warnings on using these pointers. The end
    /// pointer requires extra caution, as it does not point to a valid element
    /// in the slice.
    ///
    /// This function is useful for interacting with foreign interfaces which
    /// use two pointers to refer to a range of elements in memory, as is
    /// common in C++.
    ///
    /// [`as_mut_ptr`]: slice::as_mut_ptr
    #[stable(feature = "slice_ptr_range", since = "1.48.0")]
    #[rustc_const_stable(feature = "const_ptr_offset", since = "1.61.0")]
    #[inline]
    #[must_use]
    pub const fn as_mut_ptr_range(&mut self) -> Range<*mut T> {
        let start = self.as_mut_ptr();
        // SAFETY: See as_ptr_range() above for why `add` here is safe.
        let end = unsafe { start.add(self.len()) };
        start..end
    }

    /// Gets a reference to the underlying array.
    ///
    /// If `N` is not exactly equal to the length of `self`, then this method returns `None`.
    #[unstable(feature = "slice_as_array", issue = "133508")]
    #[inline]
    #[must_use]
    pub const fn as_array<const N: usize>(&self) -> Option<&[T; N]> {
        if self.len() == N {
            let ptr = self.as_ptr() as *const [T; N];

            // SAFETY: The underlying array of a slice can be reinterpreted as an actual array `[T; N]` if `N` is not greater than the slice's length.
            let me = unsafe { &*ptr };
            Some(me)
        } else {
            None
        }
    }

    /// Gets a mutable reference to the slice's underlying array.
    ///
    /// If `N` is not exactly equal to the length of `self`, then this method returns `None`.
    #[unstable(feature = "slice_as_array", issue = "133508")]
    #[inline]
    #[must_use]
    pub const fn as_mut_array<const N: usize>(&mut self) -> Option<&mut [T; N]> {
        if self.len() == N {
            let ptr = self.as_mut_ptr() as *mut [T; N];

            // SAFETY: The underlying array of a slice can be reinterpreted as an actual array `[T; N]` if `N` is not greater than the slice's length.
            let me = unsafe { &mut *ptr };
            Some(me)
        } else {
            None
        }
    }

    /// Swaps two elements in the slice.
    ///
    /// If `a` equals to `b`, it's guaranteed that elements won't change value.
    ///
    /// # Arguments
    ///
    /// * a - The index of the first element
    /// * b - The index of the second element
    ///
    /// # Panics
    ///
    /// Panics if `a` or `b` are out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = ["a", "b", "c", "d", "e"];
    /// v.swap(2, 4);
    /// assert!(v == ["a", "b", "e", "d", "c"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_swap", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    #[track_caller]
    pub const fn swap(&mut self, a: usize, b: usize) {
        // FIXME: use swap_unchecked here (https://github.com/rust-lang/rust/pull/88540#issuecomment-944344343)
        // Can't take two mutable loans from one vector, so instead use raw pointers.
        let pa = &raw mut self[a];
        let pb = &raw mut self[b];
        // SAFETY: `pa` and `pb` have been created from safe mutable references and refer
        // to elements in the slice and therefore are guaranteed to be valid and aligned.
        // Note that accessing the elements behind `a` and `b` is checked and will
        // panic when out of bounds.
        unsafe {
            ptr::swap(pa, pb);
        }
    }

    /// Swaps two elements in the slice, without doing bounds checking.
    ///
    /// For a safe alternative see [`swap`].
    ///
    /// # Arguments
    ///
    /// * a - The index of the first element
    /// * b - The index of the second element
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*.
    /// The caller has to ensure that `a < self.len()` and `b < self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_swap_unchecked)]
    ///
    /// let mut v = ["a", "b", "c", "d"];
    /// // SAFETY: we know that 1 and 3 are both indices of the slice
    /// unsafe { v.swap_unchecked(1, 3) };
    /// assert!(v == ["a", "d", "c", "b"]);
    /// ```
    ///
    /// [`swap`]: slice::swap
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[unstable(feature = "slice_swap_unchecked", issue = "88539")]
    #[rustc_const_unstable(feature = "slice_swap_unchecked", issue = "88539")]
    pub const unsafe fn swap_unchecked(&mut self, a: usize, b: usize) {
        assert_unsafe_precondition!(
            check_library_ub,
            "slice::swap_unchecked requires that the indices are within the slice",
            (
                len: usize = self.len(),
                a: usize = a,
                b: usize = b,
            ) => a < len && b < len,
        );

        let ptr = self.as_mut_ptr();
        // SAFETY: caller has to guarantee that `a < self.len()` and `b < self.len()`
        unsafe {
            ptr::swap(ptr.add(a), ptr.add(b));
        }
    }

    /// Reverses the order of elements in the slice, in place.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [1, 2, 3];
    /// v.reverse();
    /// assert!(v == [3, 2, 1]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn reverse(&mut self) {
        let half_len = self.len() / 2;
        let Range { start, end } = self.as_mut_ptr_range();

        // These slices will skip the middle item for an odd length,
        // since that one doesn't need to move.
        let (front_half, back_half) =
            // SAFETY: Both are subparts of the original slice, so the memory
            // range is valid, and they don't overlap because they're each only
            // half (or less) of the original slice.
            unsafe {
                (
                    slice::from_raw_parts_mut(start, half_len),
                    slice::from_raw_parts_mut(end.sub(half_len), half_len),
                )
            };

        // Introducing a function boundary here means that the two halves
        // get `noalias` markers, allowing better optimization as LLVM
        // knows that they're disjoint, unlike in the original slice.
        revswap(front_half, back_half, half_len);

        #[inline]
        fn revswap<T>(a: &mut [T], b: &mut [T], n: usize) {
            debug_assert!(a.len() == n);
            debug_assert!(b.len() == n);

            // Because this function is first compiled in isolation,
            // this check tells LLVM that the indexing below is
            // in-bounds. Then after inlining -- once the actual
            // lengths of the slices are known -- it's removed.
            let (a, b) = (&mut a[..n], &mut b[..n]);

            let mut i = 0;
            while i < n {
                mem::swap(&mut a[i], &mut b[n - 1 - i]);
                i += 1;
            }
        }
    }

    /// Returns an iterator over the slice.
    ///
    /// The iterator yields all items from start to end.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &[1, 2, 4];
    /// let mut iterator = x.iter();
    ///
    /// assert_eq!(iterator.next(), Some(&1));
    /// assert_eq!(iterator.next(), Some(&2));
    /// assert_eq!(iterator.next(), Some(&4));
    /// assert_eq!(iterator.next(), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    #[cfg_attr(not(test), rustc_diagnostic_item = "slice_iter")]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(self)
    }

    /// Returns an iterator that allows modifying each value.
    ///
    /// The iterator yields all items from start to end.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = &mut [1, 2, 4];
    /// for elem in x.iter_mut() {
    ///     *elem += 2;
    /// }
    /// assert_eq!(x, &[3, 4, 6]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut::new(self)
    }

    /// Returns an iterator over all contiguous windows of length
    /// `size`. The windows overlap. If the slice is shorter than
    /// `size`, the iterator returns no values.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// let slice = ['l', 'o', 'r', 'e', 'm'];
    /// let mut iter = slice.windows(3);
    /// assert_eq!(iter.next().unwrap(), &['l', 'o', 'r']);
    /// assert_eq!(iter.next().unwrap(), &['o', 'r', 'e']);
    /// assert_eq!(iter.next().unwrap(), &['r', 'e', 'm']);
    /// assert!(iter.next().is_none());
    /// ```
    ///
    /// If the slice is shorter than `size`:
    ///
    /// ```
    /// let slice = ['f', 'o', 'o'];
    /// let mut iter = slice.windows(4);
    /// assert!(iter.next().is_none());
    /// ```
    ///
    /// There's no `windows_mut`, as that existing would let safe code violate the
    /// "only one `&mut` at a time to the same thing" rule.  However, you can sometimes
    /// use [`Cell::as_slice_of_cells`](crate::cell::Cell::as_slice_of_cells) in
    /// conjunction with `windows` to accomplish something similar:
    /// ```
    /// use std::cell::Cell;
    ///
    /// let mut array = ['R', 'u', 's', 't', ' ', '2', '0', '1', '5'];
    /// let slice = &mut array[..];
    /// let slice_of_cells: &[Cell<char>] = Cell::from_mut(slice).as_slice_of_cells();
    /// for w in slice_of_cells.windows(3) {
    ///     Cell::swap(&w[0], &w[2]);
    /// }
    /// assert_eq!(array, ['s', 't', ' ', '2', '0', '1', '5', 'u', 'R']);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    #[track_caller]
    pub fn windows(&self, size: usize) -> Windows<'_, T> {
        let size = NonZero::new(size).expect("window size must be non-zero");
        Windows::new(self, size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time, starting at the
    /// beginning of the slice.
    ///
    /// The chunks are slices and do not overlap. If `chunk_size` does not divide the length of the
    /// slice, then the last chunk will not have length `chunk_size`.
    ///
    /// See [`chunks_exact`] for a variant of this iterator that returns chunks of always exactly
    /// `chunk_size` elements, and [`rchunks`] for the same iterator but starting at the end of the
    /// slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// let slice = ['l', 'o', 'r', 'e', 'm'];
    /// let mut iter = slice.chunks(2);
    /// assert_eq!(iter.next().unwrap(), &['l', 'o']);
    /// assert_eq!(iter.next().unwrap(), &['r', 'e']);
    /// assert_eq!(iter.next().unwrap(), &['m']);
    /// assert!(iter.next().is_none());
    /// ```
    ///
    /// [`chunks_exact`]: slice::chunks_exact
    /// [`rchunks`]: slice::rchunks
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    #[track_caller]
    pub fn chunks(&self, chunk_size: usize) -> Chunks<'_, T> {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        Chunks::new(self, chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time, starting at the
    /// beginning of the slice.
    ///
    /// The chunks are mutable slices, and do not overlap. If `chunk_size` does not divide the
    /// length of the slice, then the last chunk will not have length `chunk_size`.
    ///
    /// See [`chunks_exact_mut`] for a variant of this iterator that returns chunks of always
    /// exactly `chunk_size` elements, and [`rchunks_mut`] for the same iterator but starting at
    /// the end of the slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = &mut [0, 0, 0, 0, 0];
    /// let mut count = 1;
    ///
    /// for chunk in v.chunks_mut(2) {
    ///     for elem in chunk.iter_mut() {
    ///         *elem += count;
    ///     }
    ///     count += 1;
    /// }
    /// assert_eq!(v, &[1, 1, 2, 2, 3]);
    /// ```
    ///
    /// [`chunks_exact_mut`]: slice::chunks_exact_mut
    /// [`rchunks_mut`]: slice::rchunks_mut
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    #[track_caller]
    pub fn chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<'_, T> {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        ChunksMut::new(self, chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time, starting at the
    /// beginning of the slice.
    ///
    /// The chunks are slices and do not overlap. If `chunk_size` does not divide the length of the
    /// slice, then the last up to `chunk_size-1` elements will be omitted and can be retrieved
    /// from the `remainder` function of the iterator.
    ///
    /// Due to each chunk having exactly `chunk_size` elements, the compiler can often optimize the
    /// resulting code better than in the case of [`chunks`].
    ///
    /// See [`chunks`] for a variant of this iterator that also returns the remainder as a smaller
    /// chunk, and [`rchunks_exact`] for the same iterator but starting at the end of the slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// let slice = ['l', 'o', 'r', 'e', 'm'];
    /// let mut iter = slice.chunks_exact(2);
    /// assert_eq!(iter.next().unwrap(), &['l', 'o']);
    /// assert_eq!(iter.next().unwrap(), &['r', 'e']);
    /// assert!(iter.next().is_none());
    /// assert_eq!(iter.remainder(), &['m']);
    /// ```
    ///
    /// [`chunks`]: slice::chunks
    /// [`rchunks_exact`]: slice::rchunks_exact
    #[stable(feature = "chunks_exact", since = "1.31.0")]
    #[inline]
    #[track_caller]
    pub fn chunks_exact(&self, chunk_size: usize) -> ChunksExact<'_, T> {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        ChunksExact::new(self, chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time, starting at the
    /// beginning of the slice.
    ///
    /// The chunks are mutable slices, and do not overlap. If `chunk_size` does not divide the
    /// length of the slice, then the last up to `chunk_size-1` elements will be omitted and can be
    /// retrieved from the `into_remainder` function of the iterator.
    ///
    /// Due to each chunk having exactly `chunk_size` elements, the compiler can often optimize the
    /// resulting code better than in the case of [`chunks_mut`].
    ///
    /// See [`chunks_mut`] for a variant of this iterator that also returns the remainder as a
    /// smaller chunk, and [`rchunks_exact_mut`] for the same iterator but starting at the end of
    /// the slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = &mut [0, 0, 0, 0, 0];
    /// let mut count = 1;
    ///
    /// for chunk in v.chunks_exact_mut(2) {
    ///     for elem in chunk.iter_mut() {
    ///         *elem += count;
    ///     }
    ///     count += 1;
    /// }
    /// assert_eq!(v, &[1, 1, 2, 2, 0]);
    /// ```
    ///
    /// [`chunks_mut`]: slice::chunks_mut
    /// [`rchunks_exact_mut`]: slice::rchunks_exact_mut
    #[stable(feature = "chunks_exact", since = "1.31.0")]
    #[inline]
    #[track_caller]
    pub fn chunks_exact_mut(&mut self, chunk_size: usize) -> ChunksExactMut<'_, T> {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        ChunksExactMut::new(self, chunk_size)
    }

    /// Splits the slice into a slice of `N`-element arrays,
    /// assuming that there's no remainder.
    ///
    /// # Safety
    ///
    /// This may only be called when
    /// - The slice splits exactly into `N`-element chunks (aka `self.len() % N == 0`).
    /// - `N != 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_as_chunks)]
    /// let slice: &[char] = &['l', 'o', 'r', 'e', 'm', '!'];
    /// let chunks: &[[char; 1]] =
    ///     // SAFETY: 1-element chunks never have remainder
    ///     unsafe { slice.as_chunks_unchecked() };
    /// assert_eq!(chunks, &[['l'], ['o'], ['r'], ['e'], ['m'], ['!']]);
    /// let chunks: &[[char; 3]] =
    ///     // SAFETY: The slice length (6) is a multiple of 3
    ///     unsafe { slice.as_chunks_unchecked() };
    /// assert_eq!(chunks, &[['l', 'o', 'r'], ['e', 'm', '!']]);
    ///
    /// // These would be unsound:
    /// // let chunks: &[[_; 5]] = slice.as_chunks_unchecked() // The slice length is not a multiple of 5
    /// // let chunks: &[[_; 0]] = slice.as_chunks_unchecked() // Zero-length chunks are never allowed
    /// ```
    #[unstable(feature = "slice_as_chunks", issue = "74985")]
    #[rustc_const_unstable(feature = "slice_as_chunks", issue = "74985")]
    #[inline]
    #[must_use]
    pub const unsafe fn as_chunks_unchecked<const N: usize>(&self) -> &[[T; N]] {
        assert_unsafe_precondition!(
            check_language_ub,
            "slice::as_chunks_unchecked requires `N != 0` and the slice to split exactly into `N`-element chunks",
            (n: usize = N, len: usize = self.len()) => n != 0 && len % n == 0,
        );
        // SAFETY: Caller must guarantee that `N` is nonzero and exactly divides the slice length
        let new_len = unsafe { exact_div(self.len(), N) };
        // SAFETY: We cast a slice of `new_len * N` elements into
        // a slice of `new_len` many `N` elements chunks.
        unsafe { from_raw_parts(self.as_ptr().cast(), new_len) }
    }

    /// Splits the slice into a slice of `N`-element arrays,
    /// starting at the beginning of the slice,
    /// and a remainder slice with length strictly less than `N`.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero. This check will most probably get changed to a compile time
    /// error before this method gets stabilized.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_as_chunks)]
    /// let slice = ['l', 'o', 'r', 'e', 'm'];
    /// let (chunks, remainder) = slice.as_chunks();
    /// assert_eq!(chunks, &[['l', 'o'], ['r', 'e']]);
    /// assert_eq!(remainder, &['m']);
    /// ```
    ///
    /// If you expect the slice to be an exact multiple, you can combine
    /// `let`-`else` with an empty slice pattern:
    /// ```
    /// #![feature(slice_as_chunks)]
    /// let slice = ['R', 'u', 's', 't'];
    /// let (chunks, []) = slice.as_chunks::<2>() else {
    ///     panic!("slice didn't have even length")
    /// };
    /// assert_eq!(chunks, &[['R', 'u'], ['s', 't']]);
    /// ```
    #[unstable(feature = "slice_as_chunks", issue = "74985")]
    #[rustc_const_unstable(feature = "slice_as_chunks", issue = "74985")]
    #[inline]
    #[track_caller]
    #[must_use]
    pub const fn as_chunks<const N: usize>(&self) -> (&[[T; N]], &[T]) {
        assert!(N != 0, "chunk size must be non-zero");
        let len_rounded_down = self.len() / N * N;
        // SAFETY: The rounded-down value is always the same or smaller than the
        // original length, and thus must be in-bounds of the slice.
        let (multiple_of_n, remainder) = unsafe { self.split_at_unchecked(len_rounded_down) };
        // SAFETY: We already panicked for zero, and ensured by construction
        // that the length of the subslice is a multiple of N.
        let array_slice = unsafe { multiple_of_n.as_chunks_unchecked() };
        (array_slice, remainder)
    }

    /// Splits the slice into a slice of `N`-element arrays,
    /// starting at the end of the slice,
    /// and a remainder slice with length strictly less than `N`.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero. This check will most probably get changed to a compile time
    /// error before this method gets stabilized.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_as_chunks)]
    /// let slice = ['l', 'o', 'r', 'e', 'm'];
    /// let (remainder, chunks) = slice.as_rchunks();
    /// assert_eq!(remainder, &['l']);
    /// assert_eq!(chunks, &[['o', 'r'], ['e', 'm']]);
    /// ```
    #[unstable(feature = "slice_as_chunks", issue = "74985")]
    #[rustc_const_unstable(feature = "slice_as_chunks", issue = "74985")]
    #[inline]
    #[track_caller]
    #[must_use]
    pub const fn as_rchunks<const N: usize>(&self) -> (&[T], &[[T; N]]) {
        assert!(N != 0, "chunk size must be non-zero");
        let len = self.len() / N;
        let (remainder, multiple_of_n) = self.split_at(self.len() - len * N);
        // SAFETY: We already panicked for zero, and ensured by construction
        // that the length of the subslice is a multiple of N.
        let array_slice = unsafe { multiple_of_n.as_chunks_unchecked() };
        (remainder, array_slice)
    }

    /// Returns an iterator over `N` elements of the slice at a time, starting at the
    /// beginning of the slice.
    ///
    /// The chunks are array references and do not overlap. If `N` does not divide the
    /// length of the slice, then the last up to `N-1` elements will be omitted and can be
    /// retrieved from the `remainder` function of the iterator.
    ///
    /// This method is the const generic equivalent of [`chunks_exact`].
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero. This check will most probably get changed to a compile time
    /// error before this method gets stabilized.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(array_chunks)]
    /// let slice = ['l', 'o', 'r', 'e', 'm'];
    /// let mut iter = slice.array_chunks();
    /// assert_eq!(iter.next().unwrap(), &['l', 'o']);
    /// assert_eq!(iter.next().unwrap(), &['r', 'e']);
    /// assert!(iter.next().is_none());
    /// assert_eq!(iter.remainder(), &['m']);
    /// ```
    ///
    /// [`chunks_exact`]: slice::chunks_exact
    #[unstable(feature = "array_chunks", issue = "74985")]
    #[inline]
    #[track_caller]
    pub fn array_chunks<const N: usize>(&self) -> ArrayChunks<'_, T, N> {
        assert!(N != 0, "chunk size must be non-zero");
        ArrayChunks::new(self)
    }

    /// Splits the slice into a slice of `N`-element arrays,
    /// assuming that there's no remainder.
    ///
    /// # Safety
    ///
    /// This may only be called when
    /// - The slice splits exactly into `N`-element chunks (aka `self.len() % N == 0`).
    /// - `N != 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_as_chunks)]
    /// let slice: &mut [char] = &mut ['l', 'o', 'r', 'e', 'm', '!'];
    /// let chunks: &mut [[char; 1]] =
    ///     // SAFETY: 1-element chunks never have remainder
    ///     unsafe { slice.as_chunks_unchecked_mut() };
    /// chunks[0] = ['L'];
    /// assert_eq!(chunks, &[['L'], ['o'], ['r'], ['e'], ['m'], ['!']]);
    /// let chunks: &mut [[char; 3]] =
    ///     // SAFETY: The slice length (6) is a multiple of 3
    ///     unsafe { slice.as_chunks_unchecked_mut() };
    /// chunks[1] = ['a', 'x', '?'];
    /// assert_eq!(slice, &['L', 'o', 'r', 'a', 'x', '?']);
    ///
    /// // These would be unsound:
    /// // let chunks: &[[_; 5]] = slice.as_chunks_unchecked_mut() // The slice length is not a multiple of 5
    /// // let chunks: &[[_; 0]] = slice.as_chunks_unchecked_mut() // Zero-length chunks are never allowed
    /// ```
    #[unstable(feature = "slice_as_chunks", issue = "74985")]
    #[rustc_const_unstable(feature = "slice_as_chunks", issue = "74985")]
    #[inline]
    #[must_use]
    pub const unsafe fn as_chunks_unchecked_mut<const N: usize>(&mut self) -> &mut [[T; N]] {
        assert_unsafe_precondition!(
            check_language_ub,
            "slice::as_chunks_unchecked requires `N != 0` and the slice to split exactly into `N`-element chunks",
            (n: usize = N, len: usize = self.len()) => n != 0 && len % n == 0
        );
        // SAFETY: Caller must guarantee that `N` is nonzero and exactly divides the slice length
        let new_len = unsafe { exact_div(self.len(), N) };
        // SAFETY: We cast a slice of `new_len * N` elements into
        // a slice of `new_len` many `N` elements chunks.
        unsafe { from_raw_parts_mut(self.as_mut_ptr().cast(), new_len) }
    }

    /// Splits the slice into a slice of `N`-element arrays,
    /// starting at the beginning of the slice,
    /// and a remainder slice with length strictly less than `N`.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero. This check will most probably get changed to a compile time
    /// error before this method gets stabilized.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_as_chunks)]
    /// let v = &mut [0, 0, 0, 0, 0];
    /// let mut count = 1;
    ///
    /// let (chunks, remainder) = v.as_chunks_mut();
    /// remainder[0] = 9;
    /// for chunk in chunks {
    ///     *chunk = [count; 2];
    ///     count += 1;
    /// }
    /// assert_eq!(v, &[1, 1, 2, 2, 9]);
    /// ```
    #[unstable(feature = "slice_as_chunks", issue = "74985")]
    #[rustc_const_unstable(feature = "slice_as_chunks", issue = "74985")]
    #[inline]
    #[track_caller]
    #[must_use]
    pub const fn as_chunks_mut<const N: usize>(&mut self) -> (&mut [[T; N]], &mut [T]) {
        assert!(N != 0, "chunk size must be non-zero");
        let len_rounded_down = self.len() / N * N;
        // SAFETY: The rounded-down value is always the same or smaller than the
        // original length, and thus must be in-bounds of the slice.
        let (multiple_of_n, remainder) = unsafe { self.split_at_mut_unchecked(len_rounded_down) };
        // SAFETY: We already panicked for zero, and ensured by construction
        // that the length of the subslice is a multiple of N.
        let array_slice = unsafe { multiple_of_n.as_chunks_unchecked_mut() };
        (array_slice, remainder)
    }

    /// Splits the slice into a slice of `N`-element arrays,
    /// starting at the end of the slice,
    /// and a remainder slice with length strictly less than `N`.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero. This check will most probably get changed to a compile time
    /// error before this method gets stabilized.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_as_chunks)]
    /// let v = &mut [0, 0, 0, 0, 0];
    /// let mut count = 1;
    ///
    /// let (remainder, chunks) = v.as_rchunks_mut();
    /// remainder[0] = 9;
    /// for chunk in chunks {
    ///     *chunk = [count; 2];
    ///     count += 1;
    /// }
    /// assert_eq!(v, &[9, 1, 1, 2, 2]);
    /// ```
    #[unstable(feature = "slice_as_chunks", issue = "74985")]
    #[rustc_const_unstable(feature = "slice_as_chunks", issue = "74985")]
    #[inline]
    #[track_caller]
    #[must_use]
    pub const fn as_rchunks_mut<const N: usize>(&mut self) -> (&mut [T], &mut [[T; N]]) {
        assert!(N != 0, "chunk size must be non-zero");
        let len = self.len() / N;
        let (remainder, multiple_of_n) = self.split_at_mut(self.len() - len * N);
        // SAFETY: We already panicked for zero, and ensured by construction
        // that the length of the subslice is a multiple of N.
        let array_slice = unsafe { multiple_of_n.as_chunks_unchecked_mut() };
        (remainder, array_slice)
    }

    /// Returns an iterator over `N` elements of the slice at a time, starting at the
    /// beginning of the slice.
    ///
    /// The chunks are mutable array references and do not overlap. If `N` does not divide
    /// the length of the slice, then the last up to `N-1` elements will be omitted and
    /// can be retrieved from the `into_remainder` function of the iterator.
    ///
    /// This method is the const generic equivalent of [`chunks_exact_mut`].
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero. This check will most probably get changed to a compile time
    /// error before this method gets stabilized.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(array_chunks)]
    /// let v = &mut [0, 0, 0, 0, 0];
    /// let mut count = 1;
    ///
    /// for chunk in v.array_chunks_mut() {
    ///     *chunk = [count; 2];
    ///     count += 1;
    /// }
    /// assert_eq!(v, &[1, 1, 2, 2, 0]);
    /// ```
    ///
    /// [`chunks_exact_mut`]: slice::chunks_exact_mut
    #[unstable(feature = "array_chunks", issue = "74985")]
    #[inline]
    #[track_caller]
    pub fn array_chunks_mut<const N: usize>(&mut self) -> ArrayChunksMut<'_, T, N> {
        assert!(N != 0, "chunk size must be non-zero");
        ArrayChunksMut::new(self)
    }

    /// Returns an iterator over overlapping windows of `N` elements of a slice,
    /// starting at the beginning of the slice.
    ///
    /// This is the const generic equivalent of [`windows`].
    ///
    /// If `N` is greater than the size of the slice, it will return no windows.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero. This check will most probably get changed to a compile time
    /// error before this method gets stabilized.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(array_windows)]
    /// let slice = [0, 1, 2, 3];
    /// let mut iter = slice.array_windows();
    /// assert_eq!(iter.next().unwrap(), &[0, 1]);
    /// assert_eq!(iter.next().unwrap(), &[1, 2]);
    /// assert_eq!(iter.next().unwrap(), &[2, 3]);
    /// assert!(iter.next().is_none());
    /// ```
    ///
    /// [`windows`]: slice::windows
    #[unstable(feature = "array_windows", issue = "75027")]
    #[inline]
    #[track_caller]
    pub fn array_windows<const N: usize>(&self) -> ArrayWindows<'_, T, N> {
        assert!(N != 0, "window size must be non-zero");
        ArrayWindows::new(self)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time, starting at the end
    /// of the slice.
    ///
    /// The chunks are slices and do not overlap. If `chunk_size` does not divide the length of the
    /// slice, then the last chunk will not have length `chunk_size`.
    ///
    /// See [`rchunks_exact`] for a variant of this iterator that returns chunks of always exactly
    /// `chunk_size` elements, and [`chunks`] for the same iterator but starting at the beginning
    /// of the slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// let slice = ['l', 'o', 'r', 'e', 'm'];
    /// let mut iter = slice.rchunks(2);
    /// assert_eq!(iter.next().unwrap(), &['e', 'm']);
    /// assert_eq!(iter.next().unwrap(), &['o', 'r']);
    /// assert_eq!(iter.next().unwrap(), &['l']);
    /// assert!(iter.next().is_none());
    /// ```
    ///
    /// [`rchunks_exact`]: slice::rchunks_exact
    /// [`chunks`]: slice::chunks
    #[stable(feature = "rchunks", since = "1.31.0")]
    #[inline]
    #[track_caller]
    pub fn rchunks(&self, chunk_size: usize) -> RChunks<'_, T> {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        RChunks::new(self, chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time, starting at the end
    /// of the slice.
    ///
    /// The chunks are mutable slices, and do not overlap. If `chunk_size` does not divide the
    /// length of the slice, then the last chunk will not have length `chunk_size`.
    ///
    /// See [`rchunks_exact_mut`] for a variant of this iterator that returns chunks of always
    /// exactly `chunk_size` elements, and [`chunks_mut`] for the same iterator but starting at the
    /// beginning of the slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = &mut [0, 0, 0, 0, 0];
    /// let mut count = 1;
    ///
    /// for chunk in v.rchunks_mut(2) {
    ///     for elem in chunk.iter_mut() {
    ///         *elem += count;
    ///     }
    ///     count += 1;
    /// }
    /// assert_eq!(v, &[3, 2, 2, 1, 1]);
    /// ```
    ///
    /// [`rchunks_exact_mut`]: slice::rchunks_exact_mut
    /// [`chunks_mut`]: slice::chunks_mut
    #[stable(feature = "rchunks", since = "1.31.0")]
    #[inline]
    #[track_caller]
    pub fn rchunks_mut(&mut self, chunk_size: usize) -> RChunksMut<'_, T> {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        RChunksMut::new(self, chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time, starting at the
    /// end of the slice.
    ///
    /// The chunks are slices and do not overlap. If `chunk_size` does not divide the length of the
    /// slice, then the last up to `chunk_size-1` elements will be omitted and can be retrieved
    /// from the `remainder` function of the iterator.
    ///
    /// Due to each chunk having exactly `chunk_size` elements, the compiler can often optimize the
    /// resulting code better than in the case of [`rchunks`].
    ///
    /// See [`rchunks`] for a variant of this iterator that also returns the remainder as a smaller
    /// chunk, and [`chunks_exact`] for the same iterator but starting at the beginning of the
    /// slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// let slice = ['l', 'o', 'r', 'e', 'm'];
    /// let mut iter = slice.rchunks_exact(2);
    /// assert_eq!(iter.next().unwrap(), &['e', 'm']);
    /// assert_eq!(iter.next().unwrap(), &['o', 'r']);
    /// assert!(iter.next().is_none());
    /// assert_eq!(iter.remainder(), &['l']);
    /// ```
    ///
    /// [`chunks`]: slice::chunks
    /// [`rchunks`]: slice::rchunks
    /// [`chunks_exact`]: slice::chunks_exact
    #[stable(feature = "rchunks", since = "1.31.0")]
    #[inline]
    #[track_caller]
    pub fn rchunks_exact(&self, chunk_size: usize) -> RChunksExact<'_, T> {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        RChunksExact::new(self, chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time, starting at the end
    /// of the slice.
    ///
    /// The chunks are mutable slices, and do not overlap. If `chunk_size` does not divide the
    /// length of the slice, then the last up to `chunk_size-1` elements will be omitted and can be
    /// retrieved from the `into_remainder` function of the iterator.
    ///
    /// Due to each chunk having exactly `chunk_size` elements, the compiler can often optimize the
    /// resulting code better than in the case of [`chunks_mut`].
    ///
    /// See [`rchunks_mut`] for a variant of this iterator that also returns the remainder as a
    /// smaller chunk, and [`chunks_exact_mut`] for the same iterator but starting at the beginning
    /// of the slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = &mut [0, 0, 0, 0, 0];
    /// let mut count = 1;
    ///
    /// for chunk in v.rchunks_exact_mut(2) {
    ///     for elem in chunk.iter_mut() {
    ///         *elem += count;
    ///     }
    ///     count += 1;
    /// }
    /// assert_eq!(v, &[0, 2, 2, 1, 1]);
    /// ```
    ///
    /// [`chunks_mut`]: slice::chunks_mut
    /// [`rchunks_mut`]: slice::rchunks_mut
    /// [`chunks_exact_mut`]: slice::chunks_exact_mut
    #[stable(feature = "rchunks", since = "1.31.0")]
    #[inline]
    #[track_caller]
    pub fn rchunks_exact_mut(&mut self, chunk_size: usize) -> RChunksExactMut<'_, T> {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        RChunksExactMut::new(self, chunk_size)
    }

    /// Returns an iterator over the slice producing non-overlapping runs
    /// of elements using the predicate to separate them.
    ///
    /// The predicate is called for every pair of consecutive elements,
    /// meaning that it is called on `slice[0]` and `slice[1]`,
    /// followed by `slice[1]` and `slice[2]`, and so on.
    ///
    /// # Examples
    ///
    /// ```
    /// let slice = &[1, 1, 1, 3, 3, 2, 2, 2];
    ///
    /// let mut iter = slice.chunk_by(|a, b| a == b);
    ///
    /// assert_eq!(iter.next(), Some(&[1, 1, 1][..]));
    /// assert_eq!(iter.next(), Some(&[3, 3][..]));
    /// assert_eq!(iter.next(), Some(&[2, 2, 2][..]));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// This method can be used to extract the sorted subslices:
    ///
    /// ```
    /// let slice = &[1, 1, 2, 3, 2, 3, 2, 3, 4];
    ///
    /// let mut iter = slice.chunk_by(|a, b| a <= b);
    ///
    /// assert_eq!(iter.next(), Some(&[1, 1, 2, 3][..]));
    /// assert_eq!(iter.next(), Some(&[2, 3][..]));
    /// assert_eq!(iter.next(), Some(&[2, 3, 4][..]));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[stable(feature = "slice_group_by", since = "1.77.0")]
    #[inline]
    pub fn chunk_by<F>(&self, pred: F) -> ChunkBy<'_, T, F>
    where
        F: FnMut(&T, &T) -> bool,
    {
        ChunkBy::new(self, pred)
    }

    /// Returns an iterator over the slice producing non-overlapping mutable
    /// runs of elements using the predicate to separate them.
    ///
    /// The predicate is called for every pair of consecutive elements,
    /// meaning that it is called on `slice[0]` and `slice[1]`,
    /// followed by `slice[1]` and `slice[2]`, and so on.
    ///
    /// # Examples
    ///
    /// ```
    /// let slice = &mut [1, 1, 1, 3, 3, 2, 2, 2];
    ///
    /// let mut iter = slice.chunk_by_mut(|a, b| a == b);
    ///
    /// assert_eq!(iter.next(), Some(&mut [1, 1, 1][..]));
    /// assert_eq!(iter.next(), Some(&mut [3, 3][..]));
    /// assert_eq!(iter.next(), Some(&mut [2, 2, 2][..]));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// This method can be used to extract the sorted subslices:
    ///
    /// ```
    /// let slice = &mut [1, 1, 2, 3, 2, 3, 2, 3, 4];
    ///
    /// let mut iter = slice.chunk_by_mut(|a, b| a <= b);
    ///
    /// assert_eq!(iter.next(), Some(&mut [1, 1, 2, 3][..]));
    /// assert_eq!(iter.next(), Some(&mut [2, 3][..]));
    /// assert_eq!(iter.next(), Some(&mut [2, 3, 4][..]));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[stable(feature = "slice_group_by", since = "1.77.0")]
    #[inline]
    pub fn chunk_by_mut<F>(&mut self, pred: F) -> ChunkByMut<'_, T, F>
    where
        F: FnMut(&T, &T) -> bool,
    {
        ChunkByMut::new(self, pred)
    }

    /// Divides one slice into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.  For a non-panicking alternative see
    /// [`split_at_checked`](slice::split_at_checked).
    ///
    /// # Examples
    ///
    /// ```
    /// let v = ['a', 'b', 'c'];
    ///
    /// {
    ///    let (left, right) = v.split_at(0);
    ///    assert_eq!(left, []);
    ///    assert_eq!(right, ['a', 'b', 'c']);
    /// }
    ///
    /// {
    ///     let (left, right) = v.split_at(2);
    ///     assert_eq!(left, ['a', 'b']);
    ///     assert_eq!(right, ['c']);
    /// }
    ///
    /// {
    ///     let (left, right) = v.split_at(3);
    ///     assert_eq!(left, ['a', 'b', 'c']);
    ///     assert_eq!(right, []);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_slice_split_at_not_mut", since = "1.71.0")]
    #[inline]
    #[track_caller]
    #[must_use]
    pub const fn split_at(&self, mid: usize) -> (&[T], &[T]) {
        match self.split_at_checked(mid) {
            Some(pair) => pair,
            None => panic!("mid > len"),
        }
    }

    /// Divides one mutable slice into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.  For a non-panicking alternative see
    /// [`split_at_mut_checked`](slice::split_at_mut_checked).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [1, 0, 3, 0, 5, 6];
    /// let (left, right) = v.split_at_mut(2);
    /// assert_eq!(left, [1, 0]);
    /// assert_eq!(right, [3, 0, 5, 6]);
    /// left[1] = 2;
    /// right[1] = 4;
    /// assert_eq!(v, [1, 2, 3, 4, 5, 6]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    #[track_caller]
    #[must_use]
    #[rustc_const_stable(feature = "const_slice_split_at_mut", since = "1.83.0")]
    pub const fn split_at_mut(&mut self, mid: usize) -> (&mut [T], &mut [T]) {
        match self.split_at_mut_checked(mid) {
            Some(pair) => pair,
            None => panic!("mid > len"),
        }
    }

    /// Divides one slice into two at an index, without doing bounds checking.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// For a safe alternative see [`split_at`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used. The caller has to ensure that
    /// `0 <= mid <= self.len()`.
    ///
    /// [`split_at`]: slice::split_at
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// let v = ['a', 'b', 'c'];
    ///
    /// unsafe {
    ///    let (left, right) = v.split_at_unchecked(0);
    ///    assert_eq!(left, []);
    ///    assert_eq!(right, ['a', 'b', 'c']);
    /// }
    ///
    /// unsafe {
    ///     let (left, right) = v.split_at_unchecked(2);
    ///     assert_eq!(left, ['a', 'b']);
    ///     assert_eq!(right, ['c']);
    /// }
    ///
    /// unsafe {
    ///     let (left, right) = v.split_at_unchecked(3);
    ///     assert_eq!(left, ['a', 'b', 'c']);
    ///     assert_eq!(right, []);
    /// }
    /// ```
    #[stable(feature = "slice_split_at_unchecked", since = "1.79.0")]
    #[rustc_const_stable(feature = "const_slice_split_at_unchecked", since = "1.77.0")]
    #[inline]
    #[must_use]
    pub const unsafe fn split_at_unchecked(&self, mid: usize) -> (&[T], &[T]) {
        // FIXME(const-hack): the const function `from_raw_parts` is used to make this
        // function const; previously the implementation used
        // `(self.get_unchecked(..mid), self.get_unchecked(mid..))`

        let len = self.len();
        let ptr = self.as_ptr();

        assert_unsafe_precondition!(
            check_library_ub,
            "slice::split_at_unchecked requires the index to be within the slice",
            (mid: usize = mid, len: usize = len) => mid <= len,
        );

        // SAFETY: Caller has to check that `0 <= mid <= self.len()`
        unsafe { (from_raw_parts(ptr, mid), from_raw_parts(ptr.add(mid), unchecked_sub(len, mid))) }
    }

    /// Divides one mutable slice into two at an index, without doing bounds checking.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// For a safe alternative see [`split_at_mut`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used. The caller has to ensure that
    /// `0 <= mid <= self.len()`.
    ///
    /// [`split_at_mut`]: slice::split_at_mut
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [1, 0, 3, 0, 5, 6];
    /// // scoped to restrict the lifetime of the borrows
    /// unsafe {
    ///     let (left, right) = v.split_at_mut_unchecked(2);
    ///     assert_eq!(left, [1, 0]);
    ///     assert_eq!(right, [3, 0, 5, 6]);
    ///     left[1] = 2;
    ///     right[1] = 4;
    /// }
    /// assert_eq!(v, [1, 2, 3, 4, 5, 6]);
    /// ```
    #[stable(feature = "slice_split_at_unchecked", since = "1.79.0")]
    #[rustc_const_stable(feature = "const_slice_split_at_mut", since = "1.83.0")]
    #[inline]
    #[must_use]
    pub const unsafe fn split_at_mut_unchecked(&mut self, mid: usize) -> (&mut [T], &mut [T]) {
        let len = self.len();
        let ptr = self.as_mut_ptr();

        assert_unsafe_precondition!(
            check_library_ub,
            "slice::split_at_mut_unchecked requires the index to be within the slice",
            (mid: usize = mid, len: usize = len) => mid <= len,
        );

        // SAFETY: Caller has to check that `0 <= mid <= self.len()`.
        //
        // `[ptr; mid]` and `[mid; len]` are not overlapping, so returning a mutable reference
        // is fine.
        unsafe {
            (
                from_raw_parts_mut(ptr, mid),
                from_raw_parts_mut(ptr.add(mid), unchecked_sub(len, mid)),
            )
        }
    }

    /// Divides one slice into two at an index, returning `None` if the slice is
    /// too short.
    ///
    /// If `mid ≤ len` returns a pair of slices where the first will contain all
    /// indices from `[0, mid)` (excluding the index `mid` itself) and the
    /// second will contain all indices from `[mid, len)` (excluding the index
    /// `len` itself).
    ///
    /// Otherwise, if `mid > len`, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [1, -2, 3, -4, 5, -6];
    ///
    /// {
    ///    let (left, right) = v.split_at_checked(0).unwrap();
    ///    assert_eq!(left, []);
    ///    assert_eq!(right, [1, -2, 3, -4, 5, -6]);
    /// }
    ///
    /// {
    ///     let (left, right) = v.split_at_checked(2).unwrap();
    ///     assert_eq!(left, [1, -2]);
    ///     assert_eq!(right, [3, -4, 5, -6]);
    /// }
    ///
    /// {
    ///     let (left, right) = v.split_at_checked(6).unwrap();
    ///     assert_eq!(left, [1, -2, 3, -4, 5, -6]);
    ///     assert_eq!(right, []);
    /// }
    ///
    /// assert_eq!(None, v.split_at_checked(7));
    /// ```
    #[stable(feature = "split_at_checked", since = "1.80.0")]
    #[rustc_const_stable(feature = "split_at_checked", since = "1.80.0")]
    #[inline]
    #[must_use]
    pub const fn split_at_checked(&self, mid: usize) -> Option<(&[T], &[T])> {
        if mid <= self.len() {
            // SAFETY: `[ptr; mid]` and `[mid; len]` are inside `self`, which
            // fulfills the requirements of `split_at_unchecked`.
            Some(unsafe { self.split_at_unchecked(mid) })
        } else {
            None
        }
    }

    /// Divides one mutable slice into two at an index, returning `None` if the
    /// slice is too short.
    ///
    /// If `mid ≤ len` returns a pair of slices where the first will contain all
    /// indices from `[0, mid)` (excluding the index `mid` itself) and the
    /// second will contain all indices from `[mid, len)` (excluding the index
    /// `len` itself).
    ///
    /// Otherwise, if `mid > len`, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [1, 0, 3, 0, 5, 6];
    ///
    /// if let Some((left, right)) = v.split_at_mut_checked(2) {
    ///     assert_eq!(left, [1, 0]);
    ///     assert_eq!(right, [3, 0, 5, 6]);
    ///     left[1] = 2;
    ///     right[1] = 4;
    /// }
    /// assert_eq!(v, [1, 2, 3, 4, 5, 6]);
    ///
    /// assert_eq!(None, v.split_at_mut_checked(7));
    /// ```
    #[stable(feature = "split_at_checked", since = "1.80.0")]
    #[rustc_const_stable(feature = "const_slice_split_at_mut", since = "1.83.0")]
    #[inline]
    #[must_use]
    pub const fn split_at_mut_checked(&mut self, mid: usize) -> Option<(&mut [T], &mut [T])> {
        if mid <= self.len() {
            // SAFETY: `[ptr; mid]` and `[mid; len]` are inside `self`, which
            // fulfills the requirements of `split_at_unchecked`.
            Some(unsafe { self.split_at_mut_unchecked(mid) })
        } else {
            None
        }
    }

    /// Returns an iterator over subslices separated by elements that match
    /// `pred`. The matched element is not contained in the subslices.
    ///
    /// # Examples
    ///
    /// ```
    /// let slice = [10, 40, 33, 20];
    /// let mut iter = slice.split(|num| num % 3 == 0);
    ///
    /// assert_eq!(iter.next().unwrap(), &[10, 40]);
    /// assert_eq!(iter.next().unwrap(), &[20]);
    /// assert!(iter.next().is_none());
    /// ```
    ///
    /// If the first element is matched, an empty slice will be the first item
    /// returned by the iterator. Similarly, if the last element in the slice
    /// is matched, an empty slice will be the last item returned by the
    /// iterator:
    ///
    /// ```
    /// let slice = [10, 40, 33];
    /// let mut iter = slice.split(|num| num % 3 == 0);
    ///
    /// assert_eq!(iter.next().unwrap(), &[10, 40]);
    /// assert_eq!(iter.next().unwrap(), &[]);
    /// assert!(iter.next().is_none());
    /// ```
    ///
    /// If two matched elements are directly adjacent, an empty slice will be
    /// present between them:
    ///
    /// ```
    /// let slice = [10, 6, 33, 20];
    /// let mut iter = slice.split(|num| num % 3 == 0);
    ///
    /// assert_eq!(iter.next().unwrap(), &[10]);
    /// assert_eq!(iter.next().unwrap(), &[]);
    /// assert_eq!(iter.next().unwrap(), &[20]);
    /// assert!(iter.next().is_none());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn split<F>(&self, pred: F) -> Split<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        Split::new(self, pred)
    }

    /// Returns an iterator over mutable subslices separated by elements that
    /// match `pred`. The matched element is not contained in the subslices.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [10, 40, 30, 20, 60, 50];
    ///
    /// for group in v.split_mut(|num| *num % 3 == 0) {
    ///     group[0] = 1;
    /// }
    /// assert_eq!(v, [1, 40, 30, 1, 60, 1]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn split_mut<F>(&mut self, pred: F) -> SplitMut<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        SplitMut::new(self, pred)
    }

    /// Returns an iterator over subslices separated by elements that match
    /// `pred`. The matched element is contained in the end of the previous
    /// subslice as a terminator.
    ///
    /// # Examples
    ///
    /// ```
    /// let slice = [10, 40, 33, 20];
    /// let mut iter = slice.split_inclusive(|num| num % 3 == 0);
    ///
    /// assert_eq!(iter.next().unwrap(), &[10, 40, 33]);
    /// assert_eq!(iter.next().unwrap(), &[20]);
    /// assert!(iter.next().is_none());
    /// ```
    ///
    /// If the last element of the slice is matched,
    /// that element will be considered the terminator of the preceding slice.
    /// That slice will be the last item returned by the iterator.
    ///
    /// ```
    /// let slice = [3, 10, 40, 33];
    /// let mut iter = slice.split_inclusive(|num| num % 3 == 0);
    ///
    /// assert_eq!(iter.next().unwrap(), &[3]);
    /// assert_eq!(iter.next().unwrap(), &[10, 40, 33]);
    /// assert!(iter.next().is_none());
    /// ```
    #[stable(feature = "split_inclusive", since = "1.51.0")]
    #[inline]
    pub fn split_inclusive<F>(&self, pred: F) -> SplitInclusive<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        SplitInclusive::new(self, pred)
    }

    /// Returns an iterator over mutable subslices separated by elements that
    /// match `pred`. The matched element is contained in the previous
    /// subslice as a terminator.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [10, 40, 30, 20, 60, 50];
    ///
    /// for group in v.split_inclusive_mut(|num| *num % 3 == 0) {
    ///     let terminator_idx = group.len()-1;
    ///     group[terminator_idx] = 1;
    /// }
    /// assert_eq!(v, [10, 40, 1, 20, 1, 1]);
    /// ```
    #[stable(feature = "split_inclusive", since = "1.51.0")]
    #[inline]
    pub fn split_inclusive_mut<F>(&mut self, pred: F) -> SplitInclusiveMut<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        SplitInclusiveMut::new(self, pred)
    }

    /// Returns an iterator over subslices separated by elements that match
    /// `pred`, starting at the end of the slice and working backwards.
    /// The matched element is not contained in the subslices.
    ///
    /// # Examples
    ///
    /// ```
    /// let slice = [11, 22, 33, 0, 44, 55];
    /// let mut iter = slice.rsplit(|num| *num == 0);
    ///
    /// assert_eq!(iter.next().unwrap(), &[44, 55]);
    /// assert_eq!(iter.next().unwrap(), &[11, 22, 33]);
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// As with `split()`, if the first or last element is matched, an empty
    /// slice will be the first (or last) item returned by the iterator.
    ///
    /// ```
    /// let v = &[0, 1, 1, 2, 3, 5, 8];
    /// let mut it = v.rsplit(|n| *n % 2 == 0);
    /// assert_eq!(it.next().unwrap(), &[]);
    /// assert_eq!(it.next().unwrap(), &[3, 5]);
    /// assert_eq!(it.next().unwrap(), &[1, 1]);
    /// assert_eq!(it.next().unwrap(), &[]);
    /// assert_eq!(it.next(), None);
    /// ```
    #[stable(feature = "slice_rsplit", since = "1.27.0")]
    #[inline]
    pub fn rsplit<F>(&self, pred: F) -> RSplit<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        RSplit::new(self, pred)
    }

    /// Returns an iterator over mutable subslices separated by elements that
    /// match `pred`, starting at the end of the slice and working
    /// backwards. The matched element is not contained in the subslices.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [100, 400, 300, 200, 600, 500];
    ///
    /// let mut count = 0;
    /// for group in v.rsplit_mut(|num| *num % 3 == 0) {
    ///     count += 1;
    ///     group[0] = count;
    /// }
    /// assert_eq!(v, [3, 400, 300, 2, 600, 1]);
    /// ```
    ///
    #[stable(feature = "slice_rsplit", since = "1.27.0")]
    #[inline]
    pub fn rsplit_mut<F>(&mut self, pred: F) -> RSplitMut<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        RSplitMut::new(self, pred)
    }

    /// Returns an iterator over subslices separated by elements that match
    /// `pred`, limited to returning at most `n` items. The matched element is
    /// not contained in the subslices.
    ///
    /// The last element returned, if any, will contain the remainder of the
    /// slice.
    ///
    /// # Examples
    ///
    /// Print the slice split once by numbers divisible by 3 (i.e., `[10, 40]`,
    /// `[20, 60, 50]`):
    ///
    /// ```
    /// let v = [10, 40, 30, 20, 60, 50];
    ///
    /// for group in v.splitn(2, |num| *num % 3 == 0) {
    ///     println!("{group:?}");
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn splitn<F>(&self, n: usize, pred: F) -> SplitN<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        SplitN::new(self.split(pred), n)
    }

    /// Returns an iterator over mutable subslices separated by elements that match
    /// `pred`, limited to returning at most `n` items. The matched element is
    /// not contained in the subslices.
    ///
    /// The last element returned, if any, will contain the remainder of the
    /// slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [10, 40, 30, 20, 60, 50];
    ///
    /// for group in v.splitn_mut(2, |num| *num % 3 == 0) {
    ///     group[0] = 1;
    /// }
    /// assert_eq!(v, [1, 40, 30, 1, 60, 50]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn splitn_mut<F>(&mut self, n: usize, pred: F) -> SplitNMut<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        SplitNMut::new(self.split_mut(pred), n)
    }

    /// Returns an iterator over subslices separated by elements that match
    /// `pred` limited to returning at most `n` items. This starts at the end of
    /// the slice and works backwards. The matched element is not contained in
    /// the subslices.
    ///
    /// The last element returned, if any, will contain the remainder of the
    /// slice.
    ///
    /// # Examples
    ///
    /// Print the slice split once, starting from the end, by numbers divisible
    /// by 3 (i.e., `[50]`, `[10, 40, 30, 20]`):
    ///
    /// ```
    /// let v = [10, 40, 30, 20, 60, 50];
    ///
    /// for group in v.rsplitn(2, |num| *num % 3 == 0) {
    ///     println!("{group:?}");
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn rsplitn<F>(&self, n: usize, pred: F) -> RSplitN<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        RSplitN::new(self.rsplit(pred), n)
    }

    /// Returns an iterator over subslices separated by elements that match
    /// `pred` limited to returning at most `n` items. This starts at the end of
    /// the slice and works backwards. The matched element is not contained in
    /// the subslices.
    ///
    /// The last element returned, if any, will contain the remainder of the
    /// slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = [10, 40, 30, 20, 60, 50];
    ///
    /// for group in s.rsplitn_mut(2, |num| *num % 3 == 0) {
    ///     group[0] = 1;
    /// }
    /// assert_eq!(s, [1, 40, 30, 20, 60, 1]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn rsplitn_mut<F>(&mut self, n: usize, pred: F) -> RSplitNMut<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        RSplitNMut::new(self.rsplit_mut(pred), n)
    }

    /// Splits the slice on the first element that matches the specified
    /// predicate.
    ///
    /// If any matching elements are present in the slice, returns the prefix
    /// before the match and suffix after. The matching element itself is not
    /// included. If no elements match, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_split_once)]
    /// let s = [1, 2, 3, 2, 4];
    /// assert_eq!(s.split_once(|&x| x == 2), Some((
    ///     &[1][..],
    ///     &[3, 2, 4][..]
    /// )));
    /// assert_eq!(s.split_once(|&x| x == 0), None);
    /// ```
    #[unstable(feature = "slice_split_once", reason = "newly added", issue = "112811")]
    #[inline]
    pub fn split_once<F>(&self, pred: F) -> Option<(&[T], &[T])>
    where
        F: FnMut(&T) -> bool,
    {
        let index = self.iter().position(pred)?;
        Some((&self[..index], &self[index + 1..]))
    }

    /// Splits the slice on the last element that matches the specified
    /// predicate.
    ///
    /// If any matching elements are present in the slice, returns the prefix
    /// before the match and suffix after. The matching element itself is not
    /// included. If no elements match, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_split_once)]
    /// let s = [1, 2, 3, 2, 4];
    /// assert_eq!(s.rsplit_once(|&x| x == 2), Some((
    ///     &[1, 2, 3][..],
    ///     &[4][..]
    /// )));
    /// assert_eq!(s.rsplit_once(|&x| x == 0), None);
    /// ```
    #[unstable(feature = "slice_split_once", reason = "newly added", issue = "112811")]
    #[inline]
    pub fn rsplit_once<F>(&self, pred: F) -> Option<(&[T], &[T])>
    where
        F: FnMut(&T) -> bool,
    {
        let index = self.iter().rposition(pred)?;
        Some((&self[..index], &self[index + 1..]))
    }

    /// Returns `true` if the slice contains an element with the given value.
    ///
    /// This operation is *O*(*n*).
    ///
    /// Note that if you have a sorted slice, [`binary_search`] may be faster.
    ///
    /// [`binary_search`]: slice::binary_search
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert!(v.contains(&30));
    /// assert!(!v.contains(&50));
    /// ```
    ///
    /// If you do not have a `&T`, but some other value that you can compare
    /// with one (for example, `String` implements `PartialEq<str>`), you can
    /// use `iter().any`:
    ///
    /// ```
    /// let v = [String::from("hello"), String::from("world")]; // slice of `String`
    /// assert!(v.iter().any(|e| e == "hello")); // search with `&str`
    /// assert!(!v.iter().any(|e| e == "hi"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    #[must_use]
    pub fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        cmp::SliceContains::slice_contains(x, self)
    }

    /// Returns `true` if `needle` is a prefix of the slice or equal to the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert!(v.starts_with(&[10]));
    /// assert!(v.starts_with(&[10, 40]));
    /// assert!(v.starts_with(&v));
    /// assert!(!v.starts_with(&[50]));
    /// assert!(!v.starts_with(&[10, 50]));
    /// ```
    ///
    /// Always returns `true` if `needle` is an empty slice:
    ///
    /// ```
    /// let v = &[10, 40, 30];
    /// assert!(v.starts_with(&[]));
    /// let v: &[u8] = &[];
    /// assert!(v.starts_with(&[]));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    pub fn starts_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        let n = needle.len();
        self.len() >= n && needle == &self[..n]
    }

    /// Returns `true` if `needle` is a suffix of the slice or equal to the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert!(v.ends_with(&[30]));
    /// assert!(v.ends_with(&[40, 30]));
    /// assert!(v.ends_with(&v));
    /// assert!(!v.ends_with(&[50]));
    /// assert!(!v.ends_with(&[50, 30]));
    /// ```
    ///
    /// Always returns `true` if `needle` is an empty slice:
    ///
    /// ```
    /// let v = &[10, 40, 30];
    /// assert!(v.ends_with(&[]));
    /// let v: &[u8] = &[];
    /// assert!(v.ends_with(&[]));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    pub fn ends_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        let (m, n) = (self.len(), needle.len());
        m >= n && needle == &self[m - n..]
    }

    /// Returns a subslice with the prefix removed.
    ///
    /// If the slice starts with `prefix`, returns the subslice after the prefix, wrapped in `Some`.
    /// If `prefix` is empty, simply returns the original slice. If `prefix` is equal to the
    /// original slice, returns an empty slice.
    ///
    /// If the slice does not start with `prefix`, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = &[10, 40, 30];
    /// assert_eq!(v.strip_prefix(&[10]), Some(&[40, 30][..]));
    /// assert_eq!(v.strip_prefix(&[10, 40]), Some(&[30][..]));
    /// assert_eq!(v.strip_prefix(&[10, 40, 30]), Some(&[][..]));
    /// assert_eq!(v.strip_prefix(&[50]), None);
    /// assert_eq!(v.strip_prefix(&[10, 50]), None);
    ///
    /// let prefix : &str = "he";
    /// assert_eq!(b"hello".strip_prefix(prefix.as_bytes()),
    ///            Some(b"llo".as_ref()));
    /// ```
    #[must_use = "returns the subslice without modifying the original"]
    #[stable(feature = "slice_strip", since = "1.51.0")]
    pub fn strip_prefix<P: SlicePattern<Item = T> + ?Sized>(&self, prefix: &P) -> Option<&[T]>
    where
        T: PartialEq,
    {
        // This function will need rewriting if and when SlicePattern becomes more sophisticated.
        let prefix = prefix.as_slice();
        let n = prefix.len();
        if n <= self.len() {
            let (head, tail) = self.split_at(n);
            if head == prefix {
                return Some(tail);
            }
        }
        None
    }

    /// Returns a subslice with the suffix removed.
    ///
    /// If the slice ends with `suffix`, returns the subslice before the suffix, wrapped in `Some`.
    /// If `suffix` is empty, simply returns the original slice. If `suffix` is equal to the
    /// original slice, returns an empty slice.
    ///
    /// If the slice does not end with `suffix`, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = &[10, 40, 30];
    /// assert_eq!(v.strip_suffix(&[30]), Some(&[10, 40][..]));
    /// assert_eq!(v.strip_suffix(&[40, 30]), Some(&[10][..]));
    /// assert_eq!(v.strip_suffix(&[10, 40, 30]), Some(&[][..]));
    /// assert_eq!(v.strip_suffix(&[50]), None);
    /// assert_eq!(v.strip_suffix(&[50, 30]), None);
    /// ```
    #[must_use = "returns the subslice without modifying the original"]
    #[stable(feature = "slice_strip", since = "1.51.0")]
    pub fn strip_suffix<P: SlicePattern<Item = T> + ?Sized>(&self, suffix: &P) -> Option<&[T]>
    where
        T: PartialEq,
    {
        // This function will need rewriting if and when SlicePattern becomes more sophisticated.
        let suffix = suffix.as_slice();
        let (len, n) = (self.len(), suffix.len());
        if n <= len {
            let (head, tail) = self.split_at(len - n);
            if tail == suffix {
                return Some(head);
            }
        }
        None
    }

    /// Binary searches this slice for a given element.
    /// If the slice is not sorted, the returned result is unspecified and
    /// meaningless.
    ///
    /// If the value is found then [`Result::Ok`] is returned, containing the
    /// index of the matching element. If there are multiple matches, then any
    /// one of the matches could be returned. The index is chosen
    /// deterministically, but is subject to change in future versions of Rust.
    /// If the value is not found then [`Result::Err`] is returned, containing
    /// the index where a matching element could be inserted while maintaining
    /// sorted order.
    ///
    /// See also [`binary_search_by`], [`binary_search_by_key`], and [`partition_point`].
    ///
    /// [`binary_search_by`]: slice::binary_search_by
    /// [`binary_search_by_key`]: slice::binary_search_by_key
    /// [`partition_point`]: slice::partition_point
    ///
    /// # Examples
    ///
    /// Looks up a series of four elements. The first is found, with a
    /// uniquely determined position; the second and third are not
    /// found; the fourth could match any position in `[1, 4]`.
    ///
    /// ```
    /// let s = [0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
    ///
    /// assert_eq!(s.binary_search(&13),  Ok(9));
    /// assert_eq!(s.binary_search(&4),   Err(7));
    /// assert_eq!(s.binary_search(&100), Err(13));
    /// let r = s.binary_search(&1);
    /// assert!(match r { Ok(1..=4) => true, _ => false, });
    /// ```
    ///
    /// If you want to find that whole *range* of matching items, rather than
    /// an arbitrary matching one, that can be done using [`partition_point`]:
    /// ```
    /// let s = [0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
    ///
    /// let low = s.partition_point(|x| x < &1);
    /// assert_eq!(low, 1);
    /// let high = s.partition_point(|x| x <= &1);
    /// assert_eq!(high, 5);
    /// let r = s.binary_search(&1);
    /// assert!((low..high).contains(&r.unwrap()));
    ///
    /// assert!(s[..low].iter().all(|&x| x < 1));
    /// assert!(s[low..high].iter().all(|&x| x == 1));
    /// assert!(s[high..].iter().all(|&x| x > 1));
    ///
    /// // For something not found, the "range" of equal items is empty
    /// assert_eq!(s.partition_point(|x| x < &11), 9);
    /// assert_eq!(s.partition_point(|x| x <= &11), 9);
    /// assert_eq!(s.binary_search(&11), Err(9));
    /// ```
    ///
    /// If you want to insert an item to a sorted vector, while maintaining
    /// sort order, consider using [`partition_point`]:
    ///
    /// ```
    /// let mut s = vec![0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
    /// let num = 42;
    /// let idx = s.partition_point(|&x| x <= num);
    /// // If `num` is unique, `s.partition_point(|&x| x < num)` (with `<`) is equivalent to
    /// // `s.binary_search(&num).unwrap_or_else(|x| x)`, but using `<=` will allow `insert`
    /// // to shift less elements.
    /// s.insert(idx, num);
    /// assert_eq!(s, [0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 42, 55]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn binary_search(&self, x: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.binary_search_by(|p| p.cmp(x))
    }

    /// Binary searches this slice with a comparator function.
    ///
    /// The comparator function should return an order code that indicates
    /// whether its argument is `Less`, `Equal` or `Greater` the desired
    /// target.
    /// If the slice is not sorted or if the comparator function does not
    /// implement an order consistent with the sort order of the underlying
    /// slice, the returned result is unspecified and meaningless.
    ///
    /// If the value is found then [`Result::Ok`] is returned, containing the
    /// index of the matching element. If there are multiple matches, then any
    /// one of the matches could be returned. The index is chosen
    /// deterministically, but is subject to change in future versions of Rust.
    /// If the value is not found then [`Result::Err`] is returned, containing
    /// the index where a matching element could be inserted while maintaining
    /// sorted order.
    ///
    /// See also [`binary_search`], [`binary_search_by_key`], and [`partition_point`].
    ///
    /// [`binary_search`]: slice::binary_search
    /// [`binary_search_by_key`]: slice::binary_search_by_key
    /// [`partition_point`]: slice::partition_point
    ///
    /// # Examples
    ///
    /// Looks up a series of four elements. The first is found, with a
    /// uniquely determined position; the second and third are not
    /// found; the fourth could match any position in `[1, 4]`.
    ///
    /// ```
    /// let s = [0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
    ///
    /// let seek = 13;
    /// assert_eq!(s.binary_search_by(|probe| probe.cmp(&seek)), Ok(9));
    /// let seek = 4;
    /// assert_eq!(s.binary_search_by(|probe| probe.cmp(&seek)), Err(7));
    /// let seek = 100;
    /// assert_eq!(s.binary_search_by(|probe| probe.cmp(&seek)), Err(13));
    /// let seek = 1;
    /// let r = s.binary_search_by(|probe| probe.cmp(&seek));
    /// assert!(match r { Ok(1..=4) => true, _ => false, });
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn binary_search_by<'a, F>(&'a self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&'a T) -> Ordering,
    {
        let mut size = self.len();
        if size == 0 {
            return Err(0);
        }
        let mut base = 0usize;

        // This loop intentionally doesn't have an early exit if the comparison
        // returns Equal. We want the number of loop iterations to depend *only*
        // on the size of the input slice so that the CPU can reliably predict
        // the loop count.
        while size > 1 {
            let half = size / 2;
            let mid = base + half;

            // SAFETY: the call is made safe by the following inconstants:
            // - `mid >= 0`: by definition
            // - `mid < size`: `mid = size / 2 + size / 4 + size / 8 ...`
            let cmp = f(unsafe { self.get_unchecked(mid) });

            // Binary search interacts poorly with branch prediction, so force
            // the compiler to use conditional moves if supported by the target
            // architecture.
            base = select_unpredictable(cmp == Greater, base, mid);

            // This is imprecise in the case where `size` is odd and the
            // comparison returns Greater: the mid element still gets included
            // by `size` even though it's known to be larger than the element
            // being searched for.
            //
            // This is fine though: we gain more performance by keeping the
            // loop iteration count invariant (and thus predictable) than we
            // lose from considering one additional element.
            size -= half;
        }

        // SAFETY: base is always in [0, size) because base <= mid.
        let cmp = f(unsafe { self.get_unchecked(base) });
        if cmp == Equal {
            // SAFETY: same as the `get_unchecked` above.
            unsafe { hint::assert_unchecked(base < self.len()) };
            Ok(base)
        } else {
            let result = base + (cmp == Less) as usize;
            // SAFETY: same as the `get_unchecked` above.
            // Note that this is `<=`, unlike the assume in the `Ok` path.
            unsafe { hint::assert_unchecked(result <= self.len()) };
            Err(result)
        }
    }

    /// Binary searches this slice with a key extraction function.
    ///
    /// Assumes that the slice is sorted by the key, for instance with
    /// [`sort_by_key`] using the same key extraction function.
    /// If the slice is not sorted by the key, the returned result is
    /// unspecified and meaningless.
    ///
    /// If the value is found then [`Result::Ok`] is returned, containing the
    /// index of the matching element. If there are multiple matches, then any
    /// one of the matches could be returned. The index is chosen
    /// deterministically, but is subject to change in future versions of Rust.
    /// If the value is not found then [`Result::Err`] is returned, containing
    /// the index where a matching element could be inserted while maintaining
    /// sorted order.
    ///
    /// See also [`binary_search`], [`binary_search_by`], and [`partition_point`].
    ///
    /// [`sort_by_key`]: slice::sort_by_key
    /// [`binary_search`]: slice::binary_search
    /// [`binary_search_by`]: slice::binary_search_by
    /// [`partition_point`]: slice::partition_point
    ///
    /// # Examples
    ///
    /// Looks up a series of four elements in a slice of pairs sorted by
    /// their second elements. The first is found, with a uniquely
    /// determined position; the second and third are not found; the
    /// fourth could match any position in `[1, 4]`.
    ///
    /// ```
    /// let s = [(0, 0), (2, 1), (4, 1), (5, 1), (3, 1),
    ///          (1, 2), (2, 3), (4, 5), (5, 8), (3, 13),
    ///          (1, 21), (2, 34), (4, 55)];
    ///
    /// assert_eq!(s.binary_search_by_key(&13, |&(a, b)| b),  Ok(9));
    /// assert_eq!(s.binary_search_by_key(&4, |&(a, b)| b),   Err(7));
    /// assert_eq!(s.binary_search_by_key(&100, |&(a, b)| b), Err(13));
    /// let r = s.binary_search_by_key(&1, |&(a, b)| b);
    /// assert!(match r { Ok(1..=4) => true, _ => false, });
    /// ```
    // Lint rustdoc::broken_intra_doc_links is allowed as `slice::sort_by_key` is
    // in crate `alloc`, and as such doesn't exists yet when building `core`: #74481.
    // This breaks links when slice is displayed in core, but changing it to use relative links
    // would break when the item is re-exported. So allow the core links to be broken for now.
    #[allow(rustdoc::broken_intra_doc_links)]
    #[stable(feature = "slice_binary_search_by_key", since = "1.10.0")]
    #[inline]
    pub fn binary_search_by_key<'a, B, F>(&'a self, b: &B, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&'a T) -> B,
        B: Ord,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }

    /// Sorts the slice **without** preserving the initial order of equal elements.
    ///
    /// This sort is unstable (i.e., may reorder equal elements), in-place (i.e., does not
    /// allocate), and *O*(*n* \* log(*n*)) worst-case.
    ///
    /// If the implementation of [`Ord`] for `T` does not implement a [total order] the resulting
    /// order of elements in the slice is unspecified. All original elements will remain in the
    /// slice and any possible modifications via interior mutability are observed in the input. Same
    /// is true if the implementation of [`Ord`] for `T` panics.
    ///
    /// Sorting types that only implement [`PartialOrd`] such as [`f32`] and [`f64`] require
    /// additional precautions. For example, `f32::NAN != f32::NAN`, which doesn't fulfill the
    /// reflexivity requirement of [`Ord`]. By using an alternative comparison function with
    /// `slice::sort_unstable_by` such as [`f32::total_cmp`] or [`f64::total_cmp`] that defines a
    /// [total order] users can sort slices containing floating-point values. Alternatively, if all
    /// values in the slice are guaranteed to be in a subset for which [`PartialOrd::partial_cmp`]
    /// forms a [total order], it's possible to sort the slice with `sort_unstable_by(|a, b|
    /// a.partial_cmp(b).unwrap())`.
    ///
    /// # Current implementation
    ///
    /// The current implementation is based on [ipnsort] by Lukas Bergdoll and Orson Peters, which
    /// combines the fast average case of quicksort with the fast worst case of heapsort, achieving
    /// linear time on fully sorted and reversed inputs. On inputs with k distinct elements, the
    /// expected time to sort the data is *O*(*n* \* log(*k*)).
    ///
    /// It is typically faster than stable sorting, except in a few special cases, e.g., when the
    /// slice is partially sorted.
    ///
    /// # Panics
    ///
    /// May panic if the implementation of [`Ord`] for `T` does not implement a [total order].
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [4, -5, 1, -3, 2];
    ///
    /// v.sort_unstable();
    /// assert_eq!(v, [-5, -3, 1, 2, 4]);
    /// ```
    ///
    /// [ipnsort]: https://github.com/Voultapher/sort-research-rs/tree/main/ipnsort
    /// [total order]: https://en.wikipedia.org/wiki/Total_order
    #[stable(feature = "sort_unstable", since = "1.20.0")]
    #[inline]
    pub fn sort_unstable(&mut self)
    where
        T: Ord,
    {
        sort::unstable::sort(self, &mut T::lt);
    }

    /// Sorts the slice with a comparison function, **without** preserving the initial order of
    /// equal elements.
    ///
    /// This sort is unstable (i.e., may reorder equal elements), in-place (i.e., does not
    /// allocate), and *O*(*n* \* log(*n*)) worst-case.
    ///
    /// If the comparison function `compare` does not implement a [total order] the resulting order
    /// of elements in the slice is unspecified. All original elements will remain in the slice and
    /// any possible modifications via interior mutability are observed in the input. Same is true
    /// if `compare` panics.
    ///
    /// For example `|a, b| (a - b).cmp(a)` is a comparison function that is neither transitive nor
    /// reflexive nor total, `a < b < c < a` with `a = 1, b = 2, c = 3`. For more information and
    /// examples see the [`Ord`] documentation.
    ///
    /// # Current implementation
    ///
    /// The current implementation is based on [ipnsort] by Lukas Bergdoll and Orson Peters, which
    /// combines the fast average case of quicksort with the fast worst case of heapsort, achieving
    /// linear time on fully sorted and reversed inputs. On inputs with k distinct elements, the
    /// expected time to sort the data is *O*(*n* \* log(*k*)).
    ///
    /// It is typically faster than stable sorting, except in a few special cases, e.g., when the
    /// slice is partially sorted.
    ///
    /// # Panics
    ///
    /// May panic if `compare` does not implement a [total order].
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [4, -5, 1, -3, 2];
    /// v.sort_unstable_by(|a, b| a.cmp(b));
    /// assert_eq!(v, [-5, -3, 1, 2, 4]);
    ///
    /// // reverse sorting
    /// v.sort_unstable_by(|a, b| b.cmp(a));
    /// assert_eq!(v, [4, 2, 1, -3, -5]);
    /// ```
    ///
    /// [ipnsort]: https://github.com/Voultapher/sort-research-rs/tree/main/ipnsort
    /// [total order]: https://en.wikipedia.org/wiki/Total_order
    #[stable(feature = "sort_unstable", since = "1.20.0")]
    #[inline]
    pub fn sort_unstable_by<F>(&mut self, mut compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        sort::unstable::sort(self, &mut |a, b| compare(a, b) == Ordering::Less);
    }

    /// Sorts the slice with a key extraction function, **without** preserving the initial order of
    /// equal elements.
    ///
    /// This sort is unstable (i.e., may reorder equal elements), in-place (i.e., does not
    /// allocate), and *O*(*n* \* log(*n*)) worst-case.
    ///
    /// If the implementation of [`Ord`] for `K` does not implement a [total order] the resulting
    /// order of elements in the slice is unspecified. All original elements will remain in the
    /// slice and any possible modifications via interior mutability are observed in the input. Same
    /// is true if the implementation of [`Ord`] for `K` panics.
    ///
    /// # Current implementation
    ///
    /// The current implementation is based on [ipnsort] by Lukas Bergdoll and Orson Peters, which
    /// combines the fast average case of quicksort with the fast worst case of heapsort, achieving
    /// linear time on fully sorted and reversed inputs. On inputs with k distinct elements, the
    /// expected time to sort the data is *O*(*n* \* log(*k*)).
    ///
    /// It is typically faster than stable sorting, except in a few special cases, e.g., when the
    /// slice is partially sorted.
    ///
    /// # Panics
    ///
    /// May panic if the implementation of [`Ord`] for `K` does not implement a [total order].
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [4i32, -5, 1, -3, 2];
    ///
    /// v.sort_unstable_by_key(|k| k.abs());
    /// assert_eq!(v, [1, 2, -3, 4, -5]);
    /// ```
    ///
    /// [ipnsort]: https://github.com/Voultapher/sort-research-rs/tree/main/ipnsort
    /// [total order]: https://en.wikipedia.org/wiki/Total_order
    #[stable(feature = "sort_unstable", since = "1.20.0")]
    #[inline]
    pub fn sort_unstable_by_key<K, F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        sort::unstable::sort(self, &mut |a, b| f(a).lt(&f(b)));
    }

    /// Reorders the slice such that the element at `index` is at a sort-order position. All
    /// elements before `index` will be `<=` this value, and all elements after will be `>=`.
    ///
    /// This reordering is unstable (i.e. any element that compares equal to the nth element may end
    /// up at that position), in-place (i.e.  does not allocate), and runs in *O*(*n*) time. This
    /// function is also known as "kth element" in other libraries.
    ///
    /// Returns a triple partitioning the reordered slice:
    ///
    /// * The unsorted subslice before `index` (elements all pass `x <= self[index]`)
    /// * The element at `index`
    /// * The unsorted subslice after `index` (elements all pass `x >= self[index]`)
    ///
    /// # Current implementation
    ///
    /// The current algorithm is an introselect implementation based on [ipnsort] by Lukas Bergdoll
    /// and Orson Peters, which is also the basis for [`sort_unstable`]. The fallback algorithm is
    /// Median of Medians using Tukey's Ninther for pivot selection, which guarantees linear runtime
    /// for all inputs.
    ///
    /// [`sort_unstable`]: slice::sort_unstable
    ///
    /// # Panics
    ///
    /// Panics when `index >= len()`, and so always panics on empty slices.
    ///
    /// May panic if the implementation of [`Ord`] for `T` does not implement a [total order].
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [-5i32, 4, 2, -3, 1];
    ///
    /// // Find the items `<=` the median, the median, and `>=` the median.
    /// let (lesser, median, greater) = v.select_nth_unstable(2);
    ///
    /// assert!(lesser == [-3, -5] || lesser == [-5, -3]);
    /// assert_eq!(median, &mut 1);
    /// assert!(greater == [4, 2] || greater == [2, 4]);
    ///
    /// // We are only guaranteed the slice will be one of the following, based on the way we sort
    /// // about the specified index.
    /// assert!(v == [-3, -5, 1, 2, 4] ||
    ///         v == [-5, -3, 1, 2, 4] ||
    ///         v == [-3, -5, 1, 4, 2] ||
    ///         v == [-5, -3, 1, 4, 2]);
    /// ```
    ///
    /// [ipnsort]: https://github.com/Voultapher/sort-research-rs/tree/main/ipnsort
    /// [total order]: https://en.wikipedia.org/wiki/Total_order
    #[stable(feature = "slice_select_nth_unstable", since = "1.49.0")]
    #[inline]
    pub fn select_nth_unstable(&mut self, index: usize) -> (&mut [T], &mut T, &mut [T])
    where
        T: Ord,
    {
        sort::select::partition_at_index(self, index, T::lt)
    }

    /// Reorders the slice with a comparator function such that the element at `index` is at a
    /// sort-order position. All elements before `index` will be `<=` this value, and all elements
    /// after will be `>=` according to the comparator function.
    ///
    /// This reordering is unstable (i.e. any element that compares equal to the nth element may end
    /// up at that position), in-place (i.e.  does not allocate), and runs in *O*(*n*) time. This
    /// function is also known as "kth element" in other libraries.
    ///
    /// Returns a triple partitioning the reordered slice:
    ///
    /// * The unsorted subslice before `index` (elements all pass `compare(x, self[index]).is_le()`)
    /// * The element at `index`
    /// * The unsorted subslice after `index` (elements all pass `compare(x, self[index]).is_ge()`)
    ///
    /// # Current implementation
    ///
    /// The current algorithm is an introselect implementation based on [ipnsort] by Lukas Bergdoll
    /// and Orson Peters, which is also the basis for [`sort_unstable`]. The fallback algorithm is
    /// Median of Medians using Tukey's Ninther for pivot selection, which guarantees linear runtime
    /// for all inputs.
    ///
    /// [`sort_unstable`]: slice::sort_unstable
    ///
    /// # Panics
    ///
    /// Panics when `index >= len()`, and so always panics on empty slices.
    ///
    /// May panic if `compare` does not implement a [total order].
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [-5i32, 4, 2, -3, 1];
    ///
    /// // Find the items `>=` the median, the median, and `<=` the median, by using a reversed
    /// // comparator.
    /// let (before, median, after) = v.select_nth_unstable_by(2, |a, b| b.cmp(a));
    ///
    /// assert!(before == [4, 2] || before == [2, 4]);
    /// assert_eq!(median, &mut 1);
    /// assert!(after == [-3, -5] || after == [-5, -3]);
    ///
    /// // We are only guaranteed the slice will be one of the following, based on the way we sort
    /// // about the specified index.
    /// assert!(v == [2, 4, 1, -5, -3] ||
    ///         v == [2, 4, 1, -3, -5] ||
    ///         v == [4, 2, 1, -5, -3] ||
    ///         v == [4, 2, 1, -3, -5]);
    /// ```
    ///
    /// [ipnsort]: https://github.com/Voultapher/sort-research-rs/tree/main/ipnsort
    /// [total order]: https://en.wikipedia.org/wiki/Total_order
    #[stable(feature = "slice_select_nth_unstable", since = "1.49.0")]
    #[inline]
    pub fn select_nth_unstable_by<F>(
        &mut self,
        index: usize,
        mut compare: F,
    ) -> (&mut [T], &mut T, &mut [T])
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        sort::select::partition_at_index(self, index, |a: &T, b: &T| compare(a, b) == Less)
    }

    /// Reorders the slice with a key extraction function such that the element at `index` is at a
    /// sort-order position. All elements before `index` will have keys `<=` the key at `index`, and
    /// all elements after will have keys `>=`.
    ///
    /// This reordering is unstable (i.e. any element that compares equal to the nth element may end
    /// up at that position), in-place (i.e.  does not allocate), and runs in *O*(*n*) time. This
    /// function is also known as "kth element" in other libraries.
    ///
    /// Returns a triple partitioning the reordered slice:
    ///
    /// * The unsorted subslice before `index` (elements all pass `f(x) <= f(self[index])`)
    /// * The element at `index`
    /// * The unsorted subslice after `index` (elements all pass `f(x) >= f(self[index])`)
    ///
    /// # Current implementation
    ///
    /// The current algorithm is an introselect implementation based on [ipnsort] by Lukas Bergdoll
    /// and Orson Peters, which is also the basis for [`sort_unstable`]. The fallback algorithm is
    /// Median of Medians using Tukey's Ninther for pivot selection, which guarantees linear runtime
    /// for all inputs.
    ///
    /// [`sort_unstable`]: slice::sort_unstable
    ///
    /// # Panics
    ///
    /// Panics when `index >= len()`, meaning it always panics on empty slices.
    ///
    /// May panic if `K: Ord` does not implement a total order.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [-5i32, 4, 1, -3, 2];
    ///
    /// // Find the items <= the median absolute value, the median absolute value, and >= the median
    /// // absolute value.
    /// let (lesser, median, greater) = v.select_nth_unstable_by_key(2, |a| a.abs());
    ///
    /// assert!(lesser == [1, 2] || lesser == [2, 1]);
    /// assert_eq!(median, &mut -3);
    /// assert!(greater == [4, -5] || greater == [-5, 4]);
    ///
    /// // We are only guaranteed the slice will be one of the following, based on the way we sort
    /// // about the specified index.
    /// assert!(v == [1, 2, -3, 4, -5] ||
    ///         v == [1, 2, -3, -5, 4] ||
    ///         v == [2, 1, -3, 4, -5] ||
    ///         v == [2, 1, -3, -5, 4]);
    /// ```
    ///
    /// [ipnsort]: https://github.com/Voultapher/sort-research-rs/tree/main/ipnsort
    /// [total order]: https://en.wikipedia.org/wiki/Total_order
    #[stable(feature = "slice_select_nth_unstable", since = "1.49.0")]
    #[inline]
    pub fn select_nth_unstable_by_key<K, F>(
        &mut self,
        index: usize,
        mut f: F,
    ) -> (&mut [T], &mut T, &mut [T])
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        sort::select::partition_at_index(self, index, |a: &T, b: &T| f(a).lt(&f(b)))
    }

    /// Moves all consecutive repeated elements to the end of the slice according to the
    /// [`PartialEq`] trait implementation.
    ///
    /// Returns two slices. The first contains no consecutive repeated elements.
    /// The second contains all the duplicates in no specified order.
    ///
    /// If the slice is sorted, the first returned slice contains no duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_partition_dedup)]
    ///
    /// let mut slice = [1, 2, 2, 3, 3, 2, 1, 1];
    ///
    /// let (dedup, duplicates) = slice.partition_dedup();
    ///
    /// assert_eq!(dedup, [1, 2, 3, 2, 1]);
    /// assert_eq!(duplicates, [2, 3, 1]);
    /// ```
    #[unstable(feature = "slice_partition_dedup", issue = "54279")]
    #[inline]
    pub fn partition_dedup(&mut self) -> (&mut [T], &mut [T])
    where
        T: PartialEq,
    {
        self.partition_dedup_by(|a, b| a == b)
    }

    /// Moves all but the first of consecutive elements to the end of the slice satisfying
    /// a given equality relation.
    ///
    /// Returns two slices. The first contains no consecutive repeated elements.
    /// The second contains all the duplicates in no specified order.
    ///
    /// The `same_bucket` function is passed references to two elements from the slice and
    /// must determine if the elements compare equal. The elements are passed in opposite order
    /// from their order in the slice, so if `same_bucket(a, b)` returns `true`, `a` is moved
    /// at the end of the slice.
    ///
    /// If the slice is sorted, the first returned slice contains no duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_partition_dedup)]
    ///
    /// let mut slice = ["foo", "Foo", "BAZ", "Bar", "bar", "baz", "BAZ"];
    ///
    /// let (dedup, duplicates) = slice.partition_dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    ///
    /// assert_eq!(dedup, ["foo", "BAZ", "Bar", "baz"]);
    /// assert_eq!(duplicates, ["bar", "Foo", "BAZ"]);
    /// ```
    #[unstable(feature = "slice_partition_dedup", issue = "54279")]
    #[inline]
    pub fn partition_dedup_by<F>(&mut self, mut same_bucket: F) -> (&mut [T], &mut [T])
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        // Although we have a mutable reference to `self`, we cannot make
        // *arbitrary* changes. The `same_bucket` calls could panic, so we
        // must ensure that the slice is in a valid state at all times.
        //
        // The way that we handle this is by using swaps; we iterate
        // over all the elements, swapping as we go so that at the end
        // the elements we wish to keep are in the front, and those we
        // wish to reject are at the back. We can then split the slice.
        // This operation is still `O(n)`.
        //
        // Example: We start in this state, where `r` represents "next
        // read" and `w` represents "next_write".
        //
        //           r
        //     +---+---+---+---+---+---+
        //     | 0 | 1 | 1 | 2 | 3 | 3 |
        //     +---+---+---+---+---+---+
        //           w
        //
        // Comparing self[r] against self[w-1], this is not a duplicate, so
        // we swap self[r] and self[w] (no effect as r==w) and then increment both
        // r and w, leaving us with:
        //
        //               r
        //     +---+---+---+---+---+---+
        //     | 0 | 1 | 1 | 2 | 3 | 3 |
        //     +---+---+---+---+---+---+
        //               w
        //
        // Comparing self[r] against self[w-1], this value is a duplicate,
        // so we increment `r` but leave everything else unchanged:
        //
        //                   r
        //     +---+---+---+---+---+---+
        //     | 0 | 1 | 1 | 2 | 3 | 3 |
        //     +---+---+---+---+---+---+
        //               w
        //
        // Comparing self[r] against self[w-1], this is not a duplicate,
        // so swap self[r] and self[w] and advance r and w:
        //
        //                       r
        //     +---+---+---+---+---+---+
        //     | 0 | 1 | 2 | 1 | 3 | 3 |
        //     +---+---+---+---+---+---+
        //                   w
        //
        // Not a duplicate, repeat:
        //
        //                           r
        //     +---+---+---+---+---+---+
        //     | 0 | 1 | 2 | 3 | 1 | 3 |
        //     +---+---+---+---+---+---+
        //                       w
        //
        // Duplicate, advance r. End of slice. Split at w.

        let len = self.len();
        if len <= 1 {
            return (self, &mut []);
        }

        let ptr = self.as_mut_ptr();
        let mut next_read: usize = 1;
        let mut next_write: usize = 1;

        // SAFETY: the `while` condition guarantees `next_read` and `next_write`
        // are less than `len`, thus are inside `self`. `prev_ptr_write` points to
        // one element before `ptr_write`, but `next_write` starts at 1, so
        // `prev_ptr_write` is never less than 0 and is inside the slice.
        // This fulfils the requirements for dereferencing `ptr_read`, `prev_ptr_write`
        // and `ptr_write`, and for using `ptr.add(next_read)`, `ptr.add(next_write - 1)`
        // and `prev_ptr_write.offset(1)`.
        //
        // `next_write` is also incremented at most once per loop at most meaning
        // no element is skipped when it may need to be swapped.
        //
        // `ptr_read` and `prev_ptr_write` never point to the same element. This
        // is required for `&mut *ptr_read`, `&mut *prev_ptr_write` to be safe.
        // The explanation is simply that `next_read >= next_write` is always true,
        // thus `next_read > next_write - 1` is too.
        unsafe {
            // Avoid bounds checks by using raw pointers.
            while next_read < len {
                let ptr_read = ptr.add(next_read);
                let prev_ptr_write = ptr.add(next_write - 1);
                if !same_bucket(&mut *ptr_read, &mut *prev_ptr_write) {
                    if next_read != next_write {
                        let ptr_write = prev_ptr_write.add(1);
                        mem::swap(&mut *ptr_read, &mut *ptr_write);
                    }
                    next_write += 1;
                }
                next_read += 1;
            }
        }

        self.split_at_mut(next_write)
    }

    /// Moves all but the first of consecutive elements to the end of the slice that resolve
    /// to the same key.
    ///
    /// Returns two slices. The first contains no consecutive repeated elements.
    /// The second contains all the duplicates in no specified order.
    ///
    /// If the slice is sorted, the first returned slice contains no duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_partition_dedup)]
    ///
    /// let mut slice = [10, 20, 21, 30, 30, 20, 11, 13];
    ///
    /// let (dedup, duplicates) = slice.partition_dedup_by_key(|i| *i / 10);
    ///
    /// assert_eq!(dedup, [10, 20, 30, 20, 11]);
    /// assert_eq!(duplicates, [21, 30, 13]);
    /// ```
    #[unstable(feature = "slice_partition_dedup", issue = "54279")]
    #[inline]
    pub fn partition_dedup_by_key<K, F>(&mut self, mut key: F) -> (&mut [T], &mut [T])
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq,
    {
        self.partition_dedup_by(|a, b| key(a) == key(b))
    }

    /// Rotates the slice in-place such that the first `mid` elements of the
    /// slice move to the end while the last `self.len() - mid` elements move to
    /// the front.
    ///
    /// After calling `rotate_left`, the element previously at index `mid` will
    /// become the first element in the slice.
    ///
    /// # Panics
    ///
    /// This function will panic if `mid` is greater than the length of the
    /// slice. Note that `mid == self.len()` does _not_ panic and is a no-op
    /// rotation.
    ///
    /// # Complexity
    ///
    /// Takes linear (in `self.len()`) time.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut a = ['a', 'b', 'c', 'd', 'e', 'f'];
    /// a.rotate_left(2);
    /// assert_eq!(a, ['c', 'd', 'e', 'f', 'a', 'b']);
    /// ```
    ///
    /// Rotating a subslice:
    ///
    /// ```
    /// let mut a = ['a', 'b', 'c', 'd', 'e', 'f'];
    /// a[1..5].rotate_left(1);
    /// assert_eq!(a, ['a', 'c', 'd', 'e', 'b', 'f']);
    /// ```
    #[stable(feature = "slice_rotate", since = "1.26.0")]
    pub fn rotate_left(&mut self, mid: usize) {
        assert!(mid <= self.len());
        let k = self.len() - mid;
        let p = self.as_mut_ptr();

        // SAFETY: The range `[p.add(mid) - mid, p.add(mid) + k)` is trivially
        // valid for reading and writing, as required by `ptr_rotate`.
        unsafe {
            rotate::ptr_rotate(mid, p.add(mid), k);
        }
    }

    /// Rotates the slice in-place such that the first `self.len() - k`
    /// elements of the slice move to the end while the last `k` elements move
    /// to the front.
    ///
    /// After calling `rotate_right`, the element previously at index
    /// `self.len() - k` will become the first element in the slice.
    ///
    /// # Panics
    ///
    /// This function will panic if `k` is greater than the length of the
    /// slice. Note that `k == self.len()` does _not_ panic and is a no-op
    /// rotation.
    ///
    /// # Complexity
    ///
    /// Takes linear (in `self.len()`) time.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut a = ['a', 'b', 'c', 'd', 'e', 'f'];
    /// a.rotate_right(2);
    /// assert_eq!(a, ['e', 'f', 'a', 'b', 'c', 'd']);
    /// ```
    ///
    /// Rotating a subslice:
    ///
    /// ```
    /// let mut a = ['a', 'b', 'c', 'd', 'e', 'f'];
    /// a[1..5].rotate_right(1);
    /// assert_eq!(a, ['a', 'e', 'b', 'c', 'd', 'f']);
    /// ```
    #[stable(feature = "slice_rotate", since = "1.26.0")]
    pub fn rotate_right(&mut self, k: usize) {
        assert!(k <= self.len());
        let mid = self.len() - k;
        let p = self.as_mut_ptr();

        // SAFETY: The range `[p.add(mid) - mid, p.add(mid) + k)` is trivially
        // valid for reading and writing, as required by `ptr_rotate`.
        unsafe {
            rotate::ptr_rotate(mid, p.add(mid), k);
        }
    }

    /// Fills `self` with elements by cloning `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut buf = vec![0; 10];
    /// buf.fill(1);
    /// assert_eq!(buf, vec![1; 10]);
    /// ```
    #[doc(alias = "memset")]
    #[stable(feature = "slice_fill", since = "1.50.0")]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        specialize::SpecFill::spec_fill(self, value);
    }

    /// Fills `self` with elements returned by calling a closure repeatedly.
    ///
    /// This method uses a closure to create new values. If you'd rather
    /// [`Clone`] a given value, use [`fill`]. If you want to use the [`Default`]
    /// trait to generate values, you can pass [`Default::default`] as the
    /// argument.
    ///
    /// [`fill`]: slice::fill
    ///
    /// # Examples
    ///
    /// ```
    /// let mut buf = vec![1; 10];
    /// buf.fill_with(Default::default);
    /// assert_eq!(buf, vec![0; 10]);
    /// ```
    #[stable(feature = "slice_fill_with", since = "1.51.0")]
    pub fn fill_with<F>(&mut self, mut f: F)
    where
        F: FnMut() -> T,
    {
        for el in self {
            *el = f();
        }
    }

    /// Copies the elements from `src` into `self`.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths.
    ///
    /// # Examples
    ///
    /// Cloning two elements from a slice into another:
    ///
    /// ```
    /// let src = [1, 2, 3, 4];
    /// let mut dst = [0, 0];
    ///
    /// // Because the slices have to be the same length,
    /// // we slice the source slice from four elements
    /// // to two. It will panic if we don't do this.
    /// dst.clone_from_slice(&src[2..]);
    ///
    /// assert_eq!(src, [1, 2, 3, 4]);
    /// assert_eq!(dst, [3, 4]);
    /// ```
    ///
    /// Rust enforces that there can only be one mutable reference with no
    /// immutable references to a particular piece of data in a particular
    /// scope. Because of this, attempting to use `clone_from_slice` on a
    /// single slice will result in a compile failure:
    ///
    /// ```compile_fail
    /// let mut slice = [1, 2, 3, 4, 5];
    ///
    /// slice[..2].clone_from_slice(&slice[3..]); // compile fail!
    /// ```
    ///
    /// To work around this, we can use [`split_at_mut`] to create two distinct
    /// sub-slices from a slice:
    ///
    /// ```
    /// let mut slice = [1, 2, 3, 4, 5];
    ///
    /// {
    ///     let (left, right) = slice.split_at_mut(2);
    ///     left.clone_from_slice(&right[1..]);
    /// }
    ///
    /// assert_eq!(slice, [4, 5, 3, 4, 5]);
    /// ```
    ///
    /// [`copy_from_slice`]: slice::copy_from_slice
    /// [`split_at_mut`]: slice::split_at_mut
    #[stable(feature = "clone_from_slice", since = "1.7.0")]
    #[track_caller]
    pub fn clone_from_slice(&mut self, src: &[T])
    where
        T: Clone,
    {
        self.spec_clone_from(src);
    }

    /// Copies all elements from `src` into `self`, using a memcpy.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// If `T` does not implement `Copy`, use [`clone_from_slice`].
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths.
    ///
    /// # Examples
    ///
    /// Copying two elements from a slice into another:
    ///
    /// ```
    /// let src = [1, 2, 3, 4];
    /// let mut dst = [0, 0];
    ///
    /// // Because the slices have to be the same length,
    /// // we slice the source slice from four elements
    /// // to two. It will panic if we don't do this.
    /// dst.copy_from_slice(&src[2..]);
    ///
    /// assert_eq!(src, [1, 2, 3, 4]);
    /// assert_eq!(dst, [3, 4]);
    /// ```
    ///
    /// Rust enforces that there can only be one mutable reference with no
    /// immutable references to a particular piece of data in a particular
    /// scope. Because of this, attempting to use `copy_from_slice` on a
    /// single slice will result in a compile failure:
    ///
    /// ```compile_fail
    /// let mut slice = [1, 2, 3, 4, 5];
    ///
    /// slice[..2].copy_from_slice(&slice[3..]); // compile fail!
    /// ```
    ///
    /// To work around this, we can use [`split_at_mut`] to create two distinct
    /// sub-slices from a slice:
    ///
    /// ```
    /// let mut slice = [1, 2, 3, 4, 5];
    ///
    /// {
    ///     let (left, right) = slice.split_at_mut(2);
    ///     left.copy_from_slice(&right[1..]);
    /// }
    ///
    /// assert_eq!(slice, [4, 5, 3, 4, 5]);
    /// ```
    ///
    /// [`clone_from_slice`]: slice::clone_from_slice
    /// [`split_at_mut`]: slice::split_at_mut
    #[doc(alias = "memcpy")]
    #[stable(feature = "copy_from_slice", since = "1.9.0")]
    #[rustc_const_unstable(feature = "const_copy_from_slice", issue = "131415")]
    #[track_caller]
    pub const fn copy_from_slice(&mut self, src: &[T])
    where
        T: Copy,
    {
        // The panic code path was put into a cold function to not bloat the
        // call site.
        #[cfg_attr(not(feature = "panic_immediate_abort"), inline(never), cold)]
        #[cfg_attr(feature = "panic_immediate_abort", inline)]
        #[track_caller]
        const fn len_mismatch_fail(dst_len: usize, src_len: usize) -> ! {
            const_panic!(
                "copy_from_slice: source slice length does not match destination slice length",
                "copy_from_slice: source slice length ({src_len}) does not match destination slice length ({dst_len})",
                src_len: usize,
                dst_len: usize,
            )
        }

        if self.len() != src.len() {
            len_mismatch_fail(self.len(), src.len());
        }

        // SAFETY: `self` is valid for `self.len()` elements by definition, and `src` was
        // checked to have the same length. The slices cannot overlap because
        // mutable references are exclusive.
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(), self.as_mut_ptr(), self.len());
        }
    }

    /// Copies elements from one part of the slice to another part of itself,
    /// using a memmove.
    ///
    /// `src` is the range within `self` to copy from. `dest` is the starting
    /// index of the range within `self` to copy to, which will have the same
    /// length as `src`. The two ranges may overlap. The ends of the two ranges
    /// must be less than or equal to `self.len()`.
    ///
    /// # Panics
    ///
    /// This function will panic if either range exceeds the end of the slice,
    /// or if the end of `src` is before the start.
    ///
    /// # Examples
    ///
    /// Copying four bytes within a slice:
    ///
    /// ```
    /// let mut bytes = *b"Hello, World!";
    ///
    /// bytes.copy_within(1..5, 8);
    ///
    /// assert_eq!(&bytes, b"Hello, Wello!");
    /// ```
    #[stable(feature = "copy_within", since = "1.37.0")]
    #[track_caller]
    pub fn copy_within<R: RangeBounds<usize>>(&mut self, src: R, dest: usize)
    where
        T: Copy,
    {
        let Range { start: src_start, end: src_end } = slice::range(src, ..self.len());
        let count = src_end - src_start;
        assert!(dest <= self.len() - count, "dest is out of bounds");
        // SAFETY: the conditions for `ptr::copy` have all been checked above,
        // as have those for `ptr::add`.
        unsafe {
            // Derive both `src_ptr` and `dest_ptr` from the same loan
            let ptr = self.as_mut_ptr();
            let src_ptr = ptr.add(src_start);
            let dest_ptr = ptr.add(dest);
            ptr::copy(src_ptr, dest_ptr, count);
        }
    }

    /// Swaps all elements in `self` with those in `other`.
    ///
    /// The length of `other` must be the same as `self`.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths.
    ///
    /// # Example
    ///
    /// Swapping two elements across slices:
    ///
    /// ```
    /// let mut slice1 = [0, 0];
    /// let mut slice2 = [1, 2, 3, 4];
    ///
    /// slice1.swap_with_slice(&mut slice2[2..]);
    ///
    /// assert_eq!(slice1, [3, 4]);
    /// assert_eq!(slice2, [1, 2, 0, 0]);
    /// ```
    ///
    /// Rust enforces that there can only be one mutable reference to a
    /// particular piece of data in a particular scope. Because of this,
    /// attempting to use `swap_with_slice` on a single slice will result in
    /// a compile failure:
    ///
    /// ```compile_fail
    /// let mut slice = [1, 2, 3, 4, 5];
    /// slice[..2].swap_with_slice(&mut slice[3..]); // compile fail!
    /// ```
    ///
    /// To work around this, we can use [`split_at_mut`] to create two distinct
    /// mutable sub-slices from a slice:
    ///
    /// ```
    /// let mut slice = [1, 2, 3, 4, 5];
    ///
    /// {
    ///     let (left, right) = slice.split_at_mut(2);
    ///     left.swap_with_slice(&mut right[1..]);
    /// }
    ///
    /// assert_eq!(slice, [4, 5, 3, 1, 2]);
    /// ```
    ///
    /// [`split_at_mut`]: slice::split_at_mut
    #[stable(feature = "swap_with_slice", since = "1.27.0")]
    #[track_caller]
    pub fn swap_with_slice(&mut self, other: &mut [T]) {
        assert!(self.len() == other.len(), "destination and source slices have different lengths");
        // SAFETY: `self` is valid for `self.len()` elements by definition, and `src` was
        // checked to have the same length. The slices cannot overlap because
        // mutable references are exclusive.
        unsafe {
            ptr::swap_nonoverlapping(self.as_mut_ptr(), other.as_mut_ptr(), self.len());
        }
    }

    /// Function to calculate lengths of the middle and trailing slice for `align_to{,_mut}`.
    fn align_to_offsets<U>(&self) -> (usize, usize) {
        // What we gonna do about `rest` is figure out what multiple of `U`s we can put in a
        // lowest number of `T`s. And how many `T`s we need for each such "multiple".
        //
        // Consider for example T=u8 U=u16. Then we can put 1 U in 2 Ts. Simple. Now, consider
        // for example a case where size_of::<T> = 16, size_of::<U> = 24. We can put 2 Us in
        // place of every 3 Ts in the `rest` slice. A bit more complicated.
        //
        // Formula to calculate this is:
        //
        // Us = lcm(size_of::<T>, size_of::<U>) / size_of::<U>
        // Ts = lcm(size_of::<T>, size_of::<U>) / size_of::<T>
        //
        // Expanded and simplified:
        //
        // Us = size_of::<T> / gcd(size_of::<T>, size_of::<U>)
        // Ts = size_of::<U> / gcd(size_of::<T>, size_of::<U>)
        //
        // Luckily since all this is constant-evaluated... performance here matters not!
        const fn gcd(a: usize, b: usize) -> usize {
            if b == 0 { a } else { gcd(b, a % b) }
        }

        // Explicitly wrap the function call in a const block so it gets
        // constant-evaluated even in debug mode.
        let gcd: usize = const { gcd(mem::size_of::<T>(), mem::size_of::<U>()) };
        let ts: usize = mem::size_of::<U>() / gcd;
        let us: usize = mem::size_of::<T>() / gcd;

        // Armed with this knowledge, we can find how many `U`s we can fit!
        let us_len = self.len() / ts * us;
        // And how many `T`s will be in the trailing slice!
        let ts_len = self.len() % ts;
        (us_len, ts_len)
    }

    /// Transmutes the slice to a slice of another type, ensuring alignment of the types is
    /// maintained.
    ///
    /// This method splits the slice into three distinct slices: prefix, correctly aligned middle
    /// slice of a new type, and the suffix slice. The middle part will be as big as possible under
    /// the given alignment constraint and element size.
    ///
    /// This method has no purpose when either input element `T` or output element `U` are
    /// zero-sized and will return the original slice without splitting anything.
    ///
    /// # Safety
    ///
    /// This method is essentially a `transmute` with respect to the elements in the returned
    /// middle slice, so all the usual caveats pertaining to `transmute::<T, U>` also apply here.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// unsafe {
    ///     let bytes: [u8; 7] = [1, 2, 3, 4, 5, 6, 7];
    ///     let (prefix, shorts, suffix) = bytes.align_to::<u16>();
    ///     // less_efficient_algorithm_for_bytes(prefix);
    ///     // more_efficient_algorithm_for_aligned_shorts(shorts);
    ///     // less_efficient_algorithm_for_bytes(suffix);
    /// }
    /// ```
    #[stable(feature = "slice_align_to", since = "1.30.0")]
    #[must_use]
    pub unsafe fn align_to<U>(&self) -> (&[T], &[U], &[T]) {
        // Note that most of this function will be constant-evaluated,
        if U::IS_ZST || T::IS_ZST {
            // handle ZSTs specially, which is – don't handle them at all.
            return (self, &[], &[]);
        }

        // First, find at what point do we split between the first and 2nd slice. Easy with
        // ptr.align_offset.
        let ptr = self.as_ptr();
        // SAFETY: See the `align_to_mut` method for the detailed safety comment.
        let offset = unsafe { crate::ptr::align_offset(ptr, mem::align_of::<U>()) };
        if offset > self.len() {
            (self, &[], &[])
        } else {
            let (left, rest) = self.split_at(offset);
            let (us_len, ts_len) = rest.align_to_offsets::<U>();
            // Inform Miri that we want to consider the "middle" pointer to be suitably aligned.
            #[cfg(miri)]
            crate::intrinsics::miri_promise_symbolic_alignment(
                rest.as_ptr().cast(),
                mem::align_of::<U>(),
            );
            // SAFETY: now `rest` is definitely aligned, so `from_raw_parts` below is okay,
            // since the caller guarantees that we can transmute `T` to `U` safely.
            unsafe {
                (
                    left,
                    from_raw_parts(rest.as_ptr() as *const U, us_len),
                    from_raw_parts(rest.as_ptr().add(rest.len() - ts_len), ts_len),
                )
            }
        }
    }

    /// Transmutes the mutable slice to a mutable slice of another type, ensuring alignment of the
    /// types is maintained.
    ///
    /// This method splits the slice into three distinct slices: prefix, correctly aligned middle
    /// slice of a new type, and the suffix slice. The middle part will be as big as possible under
    /// the given alignment constraint and element size.
    ///
    /// This method has no purpose when either input element `T` or output element `U` are
    /// zero-sized and will return the original slice without splitting anything.
    ///
    /// # Safety
    ///
    /// This method is essentially a `transmute` with respect to the elements in the returned
    /// middle slice, so all the usual caveats pertaining to `transmute::<T, U>` also apply here.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// unsafe {
    ///     let mut bytes: [u8; 7] = [1, 2, 3, 4, 5, 6, 7];
    ///     let (prefix, shorts, suffix) = bytes.align_to_mut::<u16>();
    ///     // less_efficient_algorithm_for_bytes(prefix);
    ///     // more_efficient_algorithm_for_aligned_shorts(shorts);
    ///     // less_efficient_algorithm_for_bytes(suffix);
    /// }
    /// ```
    #[stable(feature = "slice_align_to", since = "1.30.0")]
    #[must_use]
    pub unsafe fn align_to_mut<U>(&mut self) -> (&mut [T], &mut [U], &mut [T]) {
        // Note that most of this function will be constant-evaluated,
        if U::IS_ZST || T::IS_ZST {
            // handle ZSTs specially, which is – don't handle them at all.
            return (self, &mut [], &mut []);
        }

        // First, find at what point do we split between the first and 2nd slice. Easy with
        // ptr.align_offset.
        let ptr = self.as_ptr();
        // SAFETY: Here we are ensuring we will use aligned pointers for U for the
        // rest of the method. This is done by passing a pointer to &[T] with an
        // alignment targeted for U.
        // `crate::ptr::align_offset` is called with a correctly aligned and
        // valid pointer `ptr` (it comes from a reference to `self`) and with
        // a size that is a power of two (since it comes from the alignment for U),
        // satisfying its safety constraints.
        let offset = unsafe { crate::ptr::align_offset(ptr, mem::align_of::<U>()) };
        if offset > self.len() {
            (self, &mut [], &mut [])
        } else {
            let (left, rest) = self.split_at_mut(offset);
            let (us_len, ts_len) = rest.align_to_offsets::<U>();
            let rest_len = rest.len();
            let mut_ptr = rest.as_mut_ptr();
            // Inform Miri that we want to consider the "middle" pointer to be suitably aligned.
            #[cfg(miri)]
            crate::intrinsics::miri_promise_symbolic_alignment(
                mut_ptr.cast() as *const (),
                mem::align_of::<U>(),
            );
            // We can't use `rest` again after this, that would invalidate its alias `mut_ptr`!
            // SAFETY: see comments for `align_to`.
            unsafe {
                (
                    left,
                    from_raw_parts_mut(mut_ptr as *mut U, us_len),
                    from_raw_parts_mut(mut_ptr.add(rest_len - ts_len), ts_len),
                )
            }
        }
    }

    /// Splits a slice into a prefix, a middle of aligned SIMD types, and a suffix.
    ///
    /// This is a safe wrapper around [`slice::align_to`], so inherits the same
    /// guarantees as that method.
    ///
    /// # Panics
    ///
    /// This will panic if the size of the SIMD type is different from
    /// `LANES` times that of the scalar.
    ///
    /// At the time of writing, the trait restrictions on `Simd<T, LANES>` keeps
    /// that from ever happening, as only power-of-two numbers of lanes are
    /// supported.  It's possible that, in the future, those restrictions might
    /// be lifted in a way that would make it possible to see panics from this
    /// method for something like `LANES == 3`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use core::simd::prelude::*;
    ///
    /// let short = &[1, 2, 3];
    /// let (prefix, middle, suffix) = short.as_simd::<4>();
    /// assert_eq!(middle, []); // Not enough elements for anything in the middle
    ///
    /// // They might be split in any possible way between prefix and suffix
    /// let it = prefix.iter().chain(suffix).copied();
    /// assert_eq!(it.collect::<Vec<_>>(), vec![1, 2, 3]);
    ///
    /// fn basic_simd_sum(x: &[f32]) -> f32 {
    ///     use std::ops::Add;
    ///     let (prefix, middle, suffix) = x.as_simd();
    ///     let sums = f32x4::from_array([
    ///         prefix.iter().copied().sum(),
    ///         0.0,
    ///         0.0,
    ///         suffix.iter().copied().sum(),
    ///     ]);
    ///     let sums = middle.iter().copied().fold(sums, f32x4::add);
    ///     sums.reduce_sum()
    /// }
    ///
    /// let numbers: Vec<f32> = (1..101).map(|x| x as _).collect();
    /// assert_eq!(basic_simd_sum(&numbers[1..99]), 4949.0);
    /// ```
    #[unstable(feature = "portable_simd", issue = "86656")]
    #[must_use]
    pub fn as_simd<const LANES: usize>(&self) -> (&[T], &[Simd<T, LANES>], &[T])
    where
        Simd<T, LANES>: AsRef<[T; LANES]>,
        T: simd::SimdElement,
        simd::LaneCount<LANES>: simd::SupportedLaneCount,
    {
        // These are expected to always match, as vector types are laid out like
        // arrays per <https://llvm.org/docs/LangRef.html#vector-type>, but we
        // might as well double-check since it'll optimize away anyhow.
        assert_eq!(mem::size_of::<Simd<T, LANES>>(), mem::size_of::<[T; LANES]>());

        // SAFETY: The simd types have the same layout as arrays, just with
        // potentially-higher alignment, so the de-facto transmutes are sound.
        unsafe { self.align_to() }
    }

    /// Splits a mutable slice into a mutable prefix, a middle of aligned SIMD types,
    /// and a mutable suffix.
    ///
    /// This is a safe wrapper around [`slice::align_to_mut`], so inherits the same
    /// guarantees as that method.
    ///
    /// This is the mutable version of [`slice::as_simd`]; see that for examples.
    ///
    /// # Panics
    ///
    /// This will panic if the size of the SIMD type is different from
    /// `LANES` times that of the scalar.
    ///
    /// At the time of writing, the trait restrictions on `Simd<T, LANES>` keeps
    /// that from ever happening, as only power-of-two numbers of lanes are
    /// supported.  It's possible that, in the future, those restrictions might
    /// be lifted in a way that would make it possible to see panics from this
    /// method for something like `LANES == 3`.
    #[unstable(feature = "portable_simd", issue = "86656")]
    #[must_use]
    pub fn as_simd_mut<const LANES: usize>(&mut self) -> (&mut [T], &mut [Simd<T, LANES>], &mut [T])
    where
        Simd<T, LANES>: AsMut<[T; LANES]>,
        T: simd::SimdElement,
        simd::LaneCount<LANES>: simd::SupportedLaneCount,
    {
        // These are expected to always match, as vector types are laid out like
        // arrays per <https://llvm.org/docs/LangRef.html#vector-type>, but we
        // might as well double-check since it'll optimize away anyhow.
        assert_eq!(mem::size_of::<Simd<T, LANES>>(), mem::size_of::<[T; LANES]>());

        // SAFETY: The simd types have the same layout as arrays, just with
        // potentially-higher alignment, so the de-facto transmutes are sound.
        unsafe { self.align_to_mut() }
    }

    /// Checks if the elements of this slice are sorted.
    ///
    /// That is, for each element `a` and its following element `b`, `a <= b` must hold. If the
    /// slice yields exactly zero or one element, `true` is returned.
    ///
    /// Note that if `Self::Item` is only `PartialOrd`, but not `Ord`, the above definition
    /// implies that this function returns `false` if any two consecutive items are not
    /// comparable.
    ///
    /// # Examples
    ///
    /// ```
    /// let empty: [i32; 0] = [];
    ///
    /// assert!([1, 2, 2, 9].is_sorted());
    /// assert!(![1, 3, 2, 4].is_sorted());
    /// assert!([0].is_sorted());
    /// assert!(empty.is_sorted());
    /// assert!(![0.0, 1.0, f32::NAN].is_sorted());
    /// ```
    #[inline]
    #[stable(feature = "is_sorted", since = "1.82.0")]
    #[must_use]
    pub fn is_sorted(&self) -> bool
    where
        T: PartialOrd,
    {
        // This odd number works the best. 32 + 1 extra due to overlapping chunk boundaries.
        const CHUNK_SIZE: usize = 33;
        if self.len() < CHUNK_SIZE {
            return self.windows(2).all(|w| w[0] <= w[1]);
        }
        let mut i = 0;
        // Check in chunks for autovectorization.
        while i < self.len() - CHUNK_SIZE {
            let chunk = &self[i..i + CHUNK_SIZE];
            if !chunk.windows(2).fold(true, |acc, w| acc & (w[0] <= w[1])) {
                return false;
            }
            // We need to ensure that chunk boundaries are also sorted.
            // Overlap the next chunk with the last element of our last chunk.
            i += CHUNK_SIZE - 1;
        }
        self[i..].windows(2).all(|w| w[0] <= w[1])
    }

    /// Checks if the elements of this slice are sorted using the given comparator function.
    ///
    /// Instead of using `PartialOrd::partial_cmp`, this function uses the given `compare`
    /// function to determine whether two elements are to be considered in sorted order.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!([1, 2, 2, 9].is_sorted_by(|a, b| a <= b));
    /// assert!(![1, 2, 2, 9].is_sorted_by(|a, b| a < b));
    ///
    /// assert!([0].is_sorted_by(|a, b| true));
    /// assert!([0].is_sorted_by(|a, b| false));
    ///
    /// let empty: [i32; 0] = [];
    /// assert!(empty.is_sorted_by(|a, b| false));
    /// assert!(empty.is_sorted_by(|a, b| true));
    /// ```
    #[stable(feature = "is_sorted", since = "1.82.0")]
    #[must_use]
    pub fn is_sorted_by<'a, F>(&'a self, mut compare: F) -> bool
    where
        F: FnMut(&'a T, &'a T) -> bool,
    {
        self.array_windows().all(|[a, b]| compare(a, b))
    }

    /// Checks if the elements of this slice are sorted using the given key extraction function.
    ///
    /// Instead of comparing the slice's elements directly, this function compares the keys of the
    /// elements, as determined by `f`. Apart from that, it's equivalent to [`is_sorted`]; see its
    /// documentation for more information.
    ///
    /// [`is_sorted`]: slice::is_sorted
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(["c", "bb", "aaa"].is_sorted_by_key(|s| s.len()));
    /// assert!(![-2i32, -1, 0, 3].is_sorted_by_key(|n| n.abs()));
    /// ```
    #[inline]
    #[stable(feature = "is_sorted", since = "1.82.0")]
    #[must_use]
    pub fn is_sorted_by_key<'a, F, K>(&'a self, f: F) -> bool
    where
        F: FnMut(&'a T) -> K,
        K: PartialOrd,
    {
        self.iter().is_sorted_by_key(f)
    }

    /// Returns the index of the partition point according to the given predicate
    /// (the index of the first element of the second partition).
    ///
    /// The slice is assumed to be partitioned according to the given predicate.
    /// This means that all elements for which the predicate returns true are at the start of the slice
    /// and all elements for which the predicate returns false are at the end.
    /// For example, `[7, 15, 3, 5, 4, 12, 6]` is partitioned under the predicate `x % 2 != 0`
    /// (all odd numbers are at the start, all even at the end).
    ///
    /// If this slice is not partitioned, the returned result is unspecified and meaningless,
    /// as this method performs a kind of binary search.
    ///
    /// See also [`binary_search`], [`binary_search_by`], and [`binary_search_by_key`].
    ///
    /// [`binary_search`]: slice::binary_search
    /// [`binary_search_by`]: slice::binary_search_by
    /// [`binary_search_by_key`]: slice::binary_search_by_key
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [1, 2, 3, 3, 5, 6, 7];
    /// let i = v.partition_point(|&x| x < 5);
    ///
    /// assert_eq!(i, 4);
    /// assert!(v[..i].iter().all(|&x| x < 5));
    /// assert!(v[i..].iter().all(|&x| !(x < 5)));
    /// ```
    ///
    /// If all elements of the slice match the predicate, including if the slice
    /// is empty, then the length of the slice will be returned:
    ///
    /// ```
    /// let a = [2, 4, 8];
    /// assert_eq!(a.partition_point(|x| x < &100), a.len());
    /// let a: [i32; 0] = [];
    /// assert_eq!(a.partition_point(|x| x < &100), 0);
    /// ```
    ///
    /// If you want to insert an item to a sorted vector, while maintaining
    /// sort order:
    ///
    /// ```
    /// let mut s = vec![0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
    /// let num = 42;
    /// let idx = s.partition_point(|&x| x <= num);
    /// s.insert(idx, num);
    /// assert_eq!(s, [0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 42, 55]);
    /// ```
    #[stable(feature = "partition_point", since = "1.52.0")]
    #[must_use]
    pub fn partition_point<P>(&self, mut pred: P) -> usize
    where
        P: FnMut(&T) -> bool,
    {
        self.binary_search_by(|x| if pred(x) { Less } else { Greater }).unwrap_or_else(|i| i)
    }

    /// Removes the subslice corresponding to the given range
    /// and returns a reference to it.
    ///
    /// Returns `None` and does not modify the slice if the given
    /// range is out of bounds.
    ///
    /// Note that this method only accepts one-sided ranges such as
    /// `2..` or `..6`, but not `2..6`.
    ///
    /// # Examples
    ///
    /// Taking the first three elements of a slice:
    ///
    /// ```
    /// #![feature(slice_take)]
    ///
    /// let mut slice: &[_] = &['a', 'b', 'c', 'd'];
    /// let mut first_three = slice.take(..3).unwrap();
    ///
    /// assert_eq!(slice, &['d']);
    /// assert_eq!(first_three, &['a', 'b', 'c']);
    /// ```
    ///
    /// Taking the last two elements of a slice:
    ///
    /// ```
    /// #![feature(slice_take)]
    ///
    /// let mut slice: &[_] = &['a', 'b', 'c', 'd'];
    /// let mut tail = slice.take(2..).unwrap();
    ///
    /// assert_eq!(slice, &['a', 'b']);
    /// assert_eq!(tail, &['c', 'd']);
    /// ```
    ///
    /// Getting `None` when `range` is out of bounds:
    ///
    /// ```
    /// #![feature(slice_take)]
    ///
    /// let mut slice: &[_] = &['a', 'b', 'c', 'd'];
    ///
    /// assert_eq!(None, slice.take(5..));
    /// assert_eq!(None, slice.take(..5));
    /// assert_eq!(None, slice.take(..=4));
    /// let expected: &[char] = &['a', 'b', 'c', 'd'];
    /// assert_eq!(Some(expected), slice.take(..4));
    /// ```
    #[inline]
    #[must_use = "method does not modify the slice if the range is out of bounds"]
    #[unstable(feature = "slice_take", issue = "62280")]
    pub fn take<'a, R: OneSidedRange<usize>>(self: &mut &'a Self, range: R) -> Option<&'a Self> {
        let (direction, split_index) = split_point_of(range)?;
        if split_index > self.len() {
            return None;
        }
        let (front, back) = self.split_at(split_index);
        match direction {
            Direction::Front => {
                *self = back;
                Some(front)
            }
            Direction::Back => {
                *self = front;
                Some(back)
            }
        }
    }

    /// Removes the subslice corresponding to the given range
    /// and returns a mutable reference to it.
    ///
    /// Returns `None` and does not modify the slice if the given
    /// range is out of bounds.
    ///
    /// Note that this method only accepts one-sided ranges such as
    /// `2..` or `..6`, but not `2..6`.
    ///
    /// # Examples
    ///
    /// Taking the first three elements of a slice:
    ///
    /// ```
    /// #![feature(slice_take)]
    ///
    /// let mut slice: &mut [_] = &mut ['a', 'b', 'c', 'd'];
    /// let mut first_three = slice.take_mut(..3).unwrap();
    ///
    /// assert_eq!(slice, &mut ['d']);
    /// assert_eq!(first_three, &mut ['a', 'b', 'c']);
    /// ```
    ///
    /// Taking the last two elements of a slice:
    ///
    /// ```
    /// #![feature(slice_take)]
    ///
    /// let mut slice: &mut [_] = &mut ['a', 'b', 'c', 'd'];
    /// let mut tail = slice.take_mut(2..).unwrap();
    ///
    /// assert_eq!(slice, &mut ['a', 'b']);
    /// assert_eq!(tail, &mut ['c', 'd']);
    /// ```
    ///
    /// Getting `None` when `range` is out of bounds:
    ///
    /// ```
    /// #![feature(slice_take)]
    ///
    /// let mut slice: &mut [_] = &mut ['a', 'b', 'c', 'd'];
    ///
    /// assert_eq!(None, slice.take_mut(5..));
    /// assert_eq!(None, slice.take_mut(..5));
    /// assert_eq!(None, slice.take_mut(..=4));
    /// let expected: &mut [_] = &mut ['a', 'b', 'c', 'd'];
    /// assert_eq!(Some(expected), slice.take_mut(..4));
    /// ```
    #[inline]
    #[must_use = "method does not modify the slice if the range is out of bounds"]
    #[unstable(feature = "slice_take", issue = "62280")]
    pub fn take_mut<'a, R: OneSidedRange<usize>>(
        self: &mut &'a mut Self,
        range: R,
    ) -> Option<&'a mut Self> {
        let (direction, split_index) = split_point_of(range)?;
        if split_index > self.len() {
            return None;
        }
        let (front, back) = mem::take(self).split_at_mut(split_index);
        match direction {
            Direction::Front => {
                *self = back;
                Some(front)
            }
            Direction::Back => {
                *self = front;
                Some(back)
            }
        }
    }

    /// Removes the first element of the slice and returns a reference
    /// to it.
    ///
    /// Returns `None` if the slice is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_take)]
    ///
    /// let mut slice: &[_] = &['a', 'b', 'c'];
    /// let first = slice.take_first().unwrap();
    ///
    /// assert_eq!(slice, &['b', 'c']);
    /// assert_eq!(first, &'a');
    /// ```
    #[inline]
    #[unstable(feature = "slice_take", issue = "62280")]
    pub fn take_first<'a>(self: &mut &'a Self) -> Option<&'a T> {
        let (first, rem) = self.split_first()?;
        *self = rem;
        Some(first)
    }

    /// Removes the first element of the slice and returns a mutable
    /// reference to it.
    ///
    /// Returns `None` if the slice is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_take)]
    ///
    /// let mut slice: &mut [_] = &mut ['a', 'b', 'c'];
    /// let first = slice.take_first_mut().unwrap();
    /// *first = 'd';
    ///
    /// assert_eq!(slice, &['b', 'c']);
    /// assert_eq!(first, &'d');
    /// ```
    #[inline]
    #[unstable(feature = "slice_take", issue = "62280")]
    pub fn take_first_mut<'a>(self: &mut &'a mut Self) -> Option<&'a mut T> {
        let (first, rem) = mem::take(self).split_first_mut()?;
        *self = rem;
        Some(first)
    }

    /// Removes the last element of the slice and returns a reference
    /// to it.
    ///
    /// Returns `None` if the slice is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_take)]
    ///
    /// let mut slice: &[_] = &['a', 'b', 'c'];
    /// let last = slice.take_last().unwrap();
    ///
    /// assert_eq!(slice, &['a', 'b']);
    /// assert_eq!(last, &'c');
    /// ```
    #[inline]
    #[unstable(feature = "slice_take", issue = "62280")]
    pub fn take_last<'a>(self: &mut &'a Self) -> Option<&'a T> {
        let (last, rem) = self.split_last()?;
        *self = rem;
        Some(last)
    }

    /// Removes the last element of the slice and returns a mutable
    /// reference to it.
    ///
    /// Returns `None` if the slice is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_take)]
    ///
    /// let mut slice: &mut [_] = &mut ['a', 'b', 'c'];
    /// let last = slice.take_last_mut().unwrap();
    /// *last = 'd';
    ///
    /// assert_eq!(slice, &['a', 'b']);
    /// assert_eq!(last, &'d');
    /// ```
    #[inline]
    #[unstable(feature = "slice_take", issue = "62280")]
    pub fn take_last_mut<'a>(self: &mut &'a mut Self) -> Option<&'a mut T> {
        let (last, rem) = mem::take(self).split_last_mut()?;
        *self = rem;
        Some(last)
    }

    /// Returns mutable references to many indices at once, without doing any checks.
    ///
    /// An index can be either a `usize`, a [`Range`] or a [`RangeInclusive`]. Note
    /// that this method takes an array, so all indices must be of the same type.
    /// If passed an array of `usize`s this method gives back an array of mutable references
    /// to single elements, while if passed an array of ranges it gives back an array of
    /// mutable references to slices.
    ///
    /// For a safe alternative see [`get_many_mut`].
    ///
    /// # Safety
    ///
    /// Calling this method with overlapping or out-of-bounds indices is *[undefined behavior]*
    /// even if the resulting references are not used.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_many_mut)]
    ///
    /// let x = &mut [1, 2, 4];
    ///
    /// unsafe {
    ///     let [a, b] = x.get_many_unchecked_mut([0, 2]);
    ///     *a *= 10;
    ///     *b *= 100;
    /// }
    /// assert_eq!(x, &[10, 2, 400]);
    ///
    /// unsafe {
    ///     let [a, b] = x.get_many_unchecked_mut([0..1, 1..3]);
    ///     a[0] = 8;
    ///     b[0] = 88;
    ///     b[1] = 888;
    /// }
    /// assert_eq!(x, &[8, 88, 888]);
    ///
    /// unsafe {
    ///     let [a, b] = x.get_many_unchecked_mut([1..=2, 0..=0]);
    ///     a[0] = 11;
    ///     a[1] = 111;
    ///     b[0] = 1;
    /// }
    /// assert_eq!(x, &[1, 11, 111]);
    /// ```
    ///
    /// [`get_many_mut`]: slice::get_many_mut
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[unstable(feature = "get_many_mut", issue = "104642")]
    #[inline]
    pub unsafe fn get_many_unchecked_mut<I, const N: usize>(
        &mut self,
        indices: [I; N],
    ) -> [&mut I::Output; N]
    where
        I: GetManyMutIndex + SliceIndex<Self>,
    {
        // NB: This implementation is written as it is because any variation of
        // `indices.map(|i| self.get_unchecked_mut(i))` would make miri unhappy,
        // or generate worse code otherwise. This is also why we need to go
        // through a raw pointer here.
        let slice: *mut [T] = self;
        let mut arr: mem::MaybeUninit<[&mut I::Output; N]> = mem::MaybeUninit::uninit();
        let arr_ptr = arr.as_mut_ptr();

        // SAFETY: We expect `indices` to contain disjunct values that are
        // in bounds of `self`.
        unsafe {
            for i in 0..N {
                let idx = indices.get_unchecked(i).clone();
                arr_ptr.cast::<&mut I::Output>().add(i).write(&mut *slice.get_unchecked_mut(idx));
            }
            arr.assume_init()
        }
    }

    /// Returns mutable references to many indices at once.
    ///
    /// An index can be either a `usize`, a [`Range`] or a [`RangeInclusive`]. Note
    /// that this method takes an array, so all indices must be of the same type.
    /// If passed an array of `usize`s this method gives back an array of mutable references
    /// to single elements, while if passed an array of ranges it gives back an array of
    /// mutable references to slices.
    ///
    /// Returns an error if any index is out-of-bounds, or if there are overlapping indices.
    /// An empty range is not considered to overlap if it is located at the beginning or at
    /// the end of another range, but is considered to overlap if it is located in the middle.
    ///
    /// This method does a O(n^2) check to check that there are no overlapping indices, so be careful
    /// when passing many indices.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_many_mut)]
    ///
    /// let v = &mut [1, 2, 3];
    /// if let Ok([a, b]) = v.get_many_mut([0, 2]) {
    ///     *a = 413;
    ///     *b = 612;
    /// }
    /// assert_eq!(v, &[413, 2, 612]);
    ///
    /// if let Ok([a, b]) = v.get_many_mut([0..1, 1..3]) {
    ///     a[0] = 8;
    ///     b[0] = 88;
    ///     b[1] = 888;
    /// }
    /// assert_eq!(v, &[8, 88, 888]);
    ///
    /// if let Ok([a, b]) = v.get_many_mut([1..=2, 0..=0]) {
    ///     a[0] = 11;
    ///     a[1] = 111;
    ///     b[0] = 1;
    /// }
    /// assert_eq!(v, &[1, 11, 111]);
    /// ```
    #[unstable(feature = "get_many_mut", issue = "104642")]
    #[inline]
    pub fn get_many_mut<I, const N: usize>(
        &mut self,
        indices: [I; N],
    ) -> Result<[&mut I::Output; N], GetManyMutError>
    where
        I: GetManyMutIndex + SliceIndex<Self>,
    {
        get_many_check_valid(&indices, self.len())?;
        // SAFETY: The `get_many_check_valid()` call checked that all indices
        // are disjunct and in bounds.
        unsafe { Ok(self.get_many_unchecked_mut(indices)) }
    }

    /// Returns the index that an element reference points to.
    ///
    /// Returns `None` if `element` does not point to the start of an element within the slice.
    ///
    /// This method is useful for extending slice iterators like [`slice::split`].
    ///
    /// Note that this uses pointer arithmetic and **does not compare elements**.
    /// To find the index of an element via comparison, use
    /// [`.iter().position()`](crate::iter::Iterator::position) instead.
    ///
    /// # Panics
    /// Panics if `T` is zero-sized.
    ///
    /// # Examples
    /// Basic usage:
    /// ```
    /// #![feature(substr_range)]
    ///
    /// let nums: &[u32] = &[1, 7, 1, 1];
    /// let num = &nums[2];
    ///
    /// assert_eq!(num, &1);
    /// assert_eq!(nums.element_offset(num), Some(2));
    /// ```
    /// Returning `None` with an unaligned element:
    /// ```
    /// #![feature(substr_range)]
    ///
    /// let arr: &[[u32; 2]] = &[[0, 1], [2, 3]];
    /// let flat_arr: &[u32] = arr.as_flattened();
    ///
    /// let ok_elm: &[u32; 2] = flat_arr[0..2].try_into().unwrap();
    /// let weird_elm: &[u32; 2] = flat_arr[1..3].try_into().unwrap();
    ///
    /// assert_eq!(ok_elm, &[0, 1]);
    /// assert_eq!(weird_elm, &[1, 2]);
    ///
    /// assert_eq!(arr.element_offset(ok_elm), Some(0)); // Points to element 0
    /// assert_eq!(arr.element_offset(weird_elm), None); // Points between element 0 and 1
    /// ```
    #[must_use]
    #[unstable(feature = "substr_range", issue = "126769")]
    pub fn element_offset(&self, element: &T) -> Option<usize> {
        if T::IS_ZST {
            panic!("elements are zero-sized");
        }

        let self_start = self.as_ptr().addr();
        let elem_start = ptr::from_ref(element).addr();

        let byte_offset = elem_start.wrapping_sub(self_start);

        if byte_offset % mem::size_of::<T>() != 0 {
            return None;
        }

        let offset = byte_offset / mem::size_of::<T>();

        if offset < self.len() { Some(offset) } else { None }
    }

    /// Returns the range of indices that a subslice points to.
    ///
    /// Returns `None` if `subslice` does not point within the slice or if it is not aligned with the
    /// elements in the slice.
    ///
    /// This method **does not compare elements**. Instead, this method finds the location in the slice that
    /// `subslice` was obtained from. To find the index of a subslice via comparison, instead use
    /// [`.windows()`](slice::windows)[`.position()`](crate::iter::Iterator::position).
    ///
    /// This method is useful for extending slice iterators like [`slice::split`].
    ///
    /// Note that this may return a false positive (either `Some(0..0)` or `Some(self.len()..self.len())`)
    /// if `subslice` has a length of zero and points to the beginning or end of another, separate, slice.
    ///
    /// # Panics
    /// Panics if `T` is zero-sized.
    ///
    /// # Examples
    /// Basic usage:
    /// ```
    /// #![feature(substr_range)]
    ///
    /// let nums = &[0, 5, 10, 0, 0, 5];
    ///
    /// let mut iter = nums
    ///     .split(|t| *t == 0)
    ///     .map(|n| nums.subslice_range(n).unwrap());
    ///
    /// assert_eq!(iter.next(), Some(0..0));
    /// assert_eq!(iter.next(), Some(1..3));
    /// assert_eq!(iter.next(), Some(4..4));
    /// assert_eq!(iter.next(), Some(5..6));
    /// ```
    #[must_use]
    #[unstable(feature = "substr_range", issue = "126769")]
    pub fn subslice_range(&self, subslice: &[T]) -> Option<Range<usize>> {
        if T::IS_ZST {
            panic!("elements are zero-sized");
        }

        let self_start = self.as_ptr().addr();
        let subslice_start = subslice.as_ptr().addr();

        let byte_start = subslice_start.wrapping_sub(self_start);

        if byte_start % core::mem::size_of::<T>() != 0 {
            return None;
        }

        let start = byte_start / core::mem::size_of::<T>();
        let end = start.wrapping_add(subslice.len());

        if start <= self.len() && end <= self.len() { Some(start..end) } else { None }
    }
}

impl<T, const N: usize> [[T; N]] {
    /// Takes a `&[[T; N]]`, and flattens it to a `&[T]`.
    ///
    /// # Panics
    ///
    /// This panics if the length of the resulting slice would overflow a `usize`.
    ///
    /// This is only possible when flattening a slice of arrays of zero-sized
    /// types, and thus tends to be irrelevant in practice. If
    /// `size_of::<T>() > 0`, this will never panic.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!([[1, 2, 3], [4, 5, 6]].as_flattened(), &[1, 2, 3, 4, 5, 6]);
    ///
    /// assert_eq!(
    ///     [[1, 2, 3], [4, 5, 6]].as_flattened(),
    ///     [[1, 2], [3, 4], [5, 6]].as_flattened(),
    /// );
    ///
    /// let slice_of_empty_arrays: &[[i32; 0]] = &[[], [], [], [], []];
    /// assert!(slice_of_empty_arrays.as_flattened().is_empty());
    ///
    /// let empty_slice_of_arrays: &[[u32; 10]] = &[];
    /// assert!(empty_slice_of_arrays.as_flattened().is_empty());
    /// ```
    #[stable(feature = "slice_flatten", since = "1.80.0")]
    #[rustc_const_unstable(feature = "const_slice_flatten", issue = "95629")]
    pub const fn as_flattened(&self) -> &[T] {
        let len = if T::IS_ZST {
            self.len().checked_mul(N).expect("slice len overflow")
        } else {
            // SAFETY: `self.len() * N` cannot overflow because `self` is
            // already in the address space.
            unsafe { self.len().unchecked_mul(N) }
        };
        // SAFETY: `[T]` is layout-identical to `[T; N]`
        unsafe { from_raw_parts(self.as_ptr().cast(), len) }
    }

    /// Takes a `&mut [[T; N]]`, and flattens it to a `&mut [T]`.
    ///
    /// # Panics
    ///
    /// This panics if the length of the resulting slice would overflow a `usize`.
    ///
    /// This is only possible when flattening a slice of arrays of zero-sized
    /// types, and thus tends to be irrelevant in practice. If
    /// `size_of::<T>() > 0`, this will never panic.
    ///
    /// # Examples
    ///
    /// ```
    /// fn add_5_to_all(slice: &mut [i32]) {
    ///     for i in slice {
    ///         *i += 5;
    ///     }
    /// }
    ///
    /// let mut array = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// add_5_to_all(array.as_flattened_mut());
    /// assert_eq!(array, [[6, 7, 8], [9, 10, 11], [12, 13, 14]]);
    /// ```
    #[stable(feature = "slice_flatten", since = "1.80.0")]
    #[rustc_const_unstable(feature = "const_slice_flatten", issue = "95629")]
    pub const fn as_flattened_mut(&mut self) -> &mut [T] {
        let len = if T::IS_ZST {
            self.len().checked_mul(N).expect("slice len overflow")
        } else {
            // SAFETY: `self.len() * N` cannot overflow because `self` is
            // already in the address space.
            unsafe { self.len().unchecked_mul(N) }
        };
        // SAFETY: `[T]` is layout-identical to `[T; N]`
        unsafe { from_raw_parts_mut(self.as_mut_ptr().cast(), len) }
    }
}

#[cfg(not(test))]
impl [f32] {
    /// Sorts the slice of floats.
    ///
    /// This sort is in-place (i.e. does not allocate), *O*(*n* \* log(*n*)) worst-case, and uses
    /// the ordering defined by [`f32::total_cmp`].
    ///
    /// # Current implementation
    ///
    /// This uses the same sorting algorithm as [`sort_unstable_by`](slice::sort_unstable_by).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(sort_floats)]
    /// let mut v = [2.6, -5e-8, f32::NAN, 8.29, f32::INFINITY, -1.0, 0.0, -f32::INFINITY, -0.0];
    ///
    /// v.sort_floats();
    /// let sorted = [-f32::INFINITY, -1.0, -5e-8, -0.0, 0.0, 2.6, 8.29, f32::INFINITY, f32::NAN];
    /// assert_eq!(&v[..8], &sorted[..8]);
    /// assert!(v[8].is_nan());
    /// ```
    #[unstable(feature = "sort_floats", issue = "93396")]
    #[inline]
    pub fn sort_floats(&mut self) {
        self.sort_unstable_by(f32::total_cmp);
    }
}

#[cfg(not(test))]
impl [f64] {
    /// Sorts the slice of floats.
    ///
    /// This sort is in-place (i.e. does not allocate), *O*(*n* \* log(*n*)) worst-case, and uses
    /// the ordering defined by [`f64::total_cmp`].
    ///
    /// # Current implementation
    ///
    /// This uses the same sorting algorithm as [`sort_unstable_by`](slice::sort_unstable_by).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(sort_floats)]
    /// let mut v = [2.6, -5e-8, f64::NAN, 8.29, f64::INFINITY, -1.0, 0.0, -f64::INFINITY, -0.0];
    ///
    /// v.sort_floats();
    /// let sorted = [-f64::INFINITY, -1.0, -5e-8, -0.0, 0.0, 2.6, 8.29, f64::INFINITY, f64::NAN];
    /// assert_eq!(&v[..8], &sorted[..8]);
    /// assert!(v[8].is_nan());
    /// ```
    #[unstable(feature = "sort_floats", issue = "93396")]
    #[inline]
    pub fn sort_floats(&mut self) {
        self.sort_unstable_by(f64::total_cmp);
    }
}

trait CloneFromSpec<T> {
    fn spec_clone_from(&mut self, src: &[T]);
}

impl<T> CloneFromSpec<T> for [T]
where
    T: Clone,
{
    #[track_caller]
    default fn spec_clone_from(&mut self, src: &[T]) {
        assert!(self.len() == src.len(), "destination and source slices have different lengths");
        // NOTE: We need to explicitly slice them to the same length
        // to make it easier for the optimizer to elide bounds checking.
        // But since it can't be relied on we also have an explicit specialization for T: Copy.
        let len = self.len();
        let src = &src[..len];
        for i in 0..len {
            self[i].clone_from(&src[i]);
        }
    }
}

impl<T> CloneFromSpec<T> for [T]
where
    T: Copy,
{
    #[track_caller]
    fn spec_clone_from(&mut self, src: &[T]) {
        self.copy_from_slice(src);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for &[T] {
    /// Creates an empty slice.
    fn default() -> Self {
        &[]
    }
}

#[stable(feature = "mut_slice_default", since = "1.5.0")]
impl<T> Default for &mut [T] {
    /// Creates a mutable empty slice.
    fn default() -> Self {
        &mut []
    }
}

#[unstable(feature = "slice_pattern", reason = "stopgap trait for slice patterns", issue = "56345")]
/// Patterns in slices - currently, only used by `strip_prefix` and `strip_suffix`.  At a future
/// point, we hope to generalise `core::str::Pattern` (which at the time of writing is limited to
/// `str`) to slices, and then this trait will be replaced or abolished.
pub trait SlicePattern {
    /// The element type of the slice being matched on.
    type Item;

    /// Currently, the consumers of `SlicePattern` need a slice.
    fn as_slice(&self) -> &[Self::Item];
}

#[stable(feature = "slice_strip", since = "1.51.0")]
impl<T> SlicePattern for [T] {
    type Item = T;

    #[inline]
    fn as_slice(&self) -> &[Self::Item] {
        self
    }
}

#[stable(feature = "slice_strip", since = "1.51.0")]
impl<T, const N: usize> SlicePattern for [T; N] {
    type Item = T;

    #[inline]
    fn as_slice(&self) -> &[Self::Item] {
        self
    }
}

/// This checks every index against each other, and against `len`.
///
/// This will do `binomial(N + 1, 2) = N * (N + 1) / 2 = 0, 1, 3, 6, 10, ..`
/// comparison operations.
#[inline]
fn get_many_check_valid<I: GetManyMutIndex, const N: usize>(
    indices: &[I; N],
    len: usize,
) -> Result<(), GetManyMutError> {
    // NB: The optimizer should inline the loops into a sequence
    // of instructions without additional branching.
    for (i, idx) in indices.iter().enumerate() {
        if !idx.is_in_bounds(len) {
            return Err(GetManyMutError::IndexOutOfBounds);
        }
        for idx2 in &indices[..i] {
            if idx.is_overlapping(idx2) {
                return Err(GetManyMutError::OverlappingIndices);
            }
        }
    }
    Ok(())
}

/// The error type returned by [`get_many_mut`][`slice::get_many_mut`].
///
/// It indicates one of two possible errors:
/// - An index is out-of-bounds.
/// - The same index appeared multiple times in the array
///   (or different but overlapping indices when ranges are provided).
///
/// # Examples
///
/// ```
/// #![feature(get_many_mut)]
/// use std::slice::GetManyMutError;
///
/// let v = &mut [1, 2, 3];
/// assert_eq!(v.get_many_mut([0, 999]), Err(GetManyMutError::IndexOutOfBounds));
/// assert_eq!(v.get_many_mut([1, 1]), Err(GetManyMutError::OverlappingIndices));
/// ```
#[unstable(feature = "get_many_mut", issue = "104642")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GetManyMutError {
    /// An index provided was out-of-bounds for the slice.
    IndexOutOfBounds,
    /// Two indices provided were overlapping.
    OverlappingIndices,
}

#[unstable(feature = "get_many_mut", issue = "104642")]
impl fmt::Display for GetManyMutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            GetManyMutError::IndexOutOfBounds => "an index is out of bounds",
            GetManyMutError::OverlappingIndices => "there were overlapping indices",
        };
        fmt::Display::fmt(msg, f)
    }
}

mod private_get_many_mut_index {
    use super::{Range, RangeInclusive, range};

    #[unstable(feature = "get_many_mut_helpers", issue = "none")]
    pub trait Sealed {}

    #[unstable(feature = "get_many_mut_helpers", issue = "none")]
    impl Sealed for usize {}
    #[unstable(feature = "get_many_mut_helpers", issue = "none")]
    impl Sealed for Range<usize> {}
    #[unstable(feature = "get_many_mut_helpers", issue = "none")]
    impl Sealed for RangeInclusive<usize> {}
    #[unstable(feature = "get_many_mut_helpers", issue = "none")]
    impl Sealed for range::Range<usize> {}
    #[unstable(feature = "get_many_mut_helpers", issue = "none")]
    impl Sealed for range::RangeInclusive<usize> {}
}

/// A helper trait for `<[T]>::get_many_mut()`.
///
/// # Safety
///
/// If `is_in_bounds()` returns `true` and `is_overlapping()` returns `false`,
/// it must be safe to index the slice with the indices.
#[unstable(feature = "get_many_mut_helpers", issue = "none")]
pub unsafe trait GetManyMutIndex: Clone + private_get_many_mut_index::Sealed {
    /// Returns `true` if `self` is in bounds for `len` slice elements.
    #[unstable(feature = "get_many_mut_helpers", issue = "none")]
    fn is_in_bounds(&self, len: usize) -> bool;

    /// Returns `true` if `self` overlaps with `other`.
    ///
    /// Note that we don't consider zero-length ranges to overlap at the beginning or the end,
    /// but do consider them to overlap in the middle.
    #[unstable(feature = "get_many_mut_helpers", issue = "none")]
    fn is_overlapping(&self, other: &Self) -> bool;
}

#[unstable(feature = "get_many_mut_helpers", issue = "none")]
// SAFETY: We implement `is_in_bounds()` and `is_overlapping()` correctly.
unsafe impl GetManyMutIndex for usize {
    #[inline]
    fn is_in_bounds(&self, len: usize) -> bool {
        *self < len
    }

    #[inline]
    fn is_overlapping(&self, other: &Self) -> bool {
        *self == *other
    }
}

#[unstable(feature = "get_many_mut_helpers", issue = "none")]
// SAFETY: We implement `is_in_bounds()` and `is_overlapping()` correctly.
unsafe impl GetManyMutIndex for Range<usize> {
    #[inline]
    fn is_in_bounds(&self, len: usize) -> bool {
        (self.start <= self.end) & (self.end <= len)
    }

    #[inline]
    fn is_overlapping(&self, other: &Self) -> bool {
        (self.start < other.end) & (other.start < self.end)
    }
}

#[unstable(feature = "get_many_mut_helpers", issue = "none")]
// SAFETY: We implement `is_in_bounds()` and `is_overlapping()` correctly.
unsafe impl GetManyMutIndex for RangeInclusive<usize> {
    #[inline]
    fn is_in_bounds(&self, len: usize) -> bool {
        (self.start <= self.end) & (self.end < len)
    }

    #[inline]
    fn is_overlapping(&self, other: &Self) -> bool {
        (self.start <= other.end) & (other.start <= self.end)
    }
}

#[unstable(feature = "get_many_mut_helpers", issue = "none")]
// SAFETY: We implement `is_in_bounds()` and `is_overlapping()` correctly.
unsafe impl GetManyMutIndex for range::Range<usize> {
    #[inline]
    fn is_in_bounds(&self, len: usize) -> bool {
        Range::from(*self).is_in_bounds(len)
    }

    #[inline]
    fn is_overlapping(&self, other: &Self) -> bool {
        Range::from(*self).is_overlapping(&Range::from(*other))
    }
}

#[unstable(feature = "get_many_mut_helpers", issue = "none")]
// SAFETY: We implement `is_in_bounds()` and `is_overlapping()` correctly.
unsafe impl GetManyMutIndex for range::RangeInclusive<usize> {
    #[inline]
    fn is_in_bounds(&self, len: usize) -> bool {
        RangeInclusive::from(*self).is_in_bounds(len)
    }

    #[inline]
    fn is_overlapping(&self, other: &Self) -> bool {
        RangeInclusive::from(*self).is_overlapping(&RangeInclusive::from(*other))
    }
}
