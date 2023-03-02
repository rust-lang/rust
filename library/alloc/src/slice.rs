//! Utilities for the slice primitive type.
//!
//! *[See also the slice primitive type](slice).*
//!
//! Most of the structs in this module are iterator types which can only be created
//! using a certain function. For example, `slice.iter()` yields an [`Iter`].
//!
//! A few functions are provided to create a slice from a value reference
//! or from a raw pointer.
#![stable(feature = "rust1", since = "1.0.0")]
// Many of the usings in this module are only used in the test configuration.
// It's cleaner to just turn off the unused_imports warning than to fix them.
#![cfg_attr(test, allow(unused_imports, dead_code))]

use core::borrow::{Borrow, BorrowMut};
#[cfg(not(no_global_oom_handling))]
use core::cmp::Ordering::{self, Less};
#[cfg(not(no_global_oom_handling))]
use core::mem::{self, SizedTypeProperties};
#[cfg(not(no_global_oom_handling))]
use core::ptr;
#[cfg(not(no_global_oom_handling))]
use core::slice::sort;

use crate::alloc::Allocator;
#[cfg(not(no_global_oom_handling))]
use crate::alloc::{self, Global};
#[cfg(not(no_global_oom_handling))]
use crate::borrow::ToOwned;
use crate::boxed::Box;
use crate::vec::Vec;

#[cfg(test)]
mod tests;

#[unstable(feature = "slice_range", issue = "76393")]
pub use core::slice::range;
#[unstable(feature = "array_chunks", issue = "74985")]
pub use core::slice::ArrayChunks;
#[unstable(feature = "array_chunks", issue = "74985")]
pub use core::slice::ArrayChunksMut;
#[unstable(feature = "array_windows", issue = "75027")]
pub use core::slice::ArrayWindows;
#[stable(feature = "inherent_ascii_escape", since = "1.60.0")]
pub use core::slice::EscapeAscii;
#[stable(feature = "slice_get_slice", since = "1.28.0")]
pub use core::slice::SliceIndex;
#[stable(feature = "from_ref", since = "1.28.0")]
pub use core::slice::{from_mut, from_ref};
#[unstable(feature = "slice_from_ptr_range", issue = "89792")]
pub use core::slice::{from_mut_ptr_range, from_ptr_range};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::slice::{from_raw_parts, from_raw_parts_mut};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::slice::{Chunks, Windows};
#[stable(feature = "chunks_exact", since = "1.31.0")]
pub use core::slice::{ChunksExact, ChunksExactMut};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::slice::{ChunksMut, Split, SplitMut};
#[unstable(feature = "slice_group_by", issue = "80552")]
pub use core::slice::{GroupBy, GroupByMut};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::slice::{Iter, IterMut};
#[stable(feature = "rchunks", since = "1.31.0")]
pub use core::slice::{RChunks, RChunksExact, RChunksExactMut, RChunksMut};
#[stable(feature = "slice_rsplit", since = "1.27.0")]
pub use core::slice::{RSplit, RSplitMut};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::slice::{RSplitN, RSplitNMut, SplitN, SplitNMut};
#[stable(feature = "split_inclusive", since = "1.51.0")]
pub use core::slice::{SplitInclusive, SplitInclusiveMut};

////////////////////////////////////////////////////////////////////////////////
// Basic slice extension methods
////////////////////////////////////////////////////////////////////////////////

// HACK(japaric) needed for the implementation of `vec!` macro during testing
// N.B., see the `hack` module in this file for more details.
#[cfg(test)]
pub use hack::into_vec;

// HACK(japaric) needed for the implementation of `Vec::clone` during testing
// N.B., see the `hack` module in this file for more details.
#[cfg(test)]
pub use hack::to_vec;

// HACK(japaric): With cfg(test) `impl [T]` is not available, these three
// functions are actually methods that are in `impl [T]` but not in
// `core::slice::SliceExt` - we need to supply these functions for the
// `test_permutations` test
pub(crate) mod hack {
    use core::alloc::Allocator;

    use crate::boxed::Box;
    use crate::vec::Vec;

    // We shouldn't add inline attribute to this since this is used in
    // `vec!` macro mostly and causes perf regression. See #71204 for
    // discussion and perf results.
    pub fn into_vec<T, A: Allocator>(b: Box<[T], A>) -> Vec<T, A> {
        unsafe {
            let len = b.len();
            let (b, alloc) = Box::into_raw_with_allocator(b);
            Vec::from_raw_parts_in(b as *mut T, len, len, alloc)
        }
    }

    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub fn to_vec<T: ConvertVec, A: Allocator>(s: &[T], alloc: A) -> Vec<T, A> {
        T::to_vec(s, alloc)
    }

    #[cfg(not(no_global_oom_handling))]
    pub trait ConvertVec {
        fn to_vec<A: Allocator>(s: &[Self], alloc: A) -> Vec<Self, A>
        where
            Self: Sized;
    }

    #[cfg(not(no_global_oom_handling))]
    impl<T: Clone> ConvertVec for T {
        #[inline]
        default fn to_vec<A: Allocator>(s: &[Self], alloc: A) -> Vec<Self, A> {
            struct DropGuard<'a, T, A: Allocator> {
                vec: &'a mut Vec<T, A>,
                num_init: usize,
            }
            impl<'a, T, A: Allocator> Drop for DropGuard<'a, T, A> {
                #[inline]
                fn drop(&mut self) {
                    // SAFETY:
                    // items were marked initialized in the loop below
                    unsafe {
                        self.vec.set_len(self.num_init);
                    }
                }
            }
            let mut vec = Vec::with_capacity_in(s.len(), alloc);
            let mut guard = DropGuard { vec: &mut vec, num_init: 0 };
            let slots = guard.vec.spare_capacity_mut();
            // .take(slots.len()) is necessary for LLVM to remove bounds checks
            // and has better codegen than zip.
            for (i, b) in s.iter().enumerate().take(slots.len()) {
                guard.num_init = i;
                slots[i].write(b.clone());
            }
            core::mem::forget(guard);
            // SAFETY:
            // the vec was allocated and initialized above to at least this length.
            unsafe {
                vec.set_len(s.len());
            }
            vec
        }
    }

    #[cfg(not(no_global_oom_handling))]
    impl<T: Copy> ConvertVec for T {
        #[inline]
        fn to_vec<A: Allocator>(s: &[Self], alloc: A) -> Vec<Self, A> {
            let mut v = Vec::with_capacity_in(s.len(), alloc);
            // SAFETY:
            // allocated above with the capacity of `s`, and initialize to `s.len()` in
            // ptr::copy_to_non_overlapping below.
            unsafe {
                s.as_ptr().copy_to_nonoverlapping(v.as_mut_ptr(), s.len());
                v.set_len(s.len());
            }
            v
        }
    }
}

#[cfg(not(test))]
impl<T> [T] {
    /// Sorts the slice.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and *O*(*n* \* log(*n*)) worst-case.
    ///
    /// When applicable, unstable sorting is preferred because it is generally faster than stable
    /// sorting and it doesn't allocate auxiliary memory.
    /// See [`sort_unstable`](slice::sort_unstable).
    ///
    /// # Current implementation
    ///
    /// The current algorithm is an adaptive, iterative merge sort inspired by
    /// [timsort](https://en.wikipedia.org/wiki/Timsort).
    /// It is designed to be very fast in cases where the slice is nearly sorted, or consists of
    /// two or more sorted sequences concatenated one after another.
    ///
    /// Also, it allocates temporary storage half the size of `self`, but for short slices a
    /// non-allocating insertion sort is used instead.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [-5, 4, 1, -3, 2];
    ///
    /// v.sort();
    /// assert!(v == [-5, -3, 1, 2, 4]);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        stable_sort(self, T::lt);
    }

    /// Sorts the slice with a comparator function.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and *O*(*n* \* log(*n*)) worst-case.
    ///
    /// The comparator function must define a total ordering for the elements in the slice. If
    /// the ordering is not total, the order of the elements is unspecified. An order is a
    /// total order if it is (for all `a`, `b` and `c`):
    ///
    /// * total and antisymmetric: exactly one of `a < b`, `a == b` or `a > b` is true, and
    /// * transitive, `a < b` and `b < c` implies `a < c`. The same must hold for both `==` and `>`.
    ///
    /// For example, while [`f64`] doesn't implement [`Ord`] because `NaN != NaN`, we can use
    /// `partial_cmp` as our sort function when we know the slice doesn't contain a `NaN`.
    ///
    /// ```
    /// let mut floats = [5f64, 4.0, 1.0, 3.0, 2.0];
    /// floats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    /// assert_eq!(floats, [1.0, 2.0, 3.0, 4.0, 5.0]);
    /// ```
    ///
    /// When applicable, unstable sorting is preferred because it is generally faster than stable
    /// sorting and it doesn't allocate auxiliary memory.
    /// See [`sort_unstable_by`](slice::sort_unstable_by).
    ///
    /// # Current implementation
    ///
    /// The current algorithm is an adaptive, iterative merge sort inspired by
    /// [timsort](https://en.wikipedia.org/wiki/Timsort).
    /// It is designed to be very fast in cases where the slice is nearly sorted, or consists of
    /// two or more sorted sequences concatenated one after another.
    ///
    /// Also, it allocates temporary storage half the size of `self`, but for short slices a
    /// non-allocating insertion sort is used instead.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [5, 4, 1, 3, 2];
    /// v.sort_by(|a, b| a.cmp(b));
    /// assert!(v == [1, 2, 3, 4, 5]);
    ///
    /// // reverse sorting
    /// v.sort_by(|a, b| b.cmp(a));
    /// assert!(v == [5, 4, 3, 2, 1]);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn sort_by<F>(&mut self, mut compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        stable_sort(self, |a, b| compare(a, b) == Less);
    }

    /// Sorts the slice with a key extraction function.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and *O*(*m* \* *n* \* log(*n*))
    /// worst-case, where the key function is *O*(*m*).
    ///
    /// For expensive key functions (e.g. functions that are not simple property accesses or
    /// basic operations), [`sort_by_cached_key`](slice::sort_by_cached_key) is likely to be
    /// significantly faster, as it does not recompute element keys.
    ///
    /// When applicable, unstable sorting is preferred because it is generally faster than stable
    /// sorting and it doesn't allocate auxiliary memory.
    /// See [`sort_unstable_by_key`](slice::sort_unstable_by_key).
    ///
    /// # Current implementation
    ///
    /// The current algorithm is an adaptive, iterative merge sort inspired by
    /// [timsort](https://en.wikipedia.org/wiki/Timsort).
    /// It is designed to be very fast in cases where the slice is nearly sorted, or consists of
    /// two or more sorted sequences concatenated one after another.
    ///
    /// Also, it allocates temporary storage half the size of `self`, but for short slices a
    /// non-allocating insertion sort is used instead.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [-5i32, 4, 1, -3, 2];
    ///
    /// v.sort_by_key(|k| k.abs());
    /// assert!(v == [1, 2, -3, 4, -5]);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "slice_sort_by_key", since = "1.7.0")]
    #[inline]
    pub fn sort_by_key<K, F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        stable_sort(self, |a, b| f(a).lt(&f(b)));
    }

    /// Sorts the slice with a key extraction function.
    ///
    /// During sorting, the key function is called at most once per element, by using
    /// temporary storage to remember the results of key evaluation.
    /// The order of calls to the key function is unspecified and may change in future versions
    /// of the standard library.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and *O*(*m* \* *n* + *n* \* log(*n*))
    /// worst-case, where the key function is *O*(*m*).
    ///
    /// For simple key functions (e.g., functions that are property accesses or
    /// basic operations), [`sort_by_key`](slice::sort_by_key) is likely to be
    /// faster.
    ///
    /// # Current implementation
    ///
    /// The current algorithm is based on [pattern-defeating quicksort][pdqsort] by Orson Peters,
    /// which combines the fast average case of randomized quicksort with the fast worst case of
    /// heapsort, while achieving linear time on slices with certain patterns. It uses some
    /// randomization to avoid degenerate cases, but with a fixed seed to always provide
    /// deterministic behavior.
    ///
    /// In the worst case, the algorithm allocates temporary storage in a `Vec<(K, usize)>` the
    /// length of the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [-5i32, 4, 32, -3, 2];
    ///
    /// v.sort_by_cached_key(|k| k.to_string());
    /// assert!(v == [-3, -5, 2, 32, 4]);
    /// ```
    ///
    /// [pdqsort]: https://github.com/orlp/pdqsort
    #[cfg(not(no_global_oom_handling))]
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "slice_sort_by_cached_key", since = "1.34.0")]
    #[inline]
    pub fn sort_by_cached_key<K, F>(&mut self, f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        // Helper macro for indexing our vector by the smallest possible type, to reduce allocation.
        macro_rules! sort_by_key {
            ($t:ty, $slice:ident, $f:ident) => {{
                let mut indices: Vec<_> =
                    $slice.iter().map($f).enumerate().map(|(i, k)| (k, i as $t)).collect();
                // The elements of `indices` are unique, as they are indexed, so any sort will be
                // stable with respect to the original slice. We use `sort_unstable` here because
                // it requires less memory allocation.
                indices.sort_unstable();
                for i in 0..$slice.len() {
                    let mut index = indices[i].1;
                    while (index as usize) < i {
                        index = indices[index as usize].1;
                    }
                    indices[i].1 = index;
                    $slice.swap(i, index as usize);
                }
            }};
        }

        let sz_u8 = mem::size_of::<(K, u8)>();
        let sz_u16 = mem::size_of::<(K, u16)>();
        let sz_u32 = mem::size_of::<(K, u32)>();
        let sz_usize = mem::size_of::<(K, usize)>();

        let len = self.len();
        if len < 2 {
            return;
        }
        if sz_u8 < sz_u16 && len <= (u8::MAX as usize) {
            return sort_by_key!(u8, self, f);
        }
        if sz_u16 < sz_u32 && len <= (u16::MAX as usize) {
            return sort_by_key!(u16, self, f);
        }
        if sz_u32 < sz_usize && len <= (u32::MAX as usize) {
            return sort_by_key!(u32, self, f);
        }
        sort_by_key!(usize, self, f)
    }

    /// Copies `self` into a new `Vec`.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = [10, 40, 30];
    /// let x = s.to_vec();
    /// // Here, `s` and `x` can be modified independently.
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[rustc_allow_incoherent_impl]
    #[rustc_conversion_suggestion]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.to_vec_in(Global)
    }

    /// Copies `self` into a new `Vec` with an allocator.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::alloc::System;
    ///
    /// let s = [10, 40, 30];
    /// let x = s.to_vec_in(System);
    /// // Here, `s` and `x` can be modified independently.
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[rustc_allow_incoherent_impl]
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn to_vec_in<A: Allocator>(&self, alloc: A) -> Vec<T, A>
    where
        T: Clone,
    {
        // N.B., see the `hack` module in this file for more details.
        hack::to_vec(self, alloc)
    }

    /// Converts `self` into a vector without clones or allocation.
    ///
    /// The resulting vector can be converted back into a box via
    /// `Vec<T>`'s `into_boxed_slice` method.
    ///
    /// # Examples
    ///
    /// ```
    /// let s: Box<[i32]> = Box::new([10, 40, 30]);
    /// let x = s.into_vec();
    /// // `s` cannot be used anymore because it has been converted into `x`.
    ///
    /// assert_eq!(x, vec![10, 40, 30]);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn into_vec<A: Allocator>(self: Box<Self, A>) -> Vec<T, A> {
        // N.B., see the `hack` module in this file for more details.
        hack::into_vec(self)
    }

    /// Creates a vector by copying a slice `n` times.
    ///
    /// # Panics
    ///
    /// This function will panic if the capacity would overflow.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// assert_eq!([1, 2].repeat(3), vec![1, 2, 1, 2, 1, 2]);
    /// ```
    ///
    /// A panic upon overflow:
    ///
    /// ```should_panic
    /// // this will panic at runtime
    /// b"0123456789abcdef".repeat(usize::MAX);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "repeat_generic_slice", since = "1.40.0")]
    pub fn repeat(&self, n: usize) -> Vec<T>
    where
        T: Copy,
    {
        if n == 0 {
            return Vec::new();
        }

        // If `n` is larger than zero, it can be split as
        // `n = 2^expn + rem (2^expn > rem, expn >= 0, rem >= 0)`.
        // `2^expn` is the number represented by the leftmost '1' bit of `n`,
        // and `rem` is the remaining part of `n`.

        // Using `Vec` to access `set_len()`.
        let capacity = self.len().checked_mul(n).expect("capacity overflow");
        let mut buf = Vec::with_capacity(capacity);

        // `2^expn` repetition is done by doubling `buf` `expn`-times.
        buf.extend(self);
        {
            let mut m = n >> 1;
            // If `m > 0`, there are remaining bits up to the leftmost '1'.
            while m > 0 {
                // `buf.extend(buf)`:
                unsafe {
                    ptr::copy_nonoverlapping(
                        buf.as_ptr(),
                        (buf.as_mut_ptr() as *mut T).add(buf.len()),
                        buf.len(),
                    );
                    // `buf` has capacity of `self.len() * n`.
                    let buf_len = buf.len();
                    buf.set_len(buf_len * 2);
                }

                m >>= 1;
            }
        }

        // `rem` (`= n - 2^expn`) repetition is done by copying
        // first `rem` repetitions from `buf` itself.
        let rem_len = capacity - buf.len(); // `self.len() * rem`
        if rem_len > 0 {
            // `buf.extend(buf[0 .. rem_len])`:
            unsafe {
                // This is non-overlapping since `2^expn > rem`.
                ptr::copy_nonoverlapping(
                    buf.as_ptr(),
                    (buf.as_mut_ptr() as *mut T).add(buf.len()),
                    rem_len,
                );
                // `buf.len() + rem_len` equals to `buf.capacity()` (`= self.len() * n`).
                buf.set_len(capacity);
            }
        }
        buf
    }

    /// Flattens a slice of `T` into a single value `Self::Output`.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(["hello", "world"].concat(), "helloworld");
    /// assert_eq!([[1, 2], [3, 4]].concat(), [1, 2, 3, 4]);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn concat<Item: ?Sized>(&self) -> <Self as Concat<Item>>::Output
    where
        Self: Concat<Item>,
    {
        Concat::concat(self)
    }

    /// Flattens a slice of `T` into a single value `Self::Output`, placing a
    /// given separator between each.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(["hello", "world"].join(" "), "hello world");
    /// assert_eq!([[1, 2], [3, 4]].join(&0), [1, 2, 0, 3, 4]);
    /// assert_eq!([[1, 2], [3, 4]].join(&[0, 0][..]), [1, 2, 0, 0, 3, 4]);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "rename_connect_to_join", since = "1.3.0")]
    pub fn join<Separator>(&self, sep: Separator) -> <Self as Join<Separator>>::Output
    where
        Self: Join<Separator>,
    {
        Join::join(self, sep)
    }

    /// Flattens a slice of `T` into a single value `Self::Output`, placing a
    /// given separator between each.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(deprecated)]
    /// assert_eq!(["hello", "world"].connect(" "), "hello world");
    /// assert_eq!([[1, 2], [3, 4]].connect(&0), [1, 2, 0, 3, 4]);
    /// ```
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(since = "1.3.0", note = "renamed to join")]
    pub fn connect<Separator>(&self, sep: Separator) -> <Self as Join<Separator>>::Output
    where
        Self: Join<Separator>,
    {
        Join::join(self, sep)
    }
}

#[cfg(not(test))]
impl [u8] {
    /// Returns a vector containing a copy of this slice where each byte
    /// is mapped to its ASCII upper case equivalent.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To uppercase the value in-place, use [`make_ascii_uppercase`].
    ///
    /// [`make_ascii_uppercase`]: slice::make_ascii_uppercase
    #[cfg(not(no_global_oom_handling))]
    #[rustc_allow_incoherent_impl]
    #[must_use = "this returns the uppercase bytes as a new Vec, \
                  without modifying the original"]
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn to_ascii_uppercase(&self) -> Vec<u8> {
        let mut me = self.to_vec();
        me.make_ascii_uppercase();
        me
    }

    /// Returns a vector containing a copy of this slice where each byte
    /// is mapped to its ASCII lower case equivalent.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To lowercase the value in-place, use [`make_ascii_lowercase`].
    ///
    /// [`make_ascii_lowercase`]: slice::make_ascii_lowercase
    #[cfg(not(no_global_oom_handling))]
    #[rustc_allow_incoherent_impl]
    #[must_use = "this returns the lowercase bytes as a new Vec, \
                  without modifying the original"]
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn to_ascii_lowercase(&self) -> Vec<u8> {
        let mut me = self.to_vec();
        me.make_ascii_lowercase();
        me
    }
}

////////////////////////////////////////////////////////////////////////////////
// Extension traits for slices over specific kinds of data
////////////////////////////////////////////////////////////////////////////////

/// Helper trait for [`[T]::concat`](slice::concat).
///
/// Note: the `Item` type parameter is not used in this trait,
/// but it allows impls to be more generic.
/// Without it, we get this error:
///
/// ```error
/// error[E0207]: the type parameter `T` is not constrained by the impl trait, self type, or predica
///    --> library/alloc/src/slice.rs:608:6
///     |
/// 608 | impl<T: Clone, V: Borrow<[T]>> Concat for [V] {
///     |      ^ unconstrained type parameter
/// ```
///
/// This is because there could exist `V` types with multiple `Borrow<[_]>` impls,
/// such that multiple `T` types would apply:
///
/// ```
/// # #[allow(dead_code)]
/// pub struct Foo(Vec<u32>, Vec<String>);
///
/// impl std::borrow::Borrow<[u32]> for Foo {
///     fn borrow(&self) -> &[u32] { &self.0 }
/// }
///
/// impl std::borrow::Borrow<[String]> for Foo {
///     fn borrow(&self) -> &[String] { &self.1 }
/// }
/// ```
#[unstable(feature = "slice_concat_trait", issue = "27747")]
pub trait Concat<Item: ?Sized> {
    #[unstable(feature = "slice_concat_trait", issue = "27747")]
    /// The resulting type after concatenation
    type Output;

    /// Implementation of [`[T]::concat`](slice::concat)
    #[unstable(feature = "slice_concat_trait", issue = "27747")]
    fn concat(slice: &Self) -> Self::Output;
}

/// Helper trait for [`[T]::join`](slice::join)
#[unstable(feature = "slice_concat_trait", issue = "27747")]
pub trait Join<Separator> {
    #[unstable(feature = "slice_concat_trait", issue = "27747")]
    /// The resulting type after concatenation
    type Output;

    /// Implementation of [`[T]::join`](slice::join)
    #[unstable(feature = "slice_concat_trait", issue = "27747")]
    fn join(slice: &Self, sep: Separator) -> Self::Output;
}

#[cfg(not(no_global_oom_handling))]
#[unstable(feature = "slice_concat_ext", issue = "27747")]
impl<T: Clone, V: Borrow<[T]>> Concat<T> for [V] {
    type Output = Vec<T>;

    fn concat(slice: &Self) -> Vec<T> {
        let size = slice.iter().map(|slice| slice.borrow().len()).sum();
        let mut result = Vec::with_capacity(size);
        for v in slice {
            result.extend_from_slice(v.borrow())
        }
        result
    }
}

#[cfg(not(no_global_oom_handling))]
#[unstable(feature = "slice_concat_ext", issue = "27747")]
impl<T: Clone, V: Borrow<[T]>> Join<&T> for [V] {
    type Output = Vec<T>;

    fn join(slice: &Self, sep: &T) -> Vec<T> {
        let mut iter = slice.iter();
        let first = match iter.next() {
            Some(first) => first,
            None => return vec![],
        };
        let size = slice.iter().map(|v| v.borrow().len()).sum::<usize>() + slice.len() - 1;
        let mut result = Vec::with_capacity(size);
        result.extend_from_slice(first.borrow());

        for v in iter {
            result.push(sep.clone());
            result.extend_from_slice(v.borrow())
        }
        result
    }
}

#[cfg(not(no_global_oom_handling))]
#[unstable(feature = "slice_concat_ext", issue = "27747")]
impl<T: Clone, V: Borrow<[T]>> Join<&[T]> for [V] {
    type Output = Vec<T>;

    fn join(slice: &Self, sep: &[T]) -> Vec<T> {
        let mut iter = slice.iter();
        let first = match iter.next() {
            Some(first) => first,
            None => return vec![],
        };
        let size =
            slice.iter().map(|v| v.borrow().len()).sum::<usize>() + sep.len() * (slice.len() - 1);
        let mut result = Vec::with_capacity(size);
        result.extend_from_slice(first.borrow());

        for v in iter {
            result.extend_from_slice(sep);
            result.extend_from_slice(v.borrow())
        }
        result
    }
}

////////////////////////////////////////////////////////////////////////////////
// Standard trait implementations for slices
////////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, A: Allocator> Borrow<[T]> for Vec<T, A> {
    fn borrow(&self) -> &[T] {
        &self[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, A: Allocator> BorrowMut<[T]> for Vec<T, A> {
    fn borrow_mut(&mut self) -> &mut [T] {
        &mut self[..]
    }
}

// Specializable trait for implementing ToOwned::clone_into. This is
// public in the crate and has the Allocator parameter so that
// vec::clone_from use it too.
#[cfg(not(no_global_oom_handling))]
pub(crate) trait SpecCloneIntoVec<T, A: Allocator> {
    fn clone_into(&self, target: &mut Vec<T, A>);
}

#[cfg(not(no_global_oom_handling))]
impl<T: Clone, A: Allocator> SpecCloneIntoVec<T, A> for [T] {
    default fn clone_into(&self, target: &mut Vec<T, A>) {
        // drop anything in target that will not be overwritten
        target.truncate(self.len());

        // target.len <= self.len due to the truncate above, so the
        // slices here are always in-bounds.
        let (init, tail) = self.split_at(target.len());

        // reuse the contained values' allocations/resources.
        target.clone_from_slice(init);
        target.extend_from_slice(tail);
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T: Copy, A: Allocator> SpecCloneIntoVec<T, A> for [T] {
    fn clone_into(&self, target: &mut Vec<T, A>) {
        target.clear();
        target.extend_from_slice(self);
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Clone> ToOwned for [T] {
    type Owned = Vec<T>;
    #[cfg(not(test))]
    fn to_owned(&self) -> Vec<T> {
        self.to_vec()
    }

    #[cfg(test)]
    fn to_owned(&self) -> Vec<T> {
        hack::to_vec(self, Global)
    }

    fn clone_into(&self, target: &mut Vec<T>) {
        SpecCloneIntoVec::clone_into(self, target);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Sorting
////////////////////////////////////////////////////////////////////////////////

#[inline]
#[cfg(not(no_global_oom_handling))]
fn stable_sort<T, F>(v: &mut [T], mut is_less: F)
where
    F: FnMut(&T, &T) -> bool,
{
    if T::IS_ZST {
        // Sorting has no meaningful behavior on zero-sized types. Do nothing.
        return;
    }

    let elem_alloc_fn = |len: usize| -> *mut T {
        // SAFETY: Creating the layout is safe as long as merge_sort never calls this with len >
        // v.len(). Alloc in general will only be used as 'shadow-region' to store temporary swap
        // elements.
        unsafe { alloc::alloc(alloc::Layout::array::<T>(len).unwrap_unchecked()) as *mut T }
    };

    let elem_dealloc_fn = |buf_ptr: *mut T, len: usize| {
        // SAFETY: Creating the layout is safe as long as merge_sort never calls this with len >
        // v.len(). The caller must ensure that buf_ptr was created by elem_alloc_fn with the same
        // len.
        unsafe {
            alloc::dealloc(buf_ptr as *mut u8, alloc::Layout::array::<T>(len).unwrap_unchecked());
        }
    };

    let run_alloc_fn = |len: usize| -> *mut sort::TimSortRun {
        // SAFETY: Creating the layout is safe as long as merge_sort never calls this with an
        // obscene length or 0.
        unsafe {
            alloc::alloc(alloc::Layout::array::<sort::TimSortRun>(len).unwrap_unchecked())
                as *mut sort::TimSortRun
        }
    };

    let run_dealloc_fn = |buf_ptr: *mut sort::TimSortRun, len: usize| {
        // SAFETY: The caller must ensure that buf_ptr was created by elem_alloc_fn with the same
        // len.
        unsafe {
            alloc::dealloc(
                buf_ptr as *mut u8,
                alloc::Layout::array::<sort::TimSortRun>(len).unwrap_unchecked(),
            );
        }
    };

    // THIS IS A HACK, please do *NOT* merge this. Instantiate a separate thing that is not
    // sort_unstable to avoid reduced costs by code-sharing.
    if v.len() > 100 {
        dummy_test_sort::quicksort(core::hint::black_box(&mut v[..1]), &mut is_less);
    }

    sort::merge_sort(v, &mut is_less, elem_alloc_fn, elem_dealloc_fn, run_alloc_fn, run_dealloc_fn);
}

#[cfg(not(no_global_oom_handling))]
mod dummy_test_sort {
    //! Slice sorting
    //!
    //! This module contains a sorting algorithm based on Orson Peters' pattern-defeating quicksort,
    //! published at: <https://github.com/orlp/pdqsort>
    //!
    //! Unstable sorting is compatible with libcore because it doesn't allocate memory, unlike our
    //! stable sorting implementation.

    use core::cmp;
    use core::mem::{self, MaybeUninit};
    use core::ptr;

    /// When dropped, copies from `src` into `dest`.
    struct CopyOnDrop<T> {
        src: *const T,
        dest: *mut T,
    }

    impl<T> Drop for CopyOnDrop<T> {
        fn drop(&mut self) {
            // SAFETY:  This is a helper class.
            //          Please refer to its usage for correctness.
            //          Namely, one must be sure that `src` and `dst` does not overlap as required by `ptr::copy_nonoverlapping`.
            unsafe {
                ptr::copy_nonoverlapping(self.src, self.dest, 1);
            }
        }
    }

    /// Shifts the first element to the right until it encounters a greater or equal element.
    fn shift_head<T, F>(v: &mut [T], is_less: &mut F)
    where
        F: FnMut(&T, &T) -> bool,
    {
        let len = v.len();
        // SAFETY: The unsafe operations below involves indexing without a bounds check (by offsetting a
        // pointer) and copying memory (`ptr::copy_nonoverlapping`).
        //
        // a. Indexing:
        //  1. We checked the size of the array to >=2.
        //  2. All the indexing that we will do is always between {0 <= index < len} at most.
        //
        // b. Memory copying
        //  1. We are obtaining pointers to references which are guaranteed to be valid.
        //  2. They cannot overlap because we obtain pointers to difference indices of the slice.
        //     Namely, `i` and `i-1`.
        //  3. If the slice is properly aligned, the elements are properly aligned.
        //     It is the caller's responsibility to make sure the slice is properly aligned.
        //
        // See comments below for further detail.
        unsafe {
            // If the first two elements are out-of-order...
            if len >= 2 && is_less(v.get_unchecked(1), v.get_unchecked(0)) {
                // Read the first element into a stack-allocated variable. If a following comparison
                // operation panics, `hole` will get dropped and automatically write the element back
                // into the slice.
                let tmp = mem::ManuallyDrop::new(ptr::read(v.get_unchecked(0)));
                let v = v.as_mut_ptr();
                let mut hole = CopyOnDrop { src: &*tmp, dest: v.add(1) };
                ptr::copy_nonoverlapping(v.add(1), v.add(0), 1);

                for i in 2..len {
                    if !is_less(&*v.add(i), &*tmp) {
                        break;
                    }

                    // Move `i`-th element one place to the left, thus shifting the hole to the right.
                    ptr::copy_nonoverlapping(v.add(i), v.add(i - 1), 1);
                    hole.dest = v.add(i);
                }
                // `hole` gets dropped and thus copies `tmp` into the remaining hole in `v`.
            }
        }
    }

    /// Shifts the last element to the left until it encounters a smaller or equal element.
    fn shift_tail<T, F>(v: &mut [T], is_less: &mut F)
    where
        F: FnMut(&T, &T) -> bool,
    {
        let len = v.len();
        // SAFETY: The unsafe operations below involves indexing without a bound check (by offsetting a
        // pointer) and copying memory (`ptr::copy_nonoverlapping`).
        //
        // a. Indexing:
        //  1. We checked the size of the array to >= 2.
        //  2. All the indexing that we will do is always between `0 <= index < len-1` at most.
        //
        // b. Memory copying
        //  1. We are obtaining pointers to references which are guaranteed to be valid.
        //  2. They cannot overlap because we obtain pointers to difference indices of the slice.
        //     Namely, `i` and `i+1`.
        //  3. If the slice is properly aligned, the elements are properly aligned.
        //     It is the caller's responsibility to make sure the slice is properly aligned.
        //
        // See comments below for further detail.
        unsafe {
            // If the last two elements are out-of-order...
            if len >= 2 && is_less(v.get_unchecked(len - 1), v.get_unchecked(len - 2)) {
                // Read the last element into a stack-allocated variable. If a following comparison
                // operation panics, `hole` will get dropped and automatically write the element back
                // into the slice.
                let tmp = mem::ManuallyDrop::new(ptr::read(v.get_unchecked(len - 1)));
                let v = v.as_mut_ptr();
                let mut hole = CopyOnDrop { src: &*tmp, dest: v.add(len - 2) };
                ptr::copy_nonoverlapping(v.add(len - 2), v.add(len - 1), 1);

                for i in (0..len - 2).rev() {
                    if !is_less(&*tmp, &*v.add(i)) {
                        break;
                    }

                    // Move `i`-th element one place to the right, thus shifting the hole to the left.
                    ptr::copy_nonoverlapping(v.add(i), v.add(i + 1), 1);
                    hole.dest = v.add(i);
                }
                // `hole` gets dropped and thus copies `tmp` into the remaining hole in `v`.
            }
        }
    }

    /// Partially sorts a slice by shifting several out-of-order elements around.
    ///
    /// Returns `true` if the slice is sorted at the end. This function is *O*(*n*) worst-case.
    #[cold]
    fn partial_insertion_sort<T, F>(v: &mut [T], is_less: &mut F) -> bool
    where
        F: FnMut(&T, &T) -> bool,
    {
        // Maximum number of adjacent out-of-order pairs that will get shifted.
        const MAX_STEPS: usize = 5;
        // If the slice is shorter than this, don't shift any elements.
        const SHORTEST_SHIFTING: usize = 50;

        let len = v.len();
        let mut i = 1;

        for _ in 0..MAX_STEPS {
            // SAFETY: We already explicitly did the bound checking with `i < len`.
            // All our subsequent indexing is only in the range `0 <= index < len`
            unsafe {
                // Find the next pair of adjacent out-of-order elements.
                while i < len && !is_less(v.get_unchecked(i), v.get_unchecked(i - 1)) {
                    i += 1;
                }
            }

            // Are we done?
            if i == len {
                return true;
            }

            // Don't shift elements on short arrays, that has a performance cost.
            if len < SHORTEST_SHIFTING {
                return false;
            }

            // Swap the found pair of elements. This puts them in correct order.
            v.swap(i - 1, i);

            // Shift the smaller element to the left.
            shift_tail(&mut v[..i], is_less);
            // Shift the greater element to the right.
            shift_head(&mut v[i..], is_less);
        }

        // Didn't manage to sort the slice in the limited number of steps.
        false
    }

    /// Sorts a slice using insertion sort, which is *O*(*n*^2) worst-case.
    fn insertion_sort<T, F>(v: &mut [T], is_less: &mut F)
    where
        F: FnMut(&T, &T) -> bool,
    {
        for i in 1..v.len() {
            shift_tail(&mut v[..i + 1], is_less);
        }
    }

    /// Sorts `v` using heapsort, which guarantees *O*(*n* \* log(*n*)) worst-case.
    pub fn heapsort<T, F>(v: &mut [T], mut is_less: F)
    where
        F: FnMut(&T, &T) -> bool,
    {
        // This binary heap respects the invariant `parent >= child`.
        let mut sift_down = |v: &mut [T], mut node| {
            loop {
                // Children of `node`.
                let mut child = 2 * node + 1;
                if child >= v.len() {
                    break;
                }

                // Choose the greater child.
                if child + 1 < v.len() && is_less(&v[child], &v[child + 1]) {
                    child += 1;
                }

                // Stop if the invariant holds at `node`.
                if !is_less(&v[node], &v[child]) {
                    break;
                }

                // Swap `node` with the greater child, move one step down, and continue sifting.
                v.swap(node, child);
                node = child;
            }
        };

        // Build the heap in linear time.
        for i in (0..v.len() / 2).rev() {
            sift_down(v, i);
        }

        // Pop maximal elements from the heap.
        for i in (1..v.len()).rev() {
            v.swap(0, i);
            sift_down(&mut v[..i], 0);
        }
    }

    /// Partitions `v` into elements smaller than `pivot`, followed by elements greater than or equal
    /// to `pivot`.
    ///
    /// Returns the number of elements smaller than `pivot`.
    ///
    /// Partitioning is performed block-by-block in order to minimize the cost of branching operations.
    /// This idea is presented in the [BlockQuicksort][pdf] paper.
    ///
    /// [pdf]: https://drops.dagstuhl.de/opus/volltexte/2016/6389/pdf/LIPIcs-ESA-2016-38.pdf
    fn partition_in_blocks<T, F>(v: &mut [T], pivot: &T, is_less: &mut F) -> usize
    where
        F: FnMut(&T, &T) -> bool,
    {
        // Number of elements in a typical block.
        const BLOCK: usize = 128;

        // The partitioning algorithm repeats the following steps until completion:
        //
        // 1. Trace a block from the left side to identify elements greater than or equal to the pivot.
        // 2. Trace a block from the right side to identify elements smaller than the pivot.
        // 3. Exchange the identified elements between the left and right side.
        //
        // We keep the following variables for a block of elements:
        //
        // 1. `block` - Number of elements in the block.
        // 2. `start` - Start pointer into the `offsets` array.
        // 3. `end` - End pointer into the `offsets` array.
        // 4. `offsets - Indices of out-of-order elements within the block.

        // The current block on the left side (from `l` to `l.add(block_l)`).
        let mut l = v.as_mut_ptr();
        let mut block_l = BLOCK;
        let mut start_l = ptr::null_mut();
        let mut end_l = ptr::null_mut();
        let mut offsets_l = [MaybeUninit::<u8>::uninit(); BLOCK];

        // The current block on the right side (from `r.sub(block_r)` to `r`).
        // SAFETY: The documentation for .add() specifically mention that `vec.as_ptr().add(vec.len())` is always safe`
        let mut r = unsafe { l.add(v.len()) };
        let mut block_r = BLOCK;
        let mut start_r = ptr::null_mut();
        let mut end_r = ptr::null_mut();
        let mut offsets_r = [MaybeUninit::<u8>::uninit(); BLOCK];

        // FIXME: When we get VLAs, try creating one array of length `min(v.len(), 2 * BLOCK)` rather
        // than two fixed-size arrays of length `BLOCK`. VLAs might be more cache-efficient.

        // Returns the number of elements between pointers `l` (inclusive) and `r` (exclusive).
        fn width<T>(l: *mut T, r: *mut T) -> usize {
            assert!(mem::size_of::<T>() > 0);
            // FIXME: this should *likely* use `offset_from`, but more
            // investigation is needed (including running tests in miri).
            (r.addr() - l.addr()) / mem::size_of::<T>()
        }

        loop {
            // We are done with partitioning block-by-block when `l` and `r` get very close. Then we do
            // some patch-up work in order to partition the remaining elements in between.
            let is_done = width(l, r) <= 2 * BLOCK;

            if is_done {
                // Number of remaining elements (still not compared to the pivot).
                let mut rem = width(l, r);
                if start_l < end_l || start_r < end_r {
                    rem -= BLOCK;
                }

                // Adjust block sizes so that the left and right block don't overlap, but get perfectly
                // aligned to cover the whole remaining gap.
                if start_l < end_l {
                    block_r = rem;
                } else if start_r < end_r {
                    block_l = rem;
                } else {
                    // There were the same number of elements to switch on both blocks during the last
                    // iteration, so there are no remaining elements on either block. Cover the remaining
                    // items with roughly equally-sized blocks.
                    block_l = rem / 2;
                    block_r = rem - block_l;
                }
                debug_assert!(block_l <= BLOCK && block_r <= BLOCK);
                debug_assert!(width(l, r) == block_l + block_r);
            }

            if start_l == end_l {
                // Trace `block_l` elements from the left side.
                start_l = MaybeUninit::slice_as_mut_ptr(&mut offsets_l);
                end_l = start_l;
                let mut elem = l;

                for i in 0..block_l {
                    // SAFETY: The unsafety operations below involve the usage of the `offset`.
                    //         According to the conditions required by the function, we satisfy them because:
                    //         1. `offsets_l` is stack-allocated, and thus considered separate allocated object.
                    //         2. The function `is_less` returns a `bool`.
                    //            Casting a `bool` will never overflow `isize`.
                    //         3. We have guaranteed that `block_l` will be `<= BLOCK`.
                    //            Plus, `end_l` was initially set to the begin pointer of `offsets_` which was declared on the stack.
                    //            Thus, we know that even in the worst case (all invocations of `is_less` returns false) we will only be at most 1 byte pass the end.
                    //        Another unsafety operation here is dereferencing `elem`.
                    //        However, `elem` was initially the begin pointer to the slice which is always valid.
                    unsafe {
                        // Branchless comparison.
                        *end_l = i as u8;
                        end_l = end_l.offset(!is_less(&*elem, pivot) as isize);
                        elem = elem.offset(1);
                    }
                }
            }

            if start_r == end_r {
                // Trace `block_r` elements from the right side.
                start_r = MaybeUninit::slice_as_mut_ptr(&mut offsets_r);
                end_r = start_r;
                let mut elem = r;

                for i in 0..block_r {
                    // SAFETY: The unsafety operations below involve the usage of the `offset`.
                    //         According to the conditions required by the function, we satisfy them because:
                    //         1. `offsets_r` is stack-allocated, and thus considered separate allocated object.
                    //         2. The function `is_less` returns a `bool`.
                    //            Casting a `bool` will never overflow `isize`.
                    //         3. We have guaranteed that `block_r` will be `<= BLOCK`.
                    //            Plus, `end_r` was initially set to the begin pointer of `offsets_` which was declared on the stack.
                    //            Thus, we know that even in the worst case (all invocations of `is_less` returns true) we will only be at most 1 byte pass the end.
                    //        Another unsafety operation here is dereferencing `elem`.
                    //        However, `elem` was initially `1 * sizeof(T)` past the end and we decrement it by `1 * sizeof(T)` before accessing it.
                    //        Plus, `block_r` was asserted to be less than `BLOCK` and `elem` will therefore at most be pointing to the beginning of the slice.
                    unsafe {
                        // Branchless comparison.
                        elem = elem.offset(-1);
                        *end_r = i as u8;
                        end_r = end_r.offset(is_less(&*elem, pivot) as isize);
                    }
                }
            }

            // Number of out-of-order elements to swap between the left and right side.
            let count = cmp::min(width(start_l, end_l), width(start_r, end_r));

            if count > 0 {
                macro_rules! left {
                    () => {
                        l.offset(*start_l as isize)
                    };
                }
                macro_rules! right {
                    () => {
                        r.offset(-(*start_r as isize) - 1)
                    };
                }

                // Instead of swapping one pair at the time, it is more efficient to perform a cyclic
                // permutation. This is not strictly equivalent to swapping, but produces a similar
                // result using fewer memory operations.

                // SAFETY: The use of `ptr::read` is valid because there is at least one element in
                // both `offsets_l` and `offsets_r`, so `left!` is a valid pointer to read from.
                //
                // The uses of `left!` involve calls to `offset` on `l`, which points to the
                // beginning of `v`. All the offsets pointed-to by `start_l` are at most `block_l`, so
                // these `offset` calls are safe as all reads are within the block. The same argument
                // applies for the uses of `right!`.
                //
                // The calls to `start_l.offset` are valid because there are at most `count-1` of them,
                // plus the final one at the end of the unsafe block, where `count` is the minimum number
                // of collected offsets in `offsets_l` and `offsets_r`, so there is no risk of there not
                // being enough elements. The same reasoning applies to the calls to `start_r.offset`.
                //
                // The calls to `copy_nonoverlapping` are safe because `left!` and `right!` are guaranteed
                // not to overlap, and are valid because of the reasoning above.
                unsafe {
                    let tmp = ptr::read(left!());
                    ptr::copy_nonoverlapping(right!(), left!(), 1);

                    for _ in 1..count {
                        start_l = start_l.offset(1);
                        ptr::copy_nonoverlapping(left!(), right!(), 1);
                        start_r = start_r.offset(1);
                        ptr::copy_nonoverlapping(right!(), left!(), 1);
                    }

                    ptr::copy_nonoverlapping(&tmp, right!(), 1);
                    mem::forget(tmp);
                    start_l = start_l.offset(1);
                    start_r = start_r.offset(1);
                }
            }

            if start_l == end_l {
                // All out-of-order elements in the left block were moved. Move to the next block.

                // block-width-guarantee
                // SAFETY: if `!is_done` then the slice width is guaranteed to be at least `2*BLOCK` wide. There
                // are at most `BLOCK` elements in `offsets_l` because of its size, so the `offset` operation is
                // safe. Otherwise, the debug assertions in the `is_done` case guarantee that
                // `width(l, r) == block_l + block_r`, namely, that the block sizes have been adjusted to account
                // for the smaller number of remaining elements.
                l = unsafe { l.offset(block_l as isize) };
            }

            if start_r == end_r {
                // All out-of-order elements in the right block were moved. Move to the previous block.

                // SAFETY: Same argument as [block-width-guarantee]. Either this is a full block `2*BLOCK`-wide,
                // or `block_r` has been adjusted for the last handful of elements.
                r = unsafe { r.offset(-(block_r as isize)) };
            }

            if is_done {
                break;
            }
        }

        // All that remains now is at most one block (either the left or the right) with out-of-order
        // elements that need to be moved. Such remaining elements can be simply shifted to the end
        // within their block.

        if start_l < end_l {
            // The left block remains.
            // Move its remaining out-of-order elements to the far right.
            debug_assert_eq!(width(l, r), block_l);
            while start_l < end_l {
                // remaining-elements-safety
                // SAFETY: while the loop condition holds there are still elements in `offsets_l`, so it
                // is safe to point `end_l` to the previous element.
                //
                // The `ptr::swap` is safe if both its arguments are valid for reads and writes:
                //  - Per the debug assert above, the distance between `l` and `r` is `block_l`
                //    elements, so there can be at most `block_l` remaining offsets between `start_l`
                //    and `end_l`. This means `r` will be moved at most `block_l` steps back, which
                //    makes the `r.offset` calls valid (at that point `l == r`).
                //  - `offsets_l` contains valid offsets into `v` collected during the partitioning of
                //    the last block, so the `l.offset` calls are valid.
                unsafe {
                    end_l = end_l.offset(-1);
                    ptr::swap(l.offset(*end_l as isize), r.offset(-1));
                    r = r.offset(-1);
                }
            }
            width(v.as_mut_ptr(), r)
        } else if start_r < end_r {
            // The right block remains.
            // Move its remaining out-of-order elements to the far left.
            debug_assert_eq!(width(l, r), block_r);
            while start_r < end_r {
                // SAFETY: See the reasoning in [remaining-elements-safety].
                unsafe {
                    end_r = end_r.offset(-1);
                    ptr::swap(l, r.offset(-(*end_r as isize) - 1));
                    l = l.offset(1);
                }
            }
            width(v.as_mut_ptr(), l)
        } else {
            // Nothing else to do, we're done.
            width(v.as_mut_ptr(), l)
        }
    }

    /// Partitions `v` into elements smaller than `v[pivot]`, followed by elements greater than or
    /// equal to `v[pivot]`.
    ///
    /// Returns a tuple of:
    ///
    /// 1. Number of elements smaller than `v[pivot]`.
    /// 2. True if `v` was already partitioned.
    fn partition<T, F>(v: &mut [T], pivot: usize, is_less: &mut F) -> (usize, bool)
    where
        F: FnMut(&T, &T) -> bool,
    {
        let (mid, was_partitioned) = {
            // Place the pivot at the beginning of slice.
            v.swap(0, pivot);
            let (pivot, v) = v.split_at_mut(1);
            let pivot = &mut pivot[0];

            // Read the pivot into a stack-allocated variable for efficiency. If a following comparison
            // operation panics, the pivot will be automatically written back into the slice.

            // SAFETY: `pivot` is a reference to the first element of `v`, so `ptr::read` is safe.
            let tmp = mem::ManuallyDrop::new(unsafe { ptr::read(pivot) });
            let _pivot_guard = CopyOnDrop { src: &*tmp, dest: pivot };
            let pivot = &*tmp;

            // Find the first pair of out-of-order elements.
            let mut l = 0;
            let mut r = v.len();

            // SAFETY: The unsafety below involves indexing an array.
            // For the first one: We already do the bounds checking here with `l < r`.
            // For the second one: We initially have `l == 0` and `r == v.len()` and we checked that `l < r` at every indexing operation.
            //                     From here we know that `r` must be at least `r == l` which was shown to be valid from the first one.
            unsafe {
                // Find the first element greater than or equal to the pivot.
                while l < r && is_less(v.get_unchecked(l), pivot) {
                    l += 1;
                }

                // Find the last element smaller that the pivot.
                while l < r && !is_less(v.get_unchecked(r - 1), pivot) {
                    r -= 1;
                }
            }

            (l + partition_in_blocks(&mut v[l..r], pivot, is_less), l >= r)

            // `_pivot_guard` goes out of scope and writes the pivot (which is a stack-allocated
            // variable) back into the slice where it originally was. This step is critical in ensuring
            // safety!
        };

        // Place the pivot between the two partitions.
        v.swap(0, mid);

        (mid, was_partitioned)
    }

    /// Partitions `v` into elements equal to `v[pivot]` followed by elements greater than `v[pivot]`.
    ///
    /// Returns the number of elements equal to the pivot. It is assumed that `v` does not contain
    /// elements smaller than the pivot.
    fn partition_equal<T, F>(v: &mut [T], pivot: usize, is_less: &mut F) -> usize
    where
        F: FnMut(&T, &T) -> bool,
    {
        // Place the pivot at the beginning of slice.
        v.swap(0, pivot);
        let (pivot, v) = v.split_at_mut(1);
        let pivot = &mut pivot[0];

        // Read the pivot into a stack-allocated variable for efficiency. If a following comparison
        // operation panics, the pivot will be automatically written back into the slice.
        // SAFETY: The pointer here is valid because it is obtained from a reference to a slice.
        let tmp = mem::ManuallyDrop::new(unsafe { ptr::read(pivot) });
        let _pivot_guard = CopyOnDrop { src: &*tmp, dest: pivot };
        let pivot = &*tmp;

        // Now partition the slice.
        let mut l = 0;
        let mut r = v.len();
        loop {
            // SAFETY: The unsafety below involves indexing an array.
            // For the first one: We already do the bounds checking here with `l < r`.
            // For the second one: We initially have `l == 0` and `r == v.len()` and we checked that `l < r` at every indexing operation.
            //                     From here we know that `r` must be at least `r == l` which was shown to be valid from the first one.
            unsafe {
                // Find the first element greater than the pivot.
                while l < r && !is_less(pivot, v.get_unchecked(l)) {
                    l += 1;
                }

                // Find the last element equal to the pivot.
                while l < r && is_less(pivot, v.get_unchecked(r - 1)) {
                    r -= 1;
                }

                // Are we done?
                if l >= r {
                    break;
                }

                // Swap the found pair of out-of-order elements.
                r -= 1;
                let ptr = v.as_mut_ptr();
                ptr::swap(ptr.add(l), ptr.add(r));
                l += 1;
            }
        }

        // We found `l` elements equal to the pivot. Add 1 to account for the pivot itself.
        l + 1

        // `_pivot_guard` goes out of scope and writes the pivot (which is a stack-allocated variable)
        // back into the slice where it originally was. This step is critical in ensuring safety!
    }

    /// Scatters some elements around in an attempt to break patterns that might cause imbalanced
    /// partitions in quicksort.
    #[cold]
    fn break_patterns<T>(v: &mut [T]) {
        let len = v.len();
        if len >= 8 {
            // Pseudorandom number generator from the "Xorshift RNGs" paper by George Marsaglia.
            let mut random = len as u32;
            let mut gen_u32 = || {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                random
            };
            let mut gen_usize = || {
                if usize::BITS <= 32 {
                    gen_u32() as usize
                } else {
                    (((gen_u32() as u64) << 32) | (gen_u32() as u64)) as usize
                }
            };

            // Take random numbers modulo this number.
            // The number fits into `usize` because `len` is not greater than `isize::MAX`.
            let modulus = len.next_power_of_two();

            // Some pivot candidates will be in the nearby of this index. Let's randomize them.
            let pos = len / 4 * 2;

            for i in 0..3 {
                // Generate a random number modulo `len`. However, in order to avoid costly operations
                // we first take it modulo a power of two, and then decrease by `len` until it fits
                // into the range `[0, len - 1]`.
                let mut other = gen_usize() & (modulus - 1);

                // `other` is guaranteed to be less than `2 * len`.
                if other >= len {
                    other -= len;
                }

                v.swap(pos - 1 + i, other);
            }
        }
    }

    /// Chooses a pivot in `v` and returns the index and `true` if the slice is likely already sorted.
    ///
    /// Elements in `v` might be reordered in the process.
    fn choose_pivot<T, F>(v: &mut [T], is_less: &mut F) -> (usize, bool)
    where
        F: FnMut(&T, &T) -> bool,
    {
        // Minimum length to choose the median-of-medians method.
        // Shorter slices use the simple median-of-three method.
        const SHORTEST_MEDIAN_OF_MEDIANS: usize = 50;
        // Maximum number of swaps that can be performed in this function.
        const MAX_SWAPS: usize = 4 * 3;

        let len = v.len();

        // Three indices near which we are going to choose a pivot.
        let mut a = len / 4 * 1;
        let mut b = len / 4 * 2;
        let mut c = len / 4 * 3;

        // Counts the total number of swaps we are about to perform while sorting indices.
        let mut swaps = 0;

        if len >= 8 {
            // Swaps indices so that `v[a] <= v[b]`.
            // SAFETY: `len >= 8` so there are at least two elements in the neighborhoods of
            // `a`, `b` and `c`. This means the three calls to `sort_adjacent` result in
            // corresponding calls to `sort3` with valid 3-item neighborhoods around each
            // pointer, which in turn means the calls to `sort2` are done with valid
            // references. Thus the `v.get_unchecked` calls are safe, as is the `ptr::swap`
            // call.
            let mut sort2 = |a: &mut usize, b: &mut usize| unsafe {
                if is_less(v.get_unchecked(*b), v.get_unchecked(*a)) {
                    ptr::swap(a, b);
                    swaps += 1;
                }
            };

            // Swaps indices so that `v[a] <= v[b] <= v[c]`.
            let mut sort3 = |a: &mut usize, b: &mut usize, c: &mut usize| {
                sort2(a, b);
                sort2(b, c);
                sort2(a, b);
            };

            if len >= SHORTEST_MEDIAN_OF_MEDIANS {
                // Finds the median of `v[a - 1], v[a], v[a + 1]` and stores the index into `a`.
                let mut sort_adjacent = |a: &mut usize| {
                    let tmp = *a;
                    sort3(&mut (tmp - 1), a, &mut (tmp + 1));
                };

                // Find medians in the neighborhoods of `a`, `b`, and `c`.
                sort_adjacent(&mut a);
                sort_adjacent(&mut b);
                sort_adjacent(&mut c);
            }

            // Find the median among `a`, `b`, and `c`.
            sort3(&mut a, &mut b, &mut c);
        }

        if swaps < MAX_SWAPS {
            (b, swaps == 0)
        } else {
            // The maximum number of swaps was performed. Chances are the slice is descending or mostly
            // descending, so reversing will probably help sort it faster.
            v.reverse();
            (len - 1 - b, true)
        }
    }

    /// Sorts `v` recursively.
    ///
    /// If the slice had a predecessor in the original array, it is specified as `pred`.
    ///
    /// `limit` is the number of allowed imbalanced partitions before switching to `heapsort`. If zero,
    /// this function will immediately switch to heapsort.
    fn recurse<'a, T, F>(
        mut v: &'a mut [T],
        is_less: &mut F,
        mut pred: Option<&'a T>,
        mut limit: u32,
    ) where
        F: FnMut(&T, &T) -> bool,
    {
        // Slices of up to this length get sorted using insertion sort.
        const MAX_INSERTION: usize = 20;

        // True if the last partitioning was reasonably balanced.
        let mut was_balanced = true;
        // True if the last partitioning didn't shuffle elements (the slice was already partitioned).
        let mut was_partitioned = true;

        loop {
            let len = v.len();

            // Very short slices get sorted using insertion sort.
            if len <= MAX_INSERTION {
                insertion_sort(v, is_less);
                return;
            }

            // If too many bad pivot choices were made, simply fall back to heapsort in order to
            // guarantee `O(n * log(n))` worst-case.
            if limit == 0 {
                heapsort(v, is_less);
                return;
            }

            // If the last partitioning was imbalanced, try breaking patterns in the slice by shuffling
            // some elements around. Hopefully we'll choose a better pivot this time.
            if !was_balanced {
                break_patterns(v);
                limit -= 1;
            }

            // Choose a pivot and try guessing whether the slice is already sorted.
            let (pivot, likely_sorted) = choose_pivot(v, is_less);

            // If the last partitioning was decently balanced and didn't shuffle elements, and if pivot
            // selection predicts the slice is likely already sorted...
            if was_balanced && was_partitioned && likely_sorted {
                // Try identifying several out-of-order elements and shifting them to correct
                // positions. If the slice ends up being completely sorted, we're done.
                if partial_insertion_sort(v, is_less) {
                    return;
                }
            }

            // If the chosen pivot is equal to the predecessor, then it's the smallest element in the
            // slice. Partition the slice into elements equal to and elements greater than the pivot.
            // This case is usually hit when the slice contains many duplicate elements.
            if let Some(p) = pred {
                if !is_less(p, &v[pivot]) {
                    let mid = partition_equal(v, pivot, is_less);

                    // Continue sorting elements greater than the pivot.
                    v = &mut v[mid..];
                    continue;
                }
            }

            // Partition the slice.
            let (mid, was_p) = partition(v, pivot, is_less);
            was_balanced = cmp::min(mid, len - mid) >= len / 8;
            was_partitioned = was_p;

            // Split the slice into `left`, `pivot`, and `right`.
            let (left, right) = v.split_at_mut(mid);
            let (pivot, right) = right.split_at_mut(1);
            let pivot = &pivot[0];

            // Recurse into the shorter side only in order to minimize the total number of recursive
            // calls and consume less stack space. Then just continue with the longer side (this is
            // akin to tail recursion).
            if left.len() < right.len() {
                recurse(left, is_less, pred, limit);
                v = right;
                pred = Some(pivot);
            } else {
                recurse(right, is_less, Some(pivot), limit);
                v = left;
            }
        }
    }

    /// Sorts `v` using pattern-defeating quicksort, which is *O*(*n* \* log(*n*)) worst-case.
    pub fn quicksort<T, F>(v: &mut [T], mut is_less: F)
    where
        F: FnMut(&T, &T) -> bool,
    {
        // Sorting has no meaningful behavior on zero-sized types.
        if mem::size_of::<T>() == 0 {
            return;
        }

        // Limit the number of imbalanced partitions to `floor(log2(len)) + 1`.
        let limit = usize::BITS - v.len().leading_zeros();

        recurse(v, &mut is_less, None, limit);
    }
}
