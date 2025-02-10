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
use core::mem::{self, MaybeUninit};
#[cfg(not(no_global_oom_handling))]
use core::ptr;
#[unstable(feature = "array_chunks", issue = "74985")]
pub use core::slice::ArrayChunks;
#[unstable(feature = "array_chunks", issue = "74985")]
pub use core::slice::ArrayChunksMut;
#[unstable(feature = "array_windows", issue = "75027")]
pub use core::slice::ArrayWindows;
#[stable(feature = "inherent_ascii_escape", since = "1.60.0")]
pub use core::slice::EscapeAscii;
#[unstable(feature = "get_many_mut", issue = "104642")]
pub use core::slice::GetManyMutError;
#[stable(feature = "slice_get_slice", since = "1.28.0")]
pub use core::slice::SliceIndex;
#[cfg(not(no_global_oom_handling))]
use core::slice::sort;
#[stable(feature = "slice_group_by", since = "1.77.0")]
pub use core::slice::{ChunkBy, ChunkByMut};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::slice::{Chunks, Windows};
#[stable(feature = "chunks_exact", since = "1.31.0")]
pub use core::slice::{ChunksExact, ChunksExactMut};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::slice::{ChunksMut, Split, SplitMut};
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
#[stable(feature = "from_ref", since = "1.28.0")]
pub use core::slice::{from_mut, from_ref};
#[unstable(feature = "slice_from_ptr_range", issue = "89792")]
pub use core::slice::{from_mut_ptr_range, from_ptr_range};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::slice::{from_raw_parts, from_raw_parts_mut};
#[unstable(feature = "slice_range", issue = "76393")]
pub use core::slice::{range, try_range};

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

use crate::alloc::Allocator;
#[cfg(not(no_global_oom_handling))]
use crate::alloc::Global;
#[cfg(not(no_global_oom_handling))]
use crate::borrow::ToOwned;
use crate::boxed::Box;
use crate::vec::Vec;

// HACK(japaric): With cfg(test) `impl [T]` is not available, these three
// functions are actually methods that are in `impl [T]` but not in
// `core::slice::SliceExt` - we need to supply these functions for the
// `test_permutations` test
#[allow(unreachable_pub)] // cfg(test) pub above
pub(crate) mod hack {
    use core::alloc::Allocator;

    use crate::boxed::Box;
    use crate::vec::Vec;

    // We shouldn't add inline attribute to this since this is used in
    // `vec!` macro mostly and causes perf regression. See #71204 for
    // discussion and perf results.
    #[allow(missing_docs)]
    pub fn into_vec<T, A: Allocator>(b: Box<[T], A>) -> Vec<T, A> {
        unsafe {
            let len = b.len();
            let (b, alloc) = Box::into_raw_with_allocator(b);
            Vec::from_raw_parts_in(b as *mut T, len, len, alloc)
        }
    }

    #[cfg(not(no_global_oom_handling))]
    #[allow(missing_docs)]
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
    /// Sorts the slice, preserving initial order of equal elements.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and *O*(*n* \* log(*n*))
    /// worst-case.
    ///
    /// If the implementation of [`Ord`] for `T` does not implement a [total order], the function
    /// may panic; even if the function exits normally, the resulting order of elements in the slice
    /// is unspecified. See also the note on panicking below.
    ///
    /// When applicable, unstable sorting is preferred because it is generally faster than stable
    /// sorting and it doesn't allocate auxiliary memory. See
    /// [`sort_unstable`](slice::sort_unstable). The exception are partially sorted slices, which
    /// may be better served with `slice::sort`.
    ///
    /// Sorting types that only implement [`PartialOrd`] such as [`f32`] and [`f64`] require
    /// additional precautions. For example, `f32::NAN != f32::NAN`, which doesn't fulfill the
    /// reflexivity requirement of [`Ord`]. By using an alternative comparison function with
    /// `slice::sort_by` such as [`f32::total_cmp`] or [`f64::total_cmp`] that defines a [total
    /// order] users can sort slices containing floating-point values. Alternatively, if all values
    /// in the slice are guaranteed to be in a subset for which [`PartialOrd::partial_cmp`] forms a
    /// [total order], it's possible to sort the slice with `sort_by(|a, b|
    /// a.partial_cmp(b).unwrap())`.
    ///
    /// # Current implementation
    ///
    /// The current implementation is based on [driftsort] by Orson Peters and Lukas Bergdoll, which
    /// combines the fast average case of quicksort with the fast worst case and partial run
    /// detection of mergesort, achieving linear time on fully sorted and reversed inputs. On inputs
    /// with k distinct elements, the expected time to sort the data is *O*(*n* \* log(*k*)).
    ///
    /// The auxiliary memory allocation behavior depends on the input length. Short slices are
    /// handled without allocation, medium sized slices allocate `self.len()` and beyond that it
    /// clamps at `self.len() / 2`.
    ///
    /// # Panics
    ///
    /// May panic if the implementation of [`Ord`] for `T` does not implement a [total order], or if
    /// the [`Ord`] implementation itself panics.
    ///
    /// All safe functions on slices preserve the invariant that even if the function panics, all
    /// original elements will remain in the slice and any possible modifications via interior
    /// mutability are observed in the input. This ensures that recovery code (for instance inside
    /// of a `Drop` or following a `catch_unwind`) will still have access to all the original
    /// elements. For instance, if the slice belongs to a `Vec`, the `Vec::drop` method will be able
    /// to dispose of all contained elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [4, -5, 1, -3, 2];
    ///
    /// v.sort();
    /// assert_eq!(v, [-5, -3, 1, 2, 4]);
    /// ```
    ///
    /// [driftsort]: https://github.com/Voultapher/driftsort
    /// [total order]: https://en.wikipedia.org/wiki/Total_order
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

    /// Sorts the slice with a comparison function, preserving initial order of equal elements.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and *O*(*n* \* log(*n*))
    /// worst-case.
    ///
    /// If the comparison function `compare` does not implement a [total order], the function may
    /// panic; even if the function exits normally, the resulting order of elements in the slice is
    /// unspecified. See also the note on panicking below.
    ///
    /// For example `|a, b| (a - b).cmp(a)` is a comparison function that is neither transitive nor
    /// reflexive nor total, `a < b < c < a` with `a = 1, b = 2, c = 3`. For more information and
    /// examples see the [`Ord`] documentation.
    ///
    /// # Current implementation
    ///
    /// The current implementation is based on [driftsort] by Orson Peters and Lukas Bergdoll, which
    /// combines the fast average case of quicksort with the fast worst case and partial run
    /// detection of mergesort, achieving linear time on fully sorted and reversed inputs. On inputs
    /// with k distinct elements, the expected time to sort the data is *O*(*n* \* log(*k*)).
    ///
    /// The auxiliary memory allocation behavior depends on the input length. Short slices are
    /// handled without allocation, medium sized slices allocate `self.len()` and beyond that it
    /// clamps at `self.len() / 2`.
    ///
    /// # Panics
    ///
    /// May panic if `compare` does not implement a [total order], or if `compare` itself panics.
    ///
    /// All safe functions on slices preserve the invariant that even if the function panics, all
    /// original elements will remain in the slice and any possible modifications via interior
    /// mutability are observed in the input. This ensures that recovery code (for instance inside
    /// of a `Drop` or following a `catch_unwind`) will still have access to all the original
    /// elements. For instance, if the slice belongs to a `Vec`, the `Vec::drop` method will be able
    /// to dispose of all contained elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [4, -5, 1, -3, 2];
    /// v.sort_by(|a, b| a.cmp(b));
    /// assert_eq!(v, [-5, -3, 1, 2, 4]);
    ///
    /// // reverse sorting
    /// v.sort_by(|a, b| b.cmp(a));
    /// assert_eq!(v, [4, 2, 1, -3, -5]);
    /// ```
    ///
    /// [driftsort]: https://github.com/Voultapher/driftsort
    /// [total order]: https://en.wikipedia.org/wiki/Total_order
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

    /// Sorts the slice with a key extraction function, preserving initial order of equal elements.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and *O*(*m* \* *n* \* log(*n*))
    /// worst-case, where the key function is *O*(*m*).
    ///
    /// If the implementation of [`Ord`] for `K` does not implement a [total order], the function
    /// may panic; even if the function exits normally, the resulting order of elements in the slice
    /// is unspecified. See also the note on panicking below.
    ///
    /// # Current implementation
    ///
    /// The current implementation is based on [driftsort] by Orson Peters and Lukas Bergdoll, which
    /// combines the fast average case of quicksort with the fast worst case and partial run
    /// detection of mergesort, achieving linear time on fully sorted and reversed inputs. On inputs
    /// with k distinct elements, the expected time to sort the data is *O*(*n* \* log(*k*)).
    ///
    /// The auxiliary memory allocation behavior depends on the input length. Short slices are
    /// handled without allocation, medium sized slices allocate `self.len()` and beyond that it
    /// clamps at `self.len() / 2`.
    ///
    /// # Panics
    ///
    /// May panic if the implementation of [`Ord`] for `K` does not implement a [total order], or if
    /// the [`Ord`] implementation or the key-function `f` panics.
    ///
    /// All safe functions on slices preserve the invariant that even if the function panics, all
    /// original elements will remain in the slice and any possible modifications via interior
    /// mutability are observed in the input. This ensures that recovery code (for instance inside
    /// of a `Drop` or following a `catch_unwind`) will still have access to all the original
    /// elements. For instance, if the slice belongs to a `Vec`, the `Vec::drop` method will be able
    /// to dispose of all contained elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [4i32, -5, 1, -3, 2];
    ///
    /// v.sort_by_key(|k| k.abs());
    /// assert_eq!(v, [1, 2, -3, 4, -5]);
    /// ```
    ///
    /// [driftsort]: https://github.com/Voultapher/driftsort
    /// [total order]: https://en.wikipedia.org/wiki/Total_order
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

    /// Sorts the slice with a key extraction function, preserving initial order of equal elements.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and *O*(*m* \* *n* + *n* \*
    /// log(*n*)) worst-case, where the key function is *O*(*m*).
    ///
    /// During sorting, the key function is called at most once per element, by using temporary
    /// storage to remember the results of key evaluation. The order of calls to the key function is
    /// unspecified and may change in future versions of the standard library.
    ///
    /// If the implementation of [`Ord`] for `K` does not implement a [total order], the function
    /// may panic; even if the function exits normally, the resulting order of elements in the slice
    /// is unspecified. See also the note on panicking below.
    ///
    /// For simple key functions (e.g., functions that are property accesses or basic operations),
    /// [`sort_by_key`](slice::sort_by_key) is likely to be faster.
    ///
    /// # Current implementation
    ///
    /// The current implementation is based on [instruction-parallel-network sort][ipnsort] by Lukas
    /// Bergdoll, which combines the fast average case of randomized quicksort with the fast worst
    /// case of heapsort, while achieving linear time on fully sorted and reversed inputs. And
    /// *O*(*k* \* log(*n*)) where *k* is the number of distinct elements in the input. It leverages
    /// superscalar out-of-order execution capabilities commonly found in CPUs, to efficiently
    /// perform the operation.
    ///
    /// In the worst case, the algorithm allocates temporary storage in a `Vec<(K, usize)>` the
    /// length of the slice.
    ///
    /// # Panics
    ///
    /// May panic if the implementation of [`Ord`] for `K` does not implement a [total order], or if
    /// the [`Ord`] implementation panics.
    ///
    /// All safe functions on slices preserve the invariant that even if the function panics, all
    /// original elements will remain in the slice and any possible modifications via interior
    /// mutability are observed in the input. This ensures that recovery code (for instance inside
    /// of a `Drop` or following a `catch_unwind`) will still have access to all the original
    /// elements. For instance, if the slice belongs to a `Vec`, the `Vec::drop` method will be able
    /// to dispose of all contained elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [4i32, -5, 1, -3, 2, 10];
    ///
    /// // Strings are sorted by lexicographical order.
    /// v.sort_by_cached_key(|k| k.to_string());
    /// assert_eq!(v, [-3, -5, 1, 10, 2, 4]);
    /// ```
    ///
    /// [ipnsort]: https://github.com/Voultapher/sort-research-rs/tree/main/ipnsort
    /// [total order]: https://en.wikipedia.org/wiki/Total_order
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
                // it requires no memory allocation.
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

        let len = self.len();
        if len < 2 {
            return;
        }

        // Avoids binary-size usage in cases where the alignment doesn't work out to make this
        // beneficial or on 32-bit platforms.
        let is_using_u32_as_idx_type_helpful =
            const { mem::size_of::<(K, u32)>() < mem::size_of::<(K, usize)>() };

        // It's possible to instantiate this for u8 and u16 but, doing so is very wasteful in terms
        // of compile-times and binary-size, the peak saved heap memory for u16 is (u8 + u16) -> 4
        // bytes * u16::MAX vs (u8 + u32) -> 8 bytes * u16::MAX, the saved heap memory is at peak
        // ~262KB.
        if is_using_u32_as_idx_type_helpful && len <= (u32::MAX as usize) {
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
    #[cfg_attr(not(test), rustc_diagnostic_item = "slice_into_vec")]
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
                    ptr::copy_nonoverlapping::<T>(
                        buf.as_ptr(),
                        (buf.as_mut_ptr()).add(buf.len()),
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
                ptr::copy_nonoverlapping::<T>(
                    buf.as_ptr(),
                    (buf.as_mut_ptr()).add(buf.len()),
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
    #[deprecated(since = "1.3.0", note = "renamed to join", suggestion = "join")]
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
    sort::stable::sort::<T, F, Vec<T>>(v, &mut is_less);
}

#[cfg(not(no_global_oom_handling))]
#[unstable(issue = "none", feature = "std_internals")]
impl<T> sort::stable::BufGuard<T> for Vec<T> {
    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
    }

    fn as_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
        self.spare_capacity_mut()
    }
}
