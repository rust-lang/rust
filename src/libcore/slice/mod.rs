// Copyright 2012-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Slice management and manipulation
//!
//! For more details see [`std::slice`].
//!
//! [`std::slice`]: ../../std/slice/index.html

#![stable(feature = "rust1", since = "1.0.0")]

// How this module is organized.
//
// The library infrastructure for slices is fairly messy. There's
// a lot of stuff defined here. Let's keep it clean.
//
// The layout of this file is thus:
//
// * Inherent methods. This is where most of the slice API resides.
// * Implementations of a few common traits with important slice ops.
// * Definitions of a bunch of iterators.
// * Free functions.
// * The `raw` and `bytes` submodules.
// * Boilerplate trait implementations.

use cmp::Ordering::{self, Less, Equal, Greater};
use cmp;
use fmt;
use intrinsics::assume;
use isize;
use iter::*;
use ops::{FnMut, Try, self};
use option::Option;
use option::Option::{None, Some};
use result::Result;
use result::Result::{Ok, Err};
use ptr;
use mem;
use marker::{Copy, Send, Sync, Sized, self};
use iter_private::TrustedRandomAccess;

#[unstable(feature = "slice_internals", issue = "0",
           reason = "exposed from core to be reused in std; use the memchr crate")]
/// Pure rust memchr implementation, taken from rust-memchr
pub mod memchr;

mod rotate;
mod sort;

#[repr(C)]
union Repr<'a, T: 'a> {
    rust: &'a [T],
    rust_mut: &'a mut [T],
    raw: FatPtr<T>,
}

#[repr(C)]
struct FatPtr<T> {
    data: *const T,
    len: usize,
}

//
// Extension traits
//

#[lang = "slice"]
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
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    #[rustc_const_unstable(feature = "const_slice_len")]
    pub const fn len(&self) -> usize {
        unsafe {
            Repr { rust: self }.raw.len
        }
    }

    /// Returns `true` if the slice has a length of 0.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert!(!a.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    #[rustc_const_unstable(feature = "const_slice_len")]
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
    #[inline]
    pub fn first(&self) -> Option<&T> {
        self.get(0)
    }

    /// Returns a mutable pointer to the first element of the slice, or `None` if it is empty.
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
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn first_mut(&mut self) -> Option<&mut T> {
        self.get_mut(0)
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
    #[inline]
    pub fn split_first(&self) -> Option<(&T, &[T])> {
        if self.is_empty() { None } else { Some((&self[0], &self[1..])) }
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
    #[inline]
    pub fn split_first_mut(&mut self) -> Option<(&mut T, &mut [T])> {
        if self.is_empty() { None } else {
            let split = self.split_at_mut(1);
            Some((&mut split.0[0], split.1))
        }
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
    #[inline]
    pub fn split_last(&self) -> Option<(&T, &[T])> {
        let len = self.len();
        if len == 0 { None } else { Some((&self[len - 1], &self[..(len - 1)])) }
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
    #[inline]
    pub fn split_last_mut(&mut self) -> Option<(&mut T, &mut [T])> {
        let len = self.len();
        if len == 0 { None } else {
            let split = self.split_at_mut(len - 1);
            Some((&mut split.1[0], split.0))
        }

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
    #[inline]
    pub fn last(&self) -> Option<&T> {
        let last_idx = self.len().checked_sub(1)?;
        self.get(last_idx)
    }

    /// Returns a mutable pointer to the last item in the slice.
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
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        let last_idx = self.len().checked_sub(1)?;
        self.get_mut(last_idx)
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
    pub fn get<I>(&self, index: I) -> Option<&I::Output>
        where I: SliceIndex<Self>
    {
        index.get(self)
    }

    /// Returns a mutable reference to an element or subslice depending on the
    /// type of index (see [`get`]) or `None` if the index is out of bounds.
    ///
    /// [`get`]: #method.get
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
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut I::Output>
        where I: SliceIndex<Self>
    {
        index.get_mut(self)
    }

    /// Returns a reference to an element or subslice, without doing bounds
    /// checking.
    ///
    /// This is generally not recommended, use with caution! For a safe
    /// alternative see [`get`].
    ///
    /// [`get`]: #method.get
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
    pub unsafe fn get_unchecked<I>(&self, index: I) -> &I::Output
        where I: SliceIndex<Self>
    {
        index.get_unchecked(self)
    }

    /// Returns a mutable reference to an element or subslice, without doing
    /// bounds checking.
    ///
    /// This is generally not recommended, use with caution! For a safe
    /// alternative see [`get_mut`].
    ///
    /// [`get_mut`]: #method.get_mut
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
    pub unsafe fn get_unchecked_mut<I>(&mut self, index: I) -> &mut I::Output
        where I: SliceIndex<Self>
    {
        index.get_unchecked_mut(self)
    }

    /// Returns a raw pointer to the slice's buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this
    /// function returns, or else it will end up pointing to garbage.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self as *const [T] as *const T
    }

    /// Returns an unsafe mutable pointer to the slice's buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this
    /// function returns, or else it will end up pointing to garbage.
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
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self as *mut [T] as *mut T
    }

    /// Swaps two elements in the slice.
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
    /// let mut v = ["a", "b", "c", "d"];
    /// v.swap(1, 3);
    /// assert!(v == ["a", "d", "c", "b"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn swap(&mut self, a: usize, b: usize) {
        unsafe {
            // Can't take two mutable loans from one vector, so instead just cast
            // them to their raw pointers to do the swap
            let pa: *mut T = &mut self[a];
            let pb: *mut T = &mut self[b];
            ptr::swap(pa, pb);
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
        let mut i: usize = 0;
        let ln = self.len();

        // For very small types, all the individual reads in the normal
        // path perform poorly.  We can do better, given efficient unaligned
        // load/store, by loading a larger chunk and reversing a register.

        // Ideally LLVM would do this for us, as it knows better than we do
        // whether unaligned reads are efficient (since that changes between
        // different ARM versions, for example) and what the best chunk size
        // would be.  Unfortunately, as of LLVM 4.0 (2017-05) it only unrolls
        // the loop, so we need to do this ourselves.  (Hypothesis: reverse
        // is troublesome because the sides can be aligned differently --
        // will be, when the length is odd -- so there's no way of emitting
        // pre- and postludes to use fully-aligned SIMD in the middle.)

        let fast_unaligned =
            cfg!(any(target_arch = "x86", target_arch = "x86_64"));

        if fast_unaligned && mem::size_of::<T>() == 1 {
            // Use the llvm.bswap intrinsic to reverse u8s in a usize
            let chunk = mem::size_of::<usize>();
            while i + chunk - 1 < ln / 2 {
                unsafe {
                    let pa: *mut T = self.get_unchecked_mut(i);
                    let pb: *mut T = self.get_unchecked_mut(ln - i - chunk);
                    let va = ptr::read_unaligned(pa as *mut usize);
                    let vb = ptr::read_unaligned(pb as *mut usize);
                    ptr::write_unaligned(pa as *mut usize, vb.swap_bytes());
                    ptr::write_unaligned(pb as *mut usize, va.swap_bytes());
                }
                i += chunk;
            }
        }

        if fast_unaligned && mem::size_of::<T>() == 2 {
            // Use rotate-by-16 to reverse u16s in a u32
            let chunk = mem::size_of::<u32>() / 2;
            while i + chunk - 1 < ln / 2 {
                unsafe {
                    let pa: *mut T = self.get_unchecked_mut(i);
                    let pb: *mut T = self.get_unchecked_mut(ln - i - chunk);
                    let va = ptr::read_unaligned(pa as *mut u32);
                    let vb = ptr::read_unaligned(pb as *mut u32);
                    ptr::write_unaligned(pa as *mut u32, vb.rotate_left(16));
                    ptr::write_unaligned(pb as *mut u32, va.rotate_left(16));
                }
                i += chunk;
            }
        }

        while i < ln / 2 {
            // Unsafe swap to avoid the bounds check in safe swap.
            unsafe {
                let pa: *mut T = self.get_unchecked_mut(i);
                let pb: *mut T = self.get_unchecked_mut(ln - i - 1);
                ptr::swap(pa, pb);
            }
            i += 1;
        }
    }

    /// Returns an iterator over the slice.
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
    pub fn iter(&self) -> Iter<T> {
        unsafe {
            let ptr = self.as_ptr();
            assume(!ptr.is_null());

            let end = if mem::size_of::<T>() == 0 {
                (ptr as *const u8).wrapping_add(self.len()) as *const T
            } else {
                ptr.add(self.len())
            };

            Iter {
                ptr,
                end,
                _marker: marker::PhantomData
            }
        }
    }

    /// Returns an iterator that allows modifying each value.
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
    pub fn iter_mut(&mut self) -> IterMut<T> {
        unsafe {
            let ptr = self.as_mut_ptr();
            assume(!ptr.is_null());

            let end = if mem::size_of::<T>() == 0 {
                (ptr as *mut u8).wrapping_add(self.len()) as *mut T
            } else {
                ptr.add(self.len())
            };

            IterMut {
                ptr,
                end,
                _marker: marker::PhantomData
            }
        }
    }

    /// Returns an iterator over all contiguous windows of length
    /// `size`. The windows overlap. If the slice is shorter than
    /// `size`, the iterator returns no values.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// let slice = ['r', 'u', 's', 't'];
    /// let mut iter = slice.windows(2);
    /// assert_eq!(iter.next().unwrap(), &['r', 'u']);
    /// assert_eq!(iter.next().unwrap(), &['u', 's']);
    /// assert_eq!(iter.next().unwrap(), &['s', 't']);
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
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn windows(&self, size: usize) -> Windows<T> {
        assert!(size != 0);
        Windows { v: self, size }
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time, starting at the
    /// beginning of the slice.
    ///
    /// The chunks are slices and do not overlap. If `chunk_size` does not divide the length of the
    /// slice, then the last chunk will not have length `chunk_size`.
    ///
    /// See [`chunks_exact`] for a variant of this iterator that returns chunks of always exactly
    /// `chunk_size` elements, and [`rchunks`] for the same iterator but starting at the end of the
    /// slice of the slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
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
    /// [`chunks_exact`]: #method.chunks_exact
    /// [`rchunks`]: #method.rchunks
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn chunks(&self, chunk_size: usize) -> Chunks<T> {
        assert!(chunk_size != 0);
        Chunks { v: self, chunk_size }
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time, starting at the
    /// beginning of the slice.
    ///
    /// The chunks are mutable slices, and do not overlap. If `chunk_size` does not divide the
    /// length of the slice, then the last chunk will not have length `chunk_size`.
    ///
    /// See [`chunks_exact_mut`] for a variant of this iterator that returns chunks of always
    /// exactly `chunk_size` elements, and [`rchunks_mut`] for the same iterator but starting at
    /// the end of the slice of the slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
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
    /// [`chunks_exact_mut`]: #method.chunks_exact_mut
    /// [`rchunks_mut`]: #method.rchunks_mut
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<T> {
        assert!(chunk_size != 0);
        ChunksMut { v: self, chunk_size }
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
    /// chunk, and [`rchunks_exact`] for the same iterator but starting at the end of the slice of
    /// the slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
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
    /// [`chunks`]: #method.chunks
    /// [`rchunks_exact`]: #method.rchunks_exact
    #[stable(feature = "chunks_exact", since = "1.31.0")]
    #[inline]
    pub fn chunks_exact(&self, chunk_size: usize) -> ChunksExact<T> {
        assert!(chunk_size != 0);
        let rem = self.len() % chunk_size;
        let len = self.len() - rem;
        let (fst, snd) = self.split_at(len);
        ChunksExact { v: fst, rem: snd, chunk_size }
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
    /// the slice of the slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
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
    /// [`chunks_mut`]: #method.chunks_mut
    /// [`rchunks_exact_mut`]: #method.rchunks_exact_mut
    #[stable(feature = "chunks_exact", since = "1.31.0")]
    #[inline]
    pub fn chunks_exact_mut(&mut self, chunk_size: usize) -> ChunksExactMut<T> {
        assert!(chunk_size != 0);
        let rem = self.len() % chunk_size;
        let len = self.len() - rem;
        let (fst, snd) = self.split_at_mut(len);
        ChunksExactMut { v: fst, rem: snd, chunk_size }
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
    /// Panics if `chunk_size` is 0.
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
    /// [`rchunks_exact`]: #method.rchunks_exact
    /// [`chunks`]: #method.chunks
    #[stable(feature = "rchunks", since = "1.31.0")]
    #[inline]
    pub fn rchunks(&self, chunk_size: usize) -> RChunks<T> {
        assert!(chunk_size != 0);
        RChunks { v: self, chunk_size }
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
    /// Panics if `chunk_size` is 0.
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
    /// [`rchunks_exact_mut`]: #method.rchunks_exact_mut
    /// [`chunks_mut`]: #method.chunks_mut
    #[stable(feature = "rchunks", since = "1.31.0")]
    #[inline]
    pub fn rchunks_mut(&mut self, chunk_size: usize) -> RChunksMut<T> {
        assert!(chunk_size != 0);
        RChunksMut { v: self, chunk_size }
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
    /// See [`rchunks`] for a variant of this iterator that also returns the remainder as a smaller
    /// chunk, and [`chunks_exact`] for the same iterator but starting at the beginning of the
    /// slice of the slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
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
    /// [`rchunks`]: #method.rchunks
    /// [`chunks_exact`]: #method.chunks_exact
    #[stable(feature = "rchunks", since = "1.31.0")]
    #[inline]
    pub fn rchunks_exact(&self, chunk_size: usize) -> RChunksExact<T> {
        assert!(chunk_size != 0);
        let rem = self.len() % chunk_size;
        let (fst, snd) = self.split_at(rem);
        RChunksExact { v: snd, rem: fst, chunk_size }
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
    /// of the slice of the slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
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
    /// [`rchunks_mut`]: #method.rchunks_mut
    /// [`chunks_exact_mut`]: #method.chunks_exact_mut
    #[stable(feature = "rchunks", since = "1.31.0")]
    #[inline]
    pub fn rchunks_exact_mut(&mut self, chunk_size: usize) -> RChunksExactMut<T> {
        assert!(chunk_size != 0);
        let rem = self.len() % chunk_size;
        let (fst, snd) = self.split_at_mut(rem);
        RChunksExactMut { v: snd, rem: fst, chunk_size }
    }

    /// Divides one slice into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [1, 2, 3, 4, 5, 6];
    ///
    /// {
    ///    let (left, right) = v.split_at(0);
    ///    assert!(left == []);
    ///    assert!(right == [1, 2, 3, 4, 5, 6]);
    /// }
    ///
    /// {
    ///     let (left, right) = v.split_at(2);
    ///     assert!(left == [1, 2]);
    ///     assert!(right == [3, 4, 5, 6]);
    /// }
    ///
    /// {
    ///     let (left, right) = v.split_at(6);
    ///     assert!(left == [1, 2, 3, 4, 5, 6]);
    ///     assert!(right == []);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn split_at(&self, mid: usize) -> (&[T], &[T]) {
        (&self[..mid], &self[mid..])
    }

    /// Divides one mutable slice into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [1, 0, 3, 0, 5, 6];
    /// // scoped to restrict the lifetime of the borrows
    /// {
    ///     let (left, right) = v.split_at_mut(2);
    ///     assert!(left == [1, 0]);
    ///     assert!(right == [3, 0, 5, 6]);
    ///     left[1] = 2;
    ///     right[1] = 4;
    /// }
    /// assert!(v == [1, 2, 3, 4, 5, 6]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn split_at_mut(&mut self, mid: usize) -> (&mut [T], &mut [T]) {
        let len = self.len();
        let ptr = self.as_mut_ptr();

        unsafe {
            assert!(mid <= len);

            (from_raw_parts_mut(ptr, mid),
             from_raw_parts_mut(ptr.add(mid), len - mid))
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
    pub fn split<F>(&self, pred: F) -> Split<T, F>
        where F: FnMut(&T) -> bool
    {
        Split {
            v: self,
            pred,
            finished: false
        }
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
    pub fn split_mut<F>(&mut self, pred: F) -> SplitMut<T, F>
        where F: FnMut(&T) -> bool
    {
        SplitMut { v: self, pred, finished: false }
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
    pub fn rsplit<F>(&self, pred: F) -> RSplit<T, F>
        where F: FnMut(&T) -> bool
    {
        RSplit { inner: self.split(pred) }
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
    pub fn rsplit_mut<F>(&mut self, pred: F) -> RSplitMut<T, F>
        where F: FnMut(&T) -> bool
    {
        RSplitMut { inner: self.split_mut(pred) }
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
    /// Print the slice split once by numbers divisible by 3 (i.e. `[10, 40]`,
    /// `[20, 60, 50]`):
    ///
    /// ```
    /// let v = [10, 40, 30, 20, 60, 50];
    ///
    /// for group in v.splitn(2, |num| *num % 3 == 0) {
    ///     println!("{:?}", group);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn splitn<F>(&self, n: usize, pred: F) -> SplitN<T, F>
        where F: FnMut(&T) -> bool
    {
        SplitN {
            inner: GenericSplitN {
                iter: self.split(pred),
                count: n
            }
        }
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
    pub fn splitn_mut<F>(&mut self, n: usize, pred: F) -> SplitNMut<T, F>
        where F: FnMut(&T) -> bool
    {
        SplitNMut {
            inner: GenericSplitN {
                iter: self.split_mut(pred),
                count: n
            }
        }
    }

    /// Returns an iterator over subslices separated by elements that match
    /// `pred` limited to returning at most `n` items. This starts at the end of
    /// the slice and works backwards.  The matched element is not contained in
    /// the subslices.
    ///
    /// The last element returned, if any, will contain the remainder of the
    /// slice.
    ///
    /// # Examples
    ///
    /// Print the slice split once, starting from the end, by numbers divisible
    /// by 3 (i.e. `[50]`, `[10, 40, 30, 20]`):
    ///
    /// ```
    /// let v = [10, 40, 30, 20, 60, 50];
    ///
    /// for group in v.rsplitn(2, |num| *num % 3 == 0) {
    ///     println!("{:?}", group);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn rsplitn<F>(&self, n: usize, pred: F) -> RSplitN<T, F>
        where F: FnMut(&T) -> bool
    {
        RSplitN {
            inner: GenericSplitN {
                iter: self.rsplit(pred),
                count: n
            }
        }
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
    pub fn rsplitn_mut<F>(&mut self, n: usize, pred: F) -> RSplitNMut<T, F>
        where F: FnMut(&T) -> bool
    {
        RSplitNMut {
            inner: GenericSplitN {
                iter: self.rsplit_mut(pred),
                count: n
            }
        }
    }

    /// Returns `true` if the slice contains an element with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert!(v.contains(&30));
    /// assert!(!v.contains(&50));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn contains(&self, x: &T) -> bool
        where T: PartialEq
    {
        x.slice_contains(self)
    }

    /// Returns `true` if `needle` is a prefix of the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert!(v.starts_with(&[10]));
    /// assert!(v.starts_with(&[10, 40]));
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
    pub fn starts_with(&self, needle: &[T]) -> bool
        where T: PartialEq
    {
        let n = needle.len();
        self.len() >= n && needle == &self[..n]
    }

    /// Returns `true` if `needle` is a suffix of the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = [10, 40, 30];
    /// assert!(v.ends_with(&[30]));
    /// assert!(v.ends_with(&[40, 30]));
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
    pub fn ends_with(&self, needle: &[T]) -> bool
        where T: PartialEq
    {
        let (m, n) = (self.len(), needle.len());
        m >= n && needle == &self[m-n..]
    }

    /// Binary searches this sorted slice for a given element.
    ///
    /// If the value is found then [`Result::Ok`] is returned, containing the
    /// index of the matching element. If there are multiple matches, then any
    /// one of the matches could be returned. If the value is not found then
    /// [`Result::Err`] is returned, containing the index where a matching
    /// element could be inserted while maintaining sorted order.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn binary_search(&self, x: &T) -> Result<usize, usize>
        where T: Ord
    {
        self.binary_search_by(|p| p.cmp(x))
    }

    /// Binary searches this sorted slice with a comparator function.
    ///
    /// The comparator function should implement an order consistent
    /// with the sort order of the underlying slice, returning an
    /// order code that indicates whether its argument is `Less`,
    /// `Equal` or `Greater` the desired target.
    ///
    /// If the value is found then [`Result::Ok`] is returned, containing the
    /// index of the matching element. If there are multiple matches, then any
    /// one of the matches could be returned. If the value is not found then
    /// [`Result::Err`] is returned, containing the index where a matching
    /// element could be inserted while maintaining sorted order.
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
        where F: FnMut(&'a T) -> Ordering
    {
        let s = self;
        let mut size = s.len();
        if size == 0 {
            return Err(0);
        }
        let mut base = 0usize;
        while size > 1 {
            let half = size / 2;
            let mid = base + half;
            // mid is always in [0, size), that means mid is >= 0 and < size.
            // mid >= 0: by definition
            // mid < size: mid = size / 2 + size / 4 + size / 8 ...
            let cmp = f(unsafe { s.get_unchecked(mid) });
            base = if cmp == Greater { base } else { mid };
            size -= half;
        }
        // base is always in [0, size) because base <= mid.
        let cmp = f(unsafe { s.get_unchecked(base) });
        if cmp == Equal { Ok(base) } else { Err(base + (cmp == Less) as usize) }

    }

    /// Binary searches this sorted slice with a key extraction function.
    ///
    /// Assumes that the slice is sorted by the key, for instance with
    /// [`sort_by_key`] using the same key extraction function.
    ///
    /// If the value is found then [`Result::Ok`] is returned, containing the
    /// index of the matching element. If there are multiple matches, then any
    /// one of the matches could be returned. If the value is not found then
    /// [`Result::Err`] is returned, containing the index where a matching
    /// element could be inserted while maintaining sorted order.
    ///
    /// [`sort_by_key`]: #method.sort_by_key
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
    /// assert_eq!(s.binary_search_by_key(&13, |&(a,b)| b),  Ok(9));
    /// assert_eq!(s.binary_search_by_key(&4, |&(a,b)| b),   Err(7));
    /// assert_eq!(s.binary_search_by_key(&100, |&(a,b)| b), Err(13));
    /// let r = s.binary_search_by_key(&1, |&(a,b)| b);
    /// assert!(match r { Ok(1..=4) => true, _ => false, });
    /// ```
    #[stable(feature = "slice_binary_search_by_key", since = "1.10.0")]
    #[inline]
    pub fn binary_search_by_key<'a, B, F>(&'a self, b: &B, mut f: F) -> Result<usize, usize>
        where F: FnMut(&'a T) -> B,
              B: Ord
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }

    /// Sorts the slice, but may not preserve the order of equal elements.
    ///
    /// This sort is unstable (i.e. may reorder equal elements), in-place (i.e. does not allocate),
    /// and `O(n log n)` worst-case.
    ///
    /// # Current implementation
    ///
    /// The current algorithm is based on [pattern-defeating quicksort][pdqsort] by Orson Peters,
    /// which combines the fast average case of randomized quicksort with the fast worst case of
    /// heapsort, while achieving linear time on slices with certain patterns. It uses some
    /// randomization to avoid degenerate cases, but with a fixed seed to always provide
    /// deterministic behavior.
    ///
    /// It is typically faster than stable sorting, except in a few special cases, e.g. when the
    /// slice consists of several concatenated sorted sequences.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [-5, 4, 1, -3, 2];
    ///
    /// v.sort_unstable();
    /// assert!(v == [-5, -3, 1, 2, 4]);
    /// ```
    ///
    /// [pdqsort]: https://github.com/orlp/pdqsort
    #[stable(feature = "sort_unstable", since = "1.20.0")]
    #[inline]
    pub fn sort_unstable(&mut self)
        where T: Ord
    {
        sort::quicksort(self, |a, b| a.lt(b));
    }

    /// Sorts the slice with a comparator function, but may not preserve the order of equal
    /// elements.
    ///
    /// This sort is unstable (i.e. may reorder equal elements), in-place (i.e. does not allocate),
    /// and `O(n log n)` worst-case.
    ///
    /// The comparator function must define a total ordering for the elements in the slice. If
    /// the ordering is not total, the order of the elements is unspecified. An order is a
    /// total order if it is (for all a, b and c):
    ///
    /// * total and antisymmetric: exactly one of a < b, a == b or a > b is true; and
    /// * transitive, a < b and b < c implies a < c. The same must hold for both == and >.
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
    /// # Current implementation
    ///
    /// The current algorithm is based on [pattern-defeating quicksort][pdqsort] by Orson Peters,
    /// which combines the fast average case of randomized quicksort with the fast worst case of
    /// heapsort, while achieving linear time on slices with certain patterns. It uses some
    /// randomization to avoid degenerate cases, but with a fixed seed to always provide
    /// deterministic behavior.
    ///
    /// It is typically faster than stable sorting, except in a few special cases, e.g. when the
    /// slice consists of several concatenated sorted sequences.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [5, 4, 1, 3, 2];
    /// v.sort_unstable_by(|a, b| a.cmp(b));
    /// assert!(v == [1, 2, 3, 4, 5]);
    ///
    /// // reverse sorting
    /// v.sort_unstable_by(|a, b| b.cmp(a));
    /// assert!(v == [5, 4, 3, 2, 1]);
    /// ```
    ///
    /// [pdqsort]: https://github.com/orlp/pdqsort
    #[stable(feature = "sort_unstable", since = "1.20.0")]
    #[inline]
    pub fn sort_unstable_by<F>(&mut self, mut compare: F)
        where F: FnMut(&T, &T) -> Ordering
    {
        sort::quicksort(self, |a, b| compare(a, b) == Ordering::Less);
    }

    /// Sorts the slice with a key extraction function, but may not preserve the order of equal
    /// elements.
    ///
    /// This sort is unstable (i.e. may reorder equal elements), in-place (i.e. does not allocate),
    /// and `O(m n log(m n))` worst-case, where the key function is `O(m)`.
    ///
    /// # Current implementation
    ///
    /// The current algorithm is based on [pattern-defeating quicksort][pdqsort] by Orson Peters,
    /// which combines the fast average case of randomized quicksort with the fast worst case of
    /// heapsort, while achieving linear time on slices with certain patterns. It uses some
    /// randomization to avoid degenerate cases, but with a fixed seed to always provide
    /// deterministic behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = [-5i32, 4, 1, -3, 2];
    ///
    /// v.sort_unstable_by_key(|k| k.abs());
    /// assert!(v == [1, 2, -3, 4, -5]);
    /// ```
    ///
    /// [pdqsort]: https://github.com/orlp/pdqsort
    #[stable(feature = "sort_unstable", since = "1.20.0")]
    #[inline]
    pub fn sort_unstable_by_key<K, F>(&mut self, mut f: F)
        where F: FnMut(&T) -> K, K: Ord
    {
        sort::quicksort(self, |a, b| f(a).lt(&f(b)));
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
        where T: PartialEq
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
        where F: FnMut(&mut T, &mut T) -> bool
    {
        // Although we have a mutable reference to `self`, we cannot make
        // *arbitrary* changes. The `same_bucket` calls could panic, so we
        // must ensure that the slice is in a valid state at all times.
        //
        // The way that we handle this is by using swaps; we iterate
        // over all the elements, swapping as we go so that at the end
        // the elements we wish to keep are in the front, and those we
        // wish to reject are at the back. We can then split the slice.
        // This operation is still O(n).
        //
        // Example: We start in this state, where `r` represents "next
        // read" and `w` represents "next_write`.
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
            return (self, &mut [])
        }

        let ptr = self.as_mut_ptr();
        let mut next_read: usize = 1;
        let mut next_write: usize = 1;

        unsafe {
            // Avoid bounds checks by using raw pointers.
            while next_read < len {
                let ptr_read = ptr.add(next_read);
                let prev_ptr_write = ptr.add(next_write - 1);
                if !same_bucket(&mut *ptr_read, &mut *prev_ptr_write) {
                    if next_read != next_write {
                        let ptr_write = prev_ptr_write.offset(1);
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
        where F: FnMut(&mut T) -> K,
              K: PartialEq,
    {
        self.partition_dedup_by(|a, b| key(a) == key(b))
    }

    /// Rotates the slice in-place such that the first `mid` elements of the
    /// slice move to the end while the last `self.len() - mid` elements move to
    /// the front. After calling `rotate_left`, the element previously at index
    /// `mid` will become the first element in the slice.
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

        unsafe {
            let p = self.as_mut_ptr();
            rotate::ptr_rotate(mid, p.add(mid), k);
        }
    }

    /// Rotates the slice in-place such that the first `self.len() - k`
    /// elements of the slice move to the end while the last `k` elements move
    /// to the front. After calling `rotate_right`, the element previously at
    /// index `self.len() - k` will become the first element in the slice.
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
    /// Rotate a subslice:
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

        unsafe {
            let p = self.as_mut_ptr();
            rotate::ptr_rotate(mid, p.add(mid), k);
        }
    }

    /// Copies the elements from `src` into `self`.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// If `src` implements `Copy`, it can be more performant to use
    /// [`copy_from_slice`].
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
    /// [`copy_from_slice`]: #method.copy_from_slice
    /// [`split_at_mut`]: #method.split_at_mut
    #[stable(feature = "clone_from_slice", since = "1.7.0")]
    pub fn clone_from_slice(&mut self, src: &[T]) where T: Clone {
        assert!(self.len() == src.len(),
                "destination and source slices have different lengths");
        // NOTE: We need to explicitly slice them to the same length
        // for bounds checking to be elided, and the optimizer will
        // generate memcpy for simple cases (for example T = u8).
        let len = self.len();
        let src = &src[..len];
        for i in 0..len {
            self[i].clone_from(&src[i]);
        }

    }

    /// Copies all elements from `src` into `self`, using a memcpy.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// If `src` does not implement `Copy`, use [`clone_from_slice`].
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
    /// [`clone_from_slice`]: #method.clone_from_slice
    /// [`split_at_mut`]: #method.split_at_mut
    #[stable(feature = "copy_from_slice", since = "1.9.0")]
    pub fn copy_from_slice(&mut self, src: &[T]) where T: Copy {
        assert_eq!(self.len(), src.len(),
                   "destination and source slices have different lengths");
        unsafe {
            ptr::copy_nonoverlapping(
                src.as_ptr(), self.as_mut_ptr(), self.len());
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
    /// # #![feature(copy_within)]
    /// let mut bytes = *b"Hello, World!";
    ///
    /// bytes.copy_within(1..5, 8);
    ///
    /// assert_eq!(&bytes, b"Hello, Wello!");
    /// ```
    #[unstable(feature = "copy_within", issue = "54236")]
    pub fn copy_within<R: ops::RangeBounds<usize>>(&mut self, src: R, dest: usize)
    where
        T: Copy,
    {
        let src_start = match src.start_bound() {
            ops::Bound::Included(&n) => n,
            ops::Bound::Excluded(&n) => n
                .checked_add(1)
                .unwrap_or_else(|| slice_index_overflow_fail()),
            ops::Bound::Unbounded => 0,
        };
        let src_end = match src.end_bound() {
            ops::Bound::Included(&n) => n
                .checked_add(1)
                .unwrap_or_else(|| slice_index_overflow_fail()),
            ops::Bound::Excluded(&n) => n,
            ops::Bound::Unbounded => self.len(),
        };
        assert!(src_start <= src_end, "src end is before src start");
        assert!(src_end <= self.len(), "src is out of bounds");
        let count = src_end - src_start;
        assert!(dest <= self.len() - count, "dest is out of bounds");
        unsafe {
            ptr::copy(
                self.get_unchecked(src_start),
                self.get_unchecked_mut(dest),
                count,
            );
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
    /// [`split_at_mut`]: #method.split_at_mut
    #[stable(feature = "swap_with_slice", since = "1.27.0")]
    pub fn swap_with_slice(&mut self, other: &mut [T]) {
        assert!(self.len() == other.len(),
                "destination and source slices have different lengths");
        unsafe {
            ptr::swap_nonoverlapping(
                self.as_mut_ptr(), other.as_mut_ptr(), self.len());
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
        #[inline]
        fn gcd(a: usize, b: usize) -> usize {
            // iterative stein’s algorithm
            // We should still make this `const fn` (and revert to recursive algorithm if we do)
            // because relying on llvm to consteval all this is… well, it makes me uncomfortable.
            let (ctz_a, mut ctz_b) = unsafe {
                if a == 0 { return b; }
                if b == 0 { return a; }
                (::intrinsics::cttz_nonzero(a), ::intrinsics::cttz_nonzero(b))
            };
            let k = ctz_a.min(ctz_b);
            let mut a = a >> ctz_a;
            let mut b = b;
            loop {
                // remove all factors of 2 from b
                b >>= ctz_b;
                if a > b {
                    ::mem::swap(&mut a, &mut b);
                }
                b = b - a;
                unsafe {
                    if b == 0 {
                        break;
                    }
                    ctz_b = ::intrinsics::cttz_nonzero(b);
                }
            }
            a << k
        }
        let gcd: usize = gcd(::mem::size_of::<T>(), ::mem::size_of::<U>());
        let ts: usize = ::mem::size_of::<U>() / gcd;
        let us: usize = ::mem::size_of::<T>() / gcd;

        // Armed with this knowledge, we can find how many `U`s we can fit!
        let us_len = self.len() / ts * us;
        // And how many `T`s will be in the trailing slice!
        let ts_len = self.len() % ts;
        (us_len, ts_len)
    }

    /// Transmute the slice to a slice of another type, ensuring alignment of the types is
    /// maintained.
    ///
    /// This method splits the slice into three distinct slices: prefix, correctly aligned middle
    /// slice of a new type, and the suffix slice. The method does a best effort to make the
    /// middle slice the greatest length possible for a given type and input slice, but only
    /// your algorithm's performance should depend on that, not its correctness.
    ///
    /// This method has no purpose when either input element `T` or output element `U` are
    /// zero-sized and will return the original slice without splitting anything.
    ///
    /// # Unsafety
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
    pub unsafe fn align_to<U>(&self) -> (&[T], &[U], &[T]) {
        // Note that most of this function will be constant-evaluated,
        if ::mem::size_of::<U>() == 0 || ::mem::size_of::<T>() == 0 {
            // handle ZSTs specially, which is – don't handle them at all.
            return (self, &[], &[]);
        }

        // First, find at what point do we split between the first and 2nd slice. Easy with
        // ptr.align_offset.
        let ptr = self.as_ptr();
        let offset = ::ptr::align_offset(ptr, ::mem::align_of::<U>());
        if offset > self.len() {
            (self, &[], &[])
        } else {
            let (left, rest) = self.split_at(offset);
            // now `rest` is definitely aligned, so `from_raw_parts_mut` below is okay
            let (us_len, ts_len) = rest.align_to_offsets::<U>();
            (left,
             from_raw_parts(rest.as_ptr() as *const U, us_len),
             from_raw_parts(rest.as_ptr().add(rest.len() - ts_len), ts_len))
        }
    }

    /// Transmute the slice to a slice of another type, ensuring alignment of the types is
    /// maintained.
    ///
    /// This method splits the slice into three distinct slices: prefix, correctly aligned middle
    /// slice of a new type, and the suffix slice. The method does a best effort to make the
    /// middle slice the greatest length possible for a given type and input slice, but only
    /// your algorithm's performance should depend on that, not its correctness.
    ///
    /// This method has no purpose when either input element `T` or output element `U` are
    /// zero-sized and will return the original slice without splitting anything.
    ///
    /// # Unsafety
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
    pub unsafe fn align_to_mut<U>(&mut self) -> (&mut [T], &mut [U], &mut [T]) {
        // Note that most of this function will be constant-evaluated,
        if ::mem::size_of::<U>() == 0 || ::mem::size_of::<T>() == 0 {
            // handle ZSTs specially, which is – don't handle them at all.
            return (self, &mut [], &mut []);
        }

        // First, find at what point do we split between the first and 2nd slice. Easy with
        // ptr.align_offset.
        let ptr = self.as_ptr();
        let offset = ::ptr::align_offset(ptr, ::mem::align_of::<U>());
        if offset > self.len() {
            (self, &mut [], &mut [])
        } else {
            let (left, rest) = self.split_at_mut(offset);
            // now `rest` is definitely aligned, so `from_raw_parts_mut` below is okay
            let (us_len, ts_len) = rest.align_to_offsets::<U>();
            let mut_ptr = rest.as_mut_ptr();
            (left,
             from_raw_parts_mut(mut_ptr as *mut U, us_len),
             from_raw_parts_mut(mut_ptr.add(rest.len() - ts_len), ts_len))
        }
    }
}

#[lang = "slice_u8"]
#[cfg(not(test))]
impl [u8] {
    /// Checks if all bytes in this slice are within the ASCII range.
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn is_ascii(&self) -> bool {
        self.iter().all(|b| b.is_ascii())
    }

    /// Checks that two slices are an ASCII case-insensitive match.
    ///
    /// Same as `to_ascii_lowercase(a) == to_ascii_lowercase(b)`,
    /// but without allocating and copying temporaries.
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn eq_ignore_ascii_case(&self, other: &[u8]) -> bool {
        self.len() == other.len() &&
            self.iter().zip(other).all(|(a, b)| {
                a.eq_ignore_ascii_case(b)
            })
    }

    /// Converts this slice to its ASCII upper case equivalent in-place.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new uppercased value without modifying the existing one, use
    /// [`to_ascii_uppercase`].
    ///
    /// [`to_ascii_uppercase`]: #method.to_ascii_uppercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn make_ascii_uppercase(&mut self) {
        for byte in self {
            byte.make_ascii_uppercase();
        }
    }

    /// Converts this slice to its ASCII lower case equivalent in-place.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new lowercased value without modifying the existing one, use
    /// [`to_ascii_lowercase`].
    ///
    /// [`to_ascii_lowercase`]: #method.to_ascii_lowercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn make_ascii_lowercase(&mut self) {
        for byte in self {
            byte.make_ascii_lowercase();
        }
    }

}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "slice indices are of type `usize` or ranges of `usize`"]
impl<T, I> ops::Index<I> for [T]
    where I: SliceIndex<[T]>
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "slice indices are of type `usize` or ranges of `usize`"]
impl<T, I> ops::IndexMut<I> for [T]
    where I: SliceIndex<[T]>
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

#[inline(never)]
#[cold]
fn slice_index_len_fail(index: usize, len: usize) -> ! {
    panic!("index {} out of range for slice of length {}", index, len);
}

#[inline(never)]
#[cold]
fn slice_index_order_fail(index: usize, end: usize) -> ! {
    panic!("slice index starts at {} but ends at {}", index, end);
}

#[inline(never)]
#[cold]
fn slice_index_overflow_fail() -> ! {
    panic!("attempted to index slice up to maximum usize");
}

mod private_slice_index {
    use super::ops;
    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    pub trait Sealed {}

    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    impl Sealed for usize {}
    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    impl Sealed for ops::Range<usize> {}
    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    impl Sealed for ops::RangeTo<usize> {}
    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    impl Sealed for ops::RangeFrom<usize> {}
    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    impl Sealed for ops::RangeFull {}
    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    impl Sealed for ops::RangeInclusive<usize> {}
    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    impl Sealed for ops::RangeToInclusive<usize> {}
}

/// A helper trait used for indexing operations.
#[stable(feature = "slice_get_slice", since = "1.28.0")]
#[rustc_on_unimplemented = "slice indices are of type `usize` or ranges of `usize`"]
pub trait SliceIndex<T: ?Sized>: private_slice_index::Sealed {
    /// The output type returned by methods.
    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    type Output: ?Sized;

    /// Returns a shared reference to the output at this location, if in
    /// bounds.
    #[unstable(feature = "slice_index_methods", issue = "0")]
    fn get(self, slice: &T) -> Option<&Self::Output>;

    /// Returns a mutable reference to the output at this location, if in
    /// bounds.
    #[unstable(feature = "slice_index_methods", issue = "0")]
    fn get_mut(self, slice: &mut T) -> Option<&mut Self::Output>;

    /// Returns a shared reference to the output at this location, without
    /// performing any bounds checking.
    #[unstable(feature = "slice_index_methods", issue = "0")]
    unsafe fn get_unchecked(self, slice: &T) -> &Self::Output;

    /// Returns a mutable reference to the output at this location, without
    /// performing any bounds checking.
    #[unstable(feature = "slice_index_methods", issue = "0")]
    unsafe fn get_unchecked_mut(self, slice: &mut T) -> &mut Self::Output;

    /// Returns a shared reference to the output at this location, panicking
    /// if out of bounds.
    #[unstable(feature = "slice_index_methods", issue = "0")]
    fn index(self, slice: &T) -> &Self::Output;

    /// Returns a mutable reference to the output at this location, panicking
    /// if out of bounds.
    #[unstable(feature = "slice_index_methods", issue = "0")]
    fn index_mut(self, slice: &mut T) -> &mut Self::Output;
}

#[stable(feature = "slice_get_slice_impls", since = "1.15.0")]
impl<T> SliceIndex<[T]> for usize {
    type Output = T;

    #[inline]
    fn get(self, slice: &[T]) -> Option<&T> {
        if self < slice.len() {
            unsafe {
                Some(self.get_unchecked(slice))
            }
        } else {
            None
        }
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut T> {
        if self < slice.len() {
            unsafe {
                Some(self.get_unchecked_mut(slice))
            }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &[T]) -> &T {
        &*slice.as_ptr().add(self)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut T {
        &mut *slice.as_mut_ptr().add(self)
    }

    #[inline]
    fn index(self, slice: &[T]) -> &T {
        // NB: use intrinsic indexing
        &(*slice)[self]
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut T {
        // NB: use intrinsic indexing
        &mut (*slice)[self]
    }
}

#[stable(feature = "slice_get_slice_impls", since = "1.15.0")]
impl<T> SliceIndex<[T]> for  ops::Range<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        if self.start > self.end || self.end > slice.len() {
            None
        } else {
            unsafe {
                Some(self.get_unchecked(slice))
            }
        }
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        if self.start > self.end || self.end > slice.len() {
            None
        } else {
            unsafe {
                Some(self.get_unchecked_mut(slice))
            }
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &[T]) -> &[T] {
        from_raw_parts(slice.as_ptr().add(self.start), self.end - self.start)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut [T] {
        from_raw_parts_mut(slice.as_mut_ptr().add(self.start), self.end - self.start)
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        if self.start > self.end {
            slice_index_order_fail(self.start, self.end);
        } else if self.end > slice.len() {
            slice_index_len_fail(self.end, slice.len());
        }
        unsafe {
            self.get_unchecked(slice)
        }
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        if self.start > self.end {
            slice_index_order_fail(self.start, self.end);
        } else if self.end > slice.len() {
            slice_index_len_fail(self.end, slice.len());
        }
        unsafe {
            self.get_unchecked_mut(slice)
        }
    }
}

#[stable(feature = "slice_get_slice_impls", since = "1.15.0")]
impl<T> SliceIndex<[T]> for ops::RangeTo<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        (0..self.end).get(slice)
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        (0..self.end).get_mut(slice)
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &[T]) -> &[T] {
        (0..self.end).get_unchecked(slice)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut [T] {
        (0..self.end).get_unchecked_mut(slice)
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        (0..self.end).index(slice)
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        (0..self.end).index_mut(slice)
    }
}

#[stable(feature = "slice_get_slice_impls", since = "1.15.0")]
impl<T> SliceIndex<[T]> for ops::RangeFrom<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        (self.start..slice.len()).get(slice)
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        (self.start..slice.len()).get_mut(slice)
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &[T]) -> &[T] {
        (self.start..slice.len()).get_unchecked(slice)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut [T] {
        (self.start..slice.len()).get_unchecked_mut(slice)
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        (self.start..slice.len()).index(slice)
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        (self.start..slice.len()).index_mut(slice)
    }
}

#[stable(feature = "slice_get_slice_impls", since = "1.15.0")]
impl<T> SliceIndex<[T]> for ops::RangeFull {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        Some(slice)
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        Some(slice)
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &[T]) -> &[T] {
        slice
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut [T] {
        slice
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        slice
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        slice
    }
}


#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<T> SliceIndex<[T]> for ops::RangeInclusive<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        if *self.end() == usize::max_value() { None }
        else { (*self.start()..self.end() + 1).get(slice) }
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        if *self.end() == usize::max_value() { None }
        else { (*self.start()..self.end() + 1).get_mut(slice) }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &[T]) -> &[T] {
        (*self.start()..self.end() + 1).get_unchecked(slice)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut [T] {
        (*self.start()..self.end() + 1).get_unchecked_mut(slice)
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        if *self.end() == usize::max_value() { slice_index_overflow_fail(); }
        (*self.start()..self.end() + 1).index(slice)
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        if *self.end() == usize::max_value() { slice_index_overflow_fail(); }
        (*self.start()..self.end() + 1).index_mut(slice)
    }
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<T> SliceIndex<[T]> for ops::RangeToInclusive<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        (0..=self.end).get(slice)
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        (0..=self.end).get_mut(slice)
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &[T]) -> &[T] {
        (0..=self.end).get_unchecked(slice)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut [T] {
        (0..=self.end).get_unchecked_mut(slice)
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        (0..=self.end).index(slice)
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        (0..=self.end).index_mut(slice)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Common traits
////////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for &[T] {
    /// Creates an empty slice.
    fn default() -> Self { &[] }
}

#[stable(feature = "mut_slice_default", since = "1.5.0")]
impl<T> Default for &mut [T] {
    /// Creates a mutable empty slice.
    fn default() -> Self { &mut [] }
}

//
// Iterators
//

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a [T] {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a mut [T] {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

// Macro helper functions
#[inline(always)]
fn size_from_ptr<T>(_: *const T) -> usize {
    mem::size_of::<T>()
}

// Inlining is_empty and len makes a huge performance difference
macro_rules! is_empty {
    // The way we encode the length of a ZST iterator, this works both for ZST
    // and non-ZST.
    ($self: ident) => {$self.ptr == $self.end}
}
// To get rid of some bounds checks (see `position`), we compute the length in a somewhat
// unexpected way. (Tested by `codegen/slice-position-bounds-check`.)
macro_rules! len {
    ($self: ident) => {{
        let start = $self.ptr;
        let diff = ($self.end as usize).wrapping_sub(start as usize);
        let size = size_from_ptr(start);
        if size == 0 {
            diff
        } else {
            // Using division instead of `offset_from` helps LLVM remove bounds checks
            diff / size
        }
    }}
}

// The shared definition of the `Iter` and `IterMut` iterators
macro_rules! iterator {
    (struct $name:ident -> $ptr:ty, $elem:ty, $raw_mut:tt, $( $mut_:tt )*) => {
        impl<'a, T> $name<'a, T> {
            // Helper function for creating a slice from the iterator.
            #[inline(always)]
            fn make_slice(&self) -> &'a [T] {
                unsafe { from_raw_parts(self.ptr, len!(self)) }
            }

            // Helper function for moving the start of the iterator forwards by `offset` elements,
            // returning the old start.
            // Unsafe because the offset must be in-bounds or one-past-the-end.
            #[inline(always)]
            unsafe fn post_inc_start(&mut self, offset: isize) -> * $raw_mut T {
                if mem::size_of::<T>() == 0 {
                    // This is *reducing* the length.  `ptr` never changes with ZST.
                    self.end = (self.end as * $raw_mut u8).wrapping_offset(-offset) as * $raw_mut T;
                    self.ptr
                } else {
                    let old = self.ptr;
                    self.ptr = self.ptr.offset(offset);
                    old
                }
            }

            // Helper function for moving the end of the iterator backwards by `offset` elements,
            // returning the new end.
            // Unsafe because the offset must be in-bounds or one-past-the-end.
            #[inline(always)]
            unsafe fn pre_dec_end(&mut self, offset: isize) -> * $raw_mut T {
                if mem::size_of::<T>() == 0 {
                    self.end = (self.end as * $raw_mut u8).wrapping_offset(-offset) as * $raw_mut T;
                    self.ptr
                } else {
                    self.end = self.end.offset(-offset);
                    self.end
                }
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, T> ExactSizeIterator for $name<'a, T> {
            #[inline(always)]
            fn len(&self) -> usize {
                len!(self)
            }

            #[inline(always)]
            fn is_empty(&self) -> bool {
                is_empty!(self)
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, T> Iterator for $name<'a, T> {
            type Item = $elem;

            #[inline]
            fn next(&mut self) -> Option<$elem> {
                // could be implemented with slices, but this avoids bounds checks
                unsafe {
                    assume(!self.ptr.is_null());
                    if mem::size_of::<T>() != 0 {
                        assume(!self.end.is_null());
                    }
                    if is_empty!(self) {
                        None
                    } else {
                        Some(& $( $mut_ )* *self.post_inc_start(1))
                    }
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let exact = len!(self);
                (exact, Some(exact))
            }

            #[inline]
            fn count(self) -> usize {
                len!(self)
            }

            #[inline]
            fn nth(&mut self, n: usize) -> Option<$elem> {
                if n >= len!(self) {
                    // This iterator is now empty.
                    if mem::size_of::<T>() == 0 {
                        // We have to do it this way as `ptr` may never be 0, but `end`
                        // could be (due to wrapping).
                        self.end = self.ptr;
                    } else {
                        self.ptr = self.end;
                    }
                    return None;
                }
                // We are in bounds. `offset` does the right thing even for ZSTs.
                unsafe {
                    let elem = Some(& $( $mut_ )* *self.ptr.add(n));
                    self.post_inc_start((n as isize).wrapping_add(1));
                    elem
                }
            }

            #[inline]
            fn last(mut self) -> Option<$elem> {
                self.next_back()
            }

            #[inline]
            fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R where
                Self: Sized, F: FnMut(B, Self::Item) -> R, R: Try<Ok=B>
            {
                // manual unrolling is needed when there are conditional exits from the loop
                let mut accum = init;
                unsafe {
                    while len!(self) >= 4 {
                        accum = f(accum, & $( $mut_ )* *self.post_inc_start(1))?;
                        accum = f(accum, & $( $mut_ )* *self.post_inc_start(1))?;
                        accum = f(accum, & $( $mut_ )* *self.post_inc_start(1))?;
                        accum = f(accum, & $( $mut_ )* *self.post_inc_start(1))?;
                    }
                    while !is_empty!(self) {
                        accum = f(accum, & $( $mut_ )* *self.post_inc_start(1))?;
                    }
                }
                Try::from_ok(accum)
            }

            #[inline]
            fn fold<Acc, Fold>(mut self, init: Acc, mut f: Fold) -> Acc
                where Fold: FnMut(Acc, Self::Item) -> Acc,
            {
                // Let LLVM unroll this, rather than using the default
                // impl that would force the manual unrolling above
                let mut accum = init;
                while let Some(x) = self.next() {
                    accum = f(accum, x);
                }
                accum
            }

            #[inline]
            #[rustc_inherit_overflow_checks]
            fn position<P>(&mut self, mut predicate: P) -> Option<usize> where
                Self: Sized,
                P: FnMut(Self::Item) -> bool,
            {
                // The addition might panic on overflow.
                let n = len!(self);
                self.try_fold(0, move |i, x| {
                    if predicate(x) { Err(i) }
                    else { Ok(i + 1) }
                }).err()
                    .map(|i| {
                        unsafe { assume(i < n) };
                        i
                    })
            }

            #[inline]
            fn rposition<P>(&mut self, mut predicate: P) -> Option<usize> where
                P: FnMut(Self::Item) -> bool,
                Self: Sized + ExactSizeIterator + DoubleEndedIterator
            {
                // No need for an overflow check here, because `ExactSizeIterator`
                let n = len!(self);
                self.try_rfold(n, move |i, x| {
                    let i = i - 1;
                    if predicate(x) { Err(i) }
                    else { Ok(i) }
                }).err()
                    .map(|i| {
                        unsafe { assume(i < n) };
                        i
                    })
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, T> DoubleEndedIterator for $name<'a, T> {
            #[inline]
            fn next_back(&mut self) -> Option<$elem> {
                // could be implemented with slices, but this avoids bounds checks
                unsafe {
                    assume(!self.ptr.is_null());
                    if mem::size_of::<T>() != 0 {
                        assume(!self.end.is_null());
                    }
                    if is_empty!(self) {
                        None
                    } else {
                        Some(& $( $mut_ )* *self.pre_dec_end(1))
                    }
                }
            }

            #[inline]
            fn try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R where
                Self: Sized, F: FnMut(B, Self::Item) -> R, R: Try<Ok=B>
            {
                // manual unrolling is needed when there are conditional exits from the loop
                let mut accum = init;
                unsafe {
                    while len!(self) >= 4 {
                        accum = f(accum, & $( $mut_ )* *self.pre_dec_end(1))?;
                        accum = f(accum, & $( $mut_ )* *self.pre_dec_end(1))?;
                        accum = f(accum, & $( $mut_ )* *self.pre_dec_end(1))?;
                        accum = f(accum, & $( $mut_ )* *self.pre_dec_end(1))?;
                    }
                    // inlining is_empty everywhere makes a huge performance difference
                    while !is_empty!(self) {
                        accum = f(accum, & $( $mut_ )* *self.pre_dec_end(1))?;
                    }
                }
                Try::from_ok(accum)
            }

            #[inline]
            fn rfold<Acc, Fold>(mut self, init: Acc, mut f: Fold) -> Acc
                where Fold: FnMut(Acc, Self::Item) -> Acc,
            {
                // Let LLVM unroll this, rather than using the default
                // impl that would force the manual unrolling above
                let mut accum = init;
                while let Some(x) = self.next_back() {
                    accum = f(accum, x);
                }
                accum
            }
        }

        #[stable(feature = "fused", since = "1.26.0")]
        impl<'a, T> FusedIterator for $name<'a, T> {}

        #[unstable(feature = "trusted_len", issue = "37572")]
        unsafe impl<'a, T> TrustedLen for $name<'a, T> {}
    }
}

/// Immutable slice iterator
///
/// This struct is created by the [`iter`] method on [slices].
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// // First, we declare a type which has `iter` method to get the `Iter` struct (&[usize here]):
/// let slice = &[1, 2, 3];
///
/// // Then, we iterate over it:
/// for element in slice.iter() {
///     println!("{}", element);
/// }
/// ```
///
/// [`iter`]: ../../std/primitive.slice.html#method.iter
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    ptr: *const T,
    end: *const T, // If T is a ZST, this is actually ptr+len.  This encoding is picked so that
                   // ptr == end is a quick test for the Iterator being empty, that works
                   // for both ZST and non-ZST.
    _marker: marker::PhantomData<&'a T>,
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Iter")
            .field(&self.as_slice())
            .finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Sync> Sync for Iter<'_, T> {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Sync> Send for Iter<'_, T> {}

impl<'a, T> Iter<'a, T> {
    /// View the underlying data as a subslice of the original data.
    ///
    /// This has the same lifetime as the original slice, and so the
    /// iterator can continue to be used while this exists.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // First, we declare a type which has the `iter` method to get the `Iter`
    /// // struct (&[usize here]):
    /// let slice = &[1, 2, 3];
    ///
    /// // Then, we get the iterator:
    /// let mut iter = slice.iter();
    /// // So if we print what `as_slice` method returns here, we have "[1, 2, 3]":
    /// println!("{:?}", iter.as_slice());
    ///
    /// // Next, we move to the second element of the slice:
    /// iter.next();
    /// // Now `as_slice` returns "[2, 3]":
    /// println!("{:?}", iter.as_slice());
    /// ```
    #[stable(feature = "iter_to_slice", since = "1.4.0")]
    pub fn as_slice(&self) -> &'a [T] {
        self.make_slice()
    }
}

iterator!{struct Iter -> *const T, &'a T, const, /* no mut */}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self { Iter { ptr: self.ptr, end: self.end, _marker: self._marker } }
}

#[stable(feature = "slice_iter_as_ref", since = "1.13.0")]
impl<T> AsRef<[T]> for Iter<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

/// Mutable slice iterator.
///
/// This struct is created by the [`iter_mut`] method on [slices].
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// // First, we declare a type which has `iter_mut` method to get the `IterMut`
/// // struct (&[usize here]):
/// let mut slice = &mut [1, 2, 3];
///
/// // Then, we iterate over it and increment each element value:
/// for element in slice.iter_mut() {
///     *element += 1;
/// }
///
/// // We now have "[2, 3, 4]":
/// println!("{:?}", slice);
/// ```
///
/// [`iter_mut`]: ../../std/primitive.slice.html#method.iter_mut
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, T: 'a> {
    ptr: *mut T,
    end: *mut T, // If T is a ZST, this is actually ptr+len.  This encoding is picked so that
                 // ptr == end is a quick test for the Iterator being empty, that works
                 // for both ZST and non-ZST.
    _marker: marker::PhantomData<&'a mut T>,
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug> fmt::Debug for IterMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("IterMut")
            .field(&self.make_slice())
            .finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Sync> Sync for IterMut<'_, T> {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Send> Send for IterMut<'_, T> {}

impl<'a, T> IterMut<'a, T> {
    /// View the underlying data as a subslice of the original data.
    ///
    /// To avoid creating `&mut` references that alias, this is forced
    /// to consume the iterator.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // First, we declare a type which has `iter_mut` method to get the `IterMut`
    /// // struct (&[usize here]):
    /// let mut slice = &mut [1, 2, 3];
    ///
    /// {
    ///     // Then, we get the iterator:
    ///     let mut iter = slice.iter_mut();
    ///     // We move to next element:
    ///     iter.next();
    ///     // So if we print what `into_slice` method returns here, we have "[2, 3]":
    ///     println!("{:?}", iter.into_slice());
    /// }
    ///
    /// // Now let's modify a value of the slice:
    /// {
    ///     // First we get back the iterator:
    ///     let mut iter = slice.iter_mut();
    ///     // We change the value of the first element of the slice returned by the `next` method:
    ///     *iter.next().unwrap() += 1;
    /// }
    /// // Now slice is "[2, 2, 3]":
    /// println!("{:?}", slice);
    /// ```
    #[stable(feature = "iter_to_slice", since = "1.4.0")]
    pub fn into_slice(self) -> &'a mut [T] {
        unsafe { from_raw_parts_mut(self.ptr, len!(self)) }
    }
}

iterator!{struct IterMut -> *mut T, &'a mut T, mut, mut}

/// An internal abstraction over the splitting iterators, so that
/// splitn, splitn_mut etc can be implemented once.
#[doc(hidden)]
trait SplitIter: DoubleEndedIterator {
    /// Marks the underlying iterator as complete, extracting the remaining
    /// portion of the slice.
    fn finish(&mut self) -> Option<Self::Item>;
}

/// An iterator over subslices separated by elements that match a predicate
/// function.
///
/// This struct is created by the [`split`] method on [slices].
///
/// [`split`]: ../../std/primitive.slice.html#method.split
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Split<'a, T:'a, P> where P: FnMut(&T) -> bool {
    v: &'a [T],
    pred: P,
    finished: bool
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for Split<'_, T, P> where P: FnMut(&T) -> bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Split")
            .field("v", &self.v)
            .field("finished", &self.finished)
            .finish()
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<T, P> Clone for Split<'_, T, P> where P: Clone + FnMut(&T) -> bool {
    fn clone(&self) -> Self {
        Split {
            v: self.v,
            pred: self.pred.clone(),
            finished: self.finished,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> Iterator for Split<'a, T, P> where P: FnMut(&T) -> bool {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.finished { return None; }

        match self.v.iter().position(|x| (self.pred)(x)) {
            None => self.finish(),
            Some(idx) => {
                let ret = Some(&self.v[..idx]);
                self.v = &self.v[idx + 1..];
                ret
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            (1, Some(self.v.len() + 1))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> DoubleEndedIterator for Split<'a, T, P> where P: FnMut(&T) -> bool {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.finished { return None; }

        match self.v.iter().rposition(|x| (self.pred)(x)) {
            None => self.finish(),
            Some(idx) => {
                let ret = Some(&self.v[idx + 1..]);
                self.v = &self.v[..idx];
                ret
            }
        }
    }
}

impl<'a, T, P> SplitIter for Split<'a, T, P> where P: FnMut(&T) -> bool {
    #[inline]
    fn finish(&mut self) -> Option<&'a [T]> {
        if self.finished { None } else { self.finished = true; Some(self.v) }
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, P> FusedIterator for Split<'_, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over the subslices of the vector which are separated
/// by elements that match `pred`.
///
/// This struct is created by the [`split_mut`] method on [slices].
///
/// [`split_mut`]: ../../std/primitive.slice.html#method.split_mut
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SplitMut<'a, T:'a, P> where P: FnMut(&T) -> bool {
    v: &'a mut [T],
    pred: P,
    finished: bool
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for SplitMut<'_, T, P> where P: FnMut(&T) -> bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SplitMut")
            .field("v", &self.v)
            .field("finished", &self.finished)
            .finish()
    }
}

impl<'a, T, P> SplitIter for SplitMut<'a, T, P> where P: FnMut(&T) -> bool {
    #[inline]
    fn finish(&mut self) -> Option<&'a mut [T]> {
        if self.finished {
            None
        } else {
            self.finished = true;
            Some(mem::replace(&mut self.v, &mut []))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> Iterator for SplitMut<'a, T, P> where P: FnMut(&T) -> bool {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.finished { return None; }

        let idx_opt = { // work around borrowck limitations
            let pred = &mut self.pred;
            self.v.iter().position(|x| (*pred)(x))
        };
        match idx_opt {
            None => self.finish(),
            Some(idx) => {
                let tmp = mem::replace(&mut self.v, &mut []);
                let (head, tail) = tmp.split_at_mut(idx);
                self.v = &mut tail[1..];
                Some(head)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            // if the predicate doesn't match anything, we yield one slice
            // if it matches every element, we yield len+1 empty slices.
            (1, Some(self.v.len() + 1))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> DoubleEndedIterator for SplitMut<'a, T, P> where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.finished { return None; }

        let idx_opt = { // work around borrowck limitations
            let pred = &mut self.pred;
            self.v.iter().rposition(|x| (*pred)(x))
        };
        match idx_opt {
            None => self.finish(),
            Some(idx) => {
                let tmp = mem::replace(&mut self.v, &mut []);
                let (head, tail) = tmp.split_at_mut(idx);
                self.v = head;
                Some(&mut tail[1..])
            }
        }
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, P> FusedIterator for SplitMut<'_, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over subslices separated by elements that match a predicate
/// function, starting from the end of the slice.
///
/// This struct is created by the [`rsplit`] method on [slices].
///
/// [`rsplit`]: ../../std/primitive.slice.html#method.rsplit
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "slice_rsplit", since = "1.27.0")]
#[derive(Clone)] // Is this correct, or does it incorrectly require `T: Clone`?
pub struct RSplit<'a, T:'a, P> where P: FnMut(&T) -> bool {
    inner: Split<'a, T, P>
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<T: fmt::Debug, P> fmt::Debug for RSplit<'_, T, P> where P: FnMut(&T) -> bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("RSplit")
            .field("v", &self.inner.v)
            .field("finished", &self.inner.finished)
            .finish()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> Iterator for RSplit<'a, T, P> where P: FnMut(&T) -> bool {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        self.inner.next_back()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> DoubleEndedIterator for RSplit<'a, T, P> where P: FnMut(&T) -> bool {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        self.inner.next()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> SplitIter for RSplit<'a, T, P> where P: FnMut(&T) -> bool {
    #[inline]
    fn finish(&mut self) -> Option<&'a [T]> {
        self.inner.finish()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<T, P> FusedIterator for RSplit<'_, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over the subslices of the vector which are separated
/// by elements that match `pred`, starting from the end of the slice.
///
/// This struct is created by the [`rsplit_mut`] method on [slices].
///
/// [`rsplit_mut`]: ../../std/primitive.slice.html#method.rsplit_mut
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "slice_rsplit", since = "1.27.0")]
pub struct RSplitMut<'a, T:'a, P> where P: FnMut(&T) -> bool {
    inner: SplitMut<'a, T, P>
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<T: fmt::Debug, P> fmt::Debug for RSplitMut<'_, T, P> where P: FnMut(&T) -> bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("RSplitMut")
            .field("v", &self.inner.v)
            .field("finished", &self.inner.finished)
            .finish()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> SplitIter for RSplitMut<'a, T, P> where P: FnMut(&T) -> bool {
    #[inline]
    fn finish(&mut self) -> Option<&'a mut [T]> {
        self.inner.finish()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> Iterator for RSplitMut<'a, T, P> where P: FnMut(&T) -> bool {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        self.inner.next_back()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> DoubleEndedIterator for RSplitMut<'a, T, P> where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        self.inner.next()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<T, P> FusedIterator for RSplitMut<'_, T, P> where P: FnMut(&T) -> bool {}

/// An private iterator over subslices separated by elements that
/// match a predicate function, splitting at most a fixed number of
/// times.
#[derive(Debug)]
struct GenericSplitN<I> {
    iter: I,
    count: usize,
}

impl<T, I: SplitIter<Item=T>> Iterator for GenericSplitN<I> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        match self.count {
            0 => None,
            1 => { self.count -= 1; self.iter.finish() }
            _ => { self.count -= 1; self.iter.next() }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper_opt) = self.iter.size_hint();
        (lower, upper_opt.map(|upper| cmp::min(self.count, upper)))
    }
}

/// An iterator over subslices separated by elements that match a predicate
/// function, limited to a given number of splits.
///
/// This struct is created by the [`splitn`] method on [slices].
///
/// [`splitn`]: ../../std/primitive.slice.html#method.splitn
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SplitN<'a, T: 'a, P> where P: FnMut(&T) -> bool {
    inner: GenericSplitN<Split<'a, T, P>>
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for SplitN<'_, T, P> where P: FnMut(&T) -> bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SplitN")
            .field("inner", &self.inner)
            .finish()
    }
}

/// An iterator over subslices separated by elements that match a
/// predicate function, limited to a given number of splits, starting
/// from the end of the slice.
///
/// This struct is created by the [`rsplitn`] method on [slices].
///
/// [`rsplitn`]: ../../std/primitive.slice.html#method.rsplitn
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RSplitN<'a, T: 'a, P> where P: FnMut(&T) -> bool {
    inner: GenericSplitN<RSplit<'a, T, P>>
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for RSplitN<'_, T, P> where P: FnMut(&T) -> bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("RSplitN")
            .field("inner", &self.inner)
            .finish()
    }
}

/// An iterator over subslices separated by elements that match a predicate
/// function, limited to a given number of splits.
///
/// This struct is created by the [`splitn_mut`] method on [slices].
///
/// [`splitn_mut`]: ../../std/primitive.slice.html#method.splitn_mut
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SplitNMut<'a, T: 'a, P> where P: FnMut(&T) -> bool {
    inner: GenericSplitN<SplitMut<'a, T, P>>
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for SplitNMut<'_, T, P> where P: FnMut(&T) -> bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SplitNMut")
            .field("inner", &self.inner)
            .finish()
    }
}

/// An iterator over subslices separated by elements that match a
/// predicate function, limited to a given number of splits, starting
/// from the end of the slice.
///
/// This struct is created by the [`rsplitn_mut`] method on [slices].
///
/// [`rsplitn_mut`]: ../../std/primitive.slice.html#method.rsplitn_mut
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RSplitNMut<'a, T: 'a, P> where P: FnMut(&T) -> bool {
    inner: GenericSplitN<RSplitMut<'a, T, P>>
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for RSplitNMut<'_, T, P> where P: FnMut(&T) -> bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("RSplitNMut")
            .field("inner", &self.inner)
            .finish()
    }
}

macro_rules! forward_iterator {
    ($name:ident: $elem:ident, $iter_of:ty) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, $elem, P> Iterator for $name<'a, $elem, P> where
            P: FnMut(&T) -> bool
        {
            type Item = $iter_of;

            #[inline]
            fn next(&mut self) -> Option<$iter_of> {
                self.inner.next()
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.inner.size_hint()
            }
        }

        #[stable(feature = "fused", since = "1.26.0")]
        impl<'a, $elem, P> FusedIterator for $name<'a, $elem, P>
            where P: FnMut(&T) -> bool {}
    }
}

forward_iterator! { SplitN: T, &'a [T] }
forward_iterator! { RSplitN: T, &'a [T] }
forward_iterator! { SplitNMut: T, &'a mut [T] }
forward_iterator! { RSplitNMut: T, &'a mut [T] }

/// An iterator over overlapping subslices of length `size`.
///
/// This struct is created by the [`windows`] method on [slices].
///
/// [`windows`]: ../../std/primitive.slice.html#method.windows
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Windows<'a, T:'a> {
    v: &'a [T],
    size: usize
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Windows<'_, T> {
    fn clone(&self) -> Self {
        Windows {
            v: self.v,
            size: self.size,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Windows<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.size > self.v.len() {
            None
        } else {
            let ret = Some(&self.v[..self.size]);
            self.v = &self.v[1..];
            ret
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.size > self.v.len() {
            (0, Some(0))
        } else {
            let size = self.v.len() - self.size + 1;
            (size, Some(size))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (end, overflow) = self.size.overflowing_add(n);
        if end > self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            let nth = &self.v[n..end];
            self.v = &self.v[n+1..];
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.size > self.v.len() {
            None
        } else {
            let start = self.v.len() - self.size;
            Some(&self.v[start..])
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Windows<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.size > self.v.len() {
            None
        } else {
            let ret = Some(&self.v[self.v.len()-self.size..]);
            self.v = &self.v[..self.v.len()-1];
            ret
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for Windows<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for Windows<'_, T> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for Windows<'_, T> {}

#[doc(hidden)]
unsafe impl<'a, T> TrustedRandomAccess for Windows<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a [T] {
        from_raw_parts(self.v.as_ptr().add(i), self.size)
    }
    fn may_have_side_effect() -> bool { false }
}

/// An iterator over a slice in (non-overlapping) chunks (`chunk_size` elements at a
/// time), starting at the beginning of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last slice
/// of the iteration will be the remainder.
///
/// This struct is created by the [`chunks`] method on [slices].
///
/// [`chunks`]: ../../std/primitive.slice.html#method.chunks
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Chunks<'a, T:'a> {
    v: &'a [T],
    chunk_size: usize
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Chunks<'_, T> {
    fn clone(&self) -> Self {
        Chunks {
            v: self.v,
            chunk_size: self.chunk_size,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Chunks<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.v.is_empty() {
            None
        } else {
            let chunksz = cmp::min(self.v.len(), self.chunk_size);
            let (fst, snd) = self.v.split_at(chunksz);
            self.v = snd;
            Some(fst)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.v.is_empty() {
            (0, Some(0))
        } else {
            let n = self.v.len() / self.chunk_size;
            let rem = self.v.len() % self.chunk_size;
            let n = if rem > 0 { n+1 } else { n };
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (start, overflow) = n.overflowing_mul(self.chunk_size);
        if start >= self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            let end = match start.checked_add(self.chunk_size) {
                Some(sum) => cmp::min(self.v.len(), sum),
                None => self.v.len(),
            };
            let nth = &self.v[start..end];
            self.v = &self.v[end..];
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let start = (self.v.len() - 1) / self.chunk_size * self.chunk_size;
            Some(&self.v[start..])
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Chunks<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.v.is_empty() {
            None
        } else {
            let remainder = self.v.len() % self.chunk_size;
            let chunksz = if remainder != 0 { remainder } else { self.chunk_size };
            let (fst, snd) = self.v.split_at(self.v.len() - chunksz);
            self.v = fst;
            Some(snd)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for Chunks<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for Chunks<'_, T> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for Chunks<'_, T> {}

#[doc(hidden)]
unsafe impl<'a, T> TrustedRandomAccess for Chunks<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a [T] {
        let start = i * self.chunk_size;
        let end = match start.checked_add(self.chunk_size) {
            None => self.v.len(),
            Some(end) => cmp::min(end, self.v.len()),
        };
        from_raw_parts(self.v.as_ptr().add(start), end - start)
    }
    fn may_have_side_effect() -> bool { false }
}

/// An iterator over a slice in (non-overlapping) mutable chunks (`chunk_size`
/// elements at a time), starting at the beginning of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last slice
/// of the iteration will be the remainder.
///
/// This struct is created by the [`chunks_mut`] method on [slices].
///
/// [`chunks_mut`]: ../../std/primitive.slice.html#method.chunks_mut
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct ChunksMut<'a, T:'a> {
    v: &'a mut [T],
    chunk_size: usize
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for ChunksMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.v.is_empty() {
            None
        } else {
            let sz = cmp::min(self.v.len(), self.chunk_size);
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(sz);
            self.v = tail;
            Some(head)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.v.is_empty() {
            (0, Some(0))
        } else {
            let n = self.v.len() / self.chunk_size;
            let rem = self.v.len() % self.chunk_size;
            let n = if rem > 0 { n + 1 } else { n };
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<&'a mut [T]> {
        let (start, overflow) = n.overflowing_mul(self.chunk_size);
        if start >= self.v.len() || overflow {
            self.v = &mut [];
            None
        } else {
            let end = match start.checked_add(self.chunk_size) {
                Some(sum) => cmp::min(self.v.len(), sum),
                None => self.v.len(),
            };
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(end);
            let (_, nth) =  head.split_at_mut(start);
            self.v = tail;
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let start = (self.v.len() - 1) / self.chunk_size * self.chunk_size;
            Some(&mut self.v[start..])
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for ChunksMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.v.is_empty() {
            None
        } else {
            let remainder = self.v.len() % self.chunk_size;
            let sz = if remainder != 0 { remainder } else { self.chunk_size };
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (head, tail) = tmp.split_at_mut(tmp_len - sz);
            self.v = head;
            Some(tail)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for ChunksMut<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for ChunksMut<'_, T> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for ChunksMut<'_, T> {}

#[doc(hidden)]
unsafe impl<'a, T> TrustedRandomAccess for ChunksMut<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a mut [T] {
        let start = i * self.chunk_size;
        let end = match start.checked_add(self.chunk_size) {
            None => self.v.len(),
            Some(end) => cmp::min(end, self.v.len()),
        };
        from_raw_parts_mut(self.v.as_mut_ptr().add(start), end - start)
    }
    fn may_have_side_effect() -> bool { false }
}

/// An iterator over a slice in (non-overlapping) chunks (`chunk_size` elements at a
/// time), starting at the beginning of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last
/// up to `chunk_size-1` elements will be omitted but can be retrieved from
/// the [`remainder`] function from the iterator.
///
/// This struct is created by the [`chunks_exact`] method on [slices].
///
/// [`chunks_exact`]: ../../std/primitive.slice.html#method.chunks_exact
/// [`remainder`]: ../../std/slice/struct.ChunksExact.html#method.remainder
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "chunks_exact", since = "1.31.0")]
pub struct ChunksExact<'a, T:'a> {
    v: &'a [T],
    rem: &'a [T],
    chunk_size: usize
}

impl<'a, T> ChunksExact<'a, T> {
    /// Return the remainder of the original slice that is not going to be
    /// returned by the iterator. The returned slice has at most `chunk_size-1`
    /// elements.
    #[stable(feature = "chunks_exact", since = "1.31.0")]
    pub fn remainder(&self) -> &'a [T] {
        self.rem
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<T> Clone for ChunksExact<'_, T> {
    fn clone(&self) -> Self {
        ChunksExact {
            v: self.v,
            rem: self.rem,
            chunk_size: self.chunk_size,
        }
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<'a, T> Iterator for ChunksExact<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let (fst, snd) = self.v.split_at(self.chunk_size);
            self.v = snd;
            Some(fst)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.v.len() / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (start, overflow) = n.overflowing_mul(self.chunk_size);
        if start >= self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            let (_, snd) = self.v.split_at(start);
            self.v = snd;
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for ChunksExact<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let (fst, snd) = self.v.split_at(self.v.len() - self.chunk_size);
            self.v = fst;
            Some(snd)
        }
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<T> ExactSizeIterator for ChunksExact<'_, T> {
    fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for ChunksExact<'_, T> {}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<T> FusedIterator for ChunksExact<'_, T> {}

#[doc(hidden)]
#[stable(feature = "chunks_exact", since = "1.31.0")]
unsafe impl<'a, T> TrustedRandomAccess for ChunksExact<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a [T] {
        let start = i * self.chunk_size;
        from_raw_parts(self.v.as_ptr().add(start), self.chunk_size)
    }
    fn may_have_side_effect() -> bool { false }
}

/// An iterator over a slice in (non-overlapping) mutable chunks (`chunk_size`
/// elements at a time), starting at the beginning of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last up to
/// `chunk_size-1` elements will be omitted but can be retrieved from the
/// [`into_remainder`] function from the iterator.
///
/// This struct is created by the [`chunks_exact_mut`] method on [slices].
///
/// [`chunks_exact_mut`]: ../../std/primitive.slice.html#method.chunks_exact_mut
/// [`into_remainder`]: ../../std/slice/struct.ChunksExactMut.html#method.into_remainder
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "chunks_exact", since = "1.31.0")]
pub struct ChunksExactMut<'a, T:'a> {
    v: &'a mut [T],
    rem: &'a mut [T],
    chunk_size: usize
}

impl<'a, T> ChunksExactMut<'a, T> {
    /// Return the remainder of the original slice that is not going to be
    /// returned by the iterator. The returned slice has at most `chunk_size-1`
    /// elements.
    #[stable(feature = "chunks_exact", since = "1.31.0")]
    pub fn into_remainder(self) -> &'a mut [T] {
        self.rem
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<'a, T> Iterator for ChunksExactMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(self.chunk_size);
            self.v = tail;
            Some(head)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.v.len() / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<&'a mut [T]> {
        let (start, overflow) = n.overflowing_mul(self.chunk_size);
        if start >= self.v.len() || overflow {
            self.v = &mut [];
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let (_, snd) = tmp.split_at_mut(start);
            self.v = snd;
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for ChunksExactMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (head, tail) = tmp.split_at_mut(tmp_len - self.chunk_size);
            self.v = head;
            Some(tail)
        }
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<T> ExactSizeIterator for ChunksExactMut<'_, T> {
    fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for ChunksExactMut<'_, T> {}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<T> FusedIterator for ChunksExactMut<'_, T> {}

#[doc(hidden)]
#[stable(feature = "chunks_exact", since = "1.31.0")]
unsafe impl<'a, T> TrustedRandomAccess for ChunksExactMut<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a mut [T] {
        let start = i * self.chunk_size;
        from_raw_parts_mut(self.v.as_mut_ptr().add(start), self.chunk_size)
    }
    fn may_have_side_effect() -> bool { false }
}

/// An iterator over a slice in (non-overlapping) chunks (`chunk_size` elements at a
/// time), starting at the end of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last slice
/// of the iteration will be the remainder.
///
/// This struct is created by the [`rchunks`] method on [slices].
///
/// [`rchunks`]: ../../std/primitive.slice.html#method.rchunks
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rchunks", since = "1.31.0")]
pub struct RChunks<'a, T:'a> {
    v: &'a [T],
    chunk_size: usize
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> Clone for RChunks<'a, T> {
    fn clone(&self) -> RChunks<'a, T> {
        RChunks {
            v: self.v,
            chunk_size: self.chunk_size,
        }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> Iterator for RChunks<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.v.is_empty() {
            None
        } else {
            let chunksz = cmp::min(self.v.len(), self.chunk_size);
            let (fst, snd) = self.v.split_at(self.v.len() - chunksz);
            self.v = fst;
            Some(snd)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.v.is_empty() {
            (0, Some(0))
        } else {
            let n = self.v.len() / self.chunk_size;
            let rem = self.v.len() % self.chunk_size;
            let n = if rem > 0 { n+1 } else { n };
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (end, overflow) = n.overflowing_mul(self.chunk_size);
        if end >= self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            // Can't underflow because of the check above
            let end = self.v.len() - end;
            let start = match end.checked_sub(self.chunk_size) {
                Some(sum) => sum,
                None => 0,
            };
            let nth = &self.v[start..end];
            self.v = &self.v[0..start];
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let rem = self.v.len() % self.chunk_size;
            let end = if rem == 0 { self.chunk_size } else { rem };
            Some(&self.v[0..end])
        }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for RChunks<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.v.is_empty() {
            None
        } else {
            let remainder = self.v.len() % self.chunk_size;
            let chunksz = if remainder != 0 { remainder } else { self.chunk_size };
            let (fst, snd) = self.v.split_at(chunksz);
            self.v = snd;
            Some(fst)
        }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> ExactSizeIterator for RChunks<'a, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<'a, T> TrustedLen for RChunks<'a, T> {}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> FusedIterator for RChunks<'a, T> {}

#[doc(hidden)]
#[stable(feature = "rchunks", since = "1.31.0")]
unsafe impl<'a, T> TrustedRandomAccess for RChunks<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a [T] {
        let end = self.v.len() - i * self.chunk_size;
        let start = match end.checked_sub(self.chunk_size) {
            None => 0,
            Some(start) => start,
        };
        from_raw_parts(self.v.as_ptr().add(start), end - start)
    }
    fn may_have_side_effect() -> bool { false }
}

/// An iterator over a slice in (non-overlapping) mutable chunks (`chunk_size`
/// elements at a time), starting at the end of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last slice
/// of the iteration will be the remainder.
///
/// This struct is created by the [`rchunks_mut`] method on [slices].
///
/// [`rchunks_mut`]: ../../std/primitive.slice.html#method.rchunks_mut
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rchunks", since = "1.31.0")]
pub struct RChunksMut<'a, T:'a> {
    v: &'a mut [T],
    chunk_size: usize
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> Iterator for RChunksMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.v.is_empty() {
            None
        } else {
            let sz = cmp::min(self.v.len(), self.chunk_size);
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (head, tail) = tmp.split_at_mut(tmp_len - sz);
            self.v = head;
            Some(tail)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.v.is_empty() {
            (0, Some(0))
        } else {
            let n = self.v.len() / self.chunk_size;
            let rem = self.v.len() % self.chunk_size;
            let n = if rem > 0 { n + 1 } else { n };
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<&'a mut [T]> {
        let (end, overflow) = n.overflowing_mul(self.chunk_size);
        if end >= self.v.len() || overflow {
            self.v = &mut [];
            None
        } else {
            // Can't underflow because of the check above
            let end = self.v.len() - end;
            let start = match end.checked_sub(self.chunk_size) {
                Some(sum) => sum,
                None => 0,
            };
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(start);
            let (nth, _) = tail.split_at_mut(end - start);
            self.v = head;
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let rem = self.v.len() % self.chunk_size;
            let end = if rem == 0 { self.chunk_size } else { rem };
            Some(&mut self.v[0..end])
        }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for RChunksMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.v.is_empty() {
            None
        } else {
            let remainder = self.v.len() % self.chunk_size;
            let sz = if remainder != 0 { remainder } else { self.chunk_size };
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(sz);
            self.v = tail;
            Some(head)
        }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> ExactSizeIterator for RChunksMut<'a, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<'a, T> TrustedLen for RChunksMut<'a, T> {}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> FusedIterator for RChunksMut<'a, T> {}

#[doc(hidden)]
#[stable(feature = "rchunks", since = "1.31.0")]
unsafe impl<'a, T> TrustedRandomAccess for RChunksMut<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a mut [T] {
        let end = self.v.len() - i * self.chunk_size;
        let start = match end.checked_sub(self.chunk_size) {
            None => 0,
            Some(start) => start,
        };
        from_raw_parts_mut(self.v.as_mut_ptr().add(start), end - start)
    }
    fn may_have_side_effect() -> bool { false }
}

/// An iterator over a slice in (non-overlapping) chunks (`chunk_size` elements at a
/// time), starting at the end of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last
/// up to `chunk_size-1` elements will be omitted but can be retrieved from
/// the [`remainder`] function from the iterator.
///
/// This struct is created by the [`rchunks_exact`] method on [slices].
///
/// [`rchunks_exact`]: ../../std/primitive.slice.html#method.rchunks_exact
/// [`remainder`]: ../../std/slice/struct.ChunksExact.html#method.remainder
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rchunks", since = "1.31.0")]
pub struct RChunksExact<'a, T:'a> {
    v: &'a [T],
    rem: &'a [T],
    chunk_size: usize
}

impl<'a, T> RChunksExact<'a, T> {
    /// Return the remainder of the original slice that is not going to be
    /// returned by the iterator. The returned slice has at most `chunk_size-1`
    /// elements.
    #[stable(feature = "rchunks", since = "1.31.0")]
    pub fn remainder(&self) -> &'a [T] {
        self.rem
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> Clone for RChunksExact<'a, T> {
    fn clone(&self) -> RChunksExact<'a, T> {
        RChunksExact {
            v: self.v,
            rem: self.rem,
            chunk_size: self.chunk_size,
        }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> Iterator for RChunksExact<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let (fst, snd) = self.v.split_at(self.v.len() - self.chunk_size);
            self.v = fst;
            Some(snd)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.v.len() / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (end, overflow) = n.overflowing_mul(self.chunk_size);
        if end >= self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            let (fst, _) = self.v.split_at(self.v.len() - end);
            self.v = fst;
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for RChunksExact<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let (fst, snd) = self.v.split_at(self.chunk_size);
            self.v = snd;
            Some(fst)
        }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> ExactSizeIterator for RChunksExact<'a, T> {
    fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<'a, T> TrustedLen for RChunksExact<'a, T> {}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> FusedIterator for RChunksExact<'a, T> {}

#[doc(hidden)]
#[stable(feature = "rchunks", since = "1.31.0")]
unsafe impl<'a, T> TrustedRandomAccess for RChunksExact<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a [T] {
        let end = self.v.len() - i * self.chunk_size;
        let start = end - self.chunk_size;
        from_raw_parts(self.v.as_ptr().add(start), self.chunk_size)
    }
    fn may_have_side_effect() -> bool { false }
}

/// An iterator over a slice in (non-overlapping) mutable chunks (`chunk_size`
/// elements at a time), starting at the end of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last up to
/// `chunk_size-1` elements will be omitted but can be retrieved from the
/// [`into_remainder`] function from the iterator.
///
/// This struct is created by the [`rchunks_exact_mut`] method on [slices].
///
/// [`rchunks_exact_mut`]: ../../std/primitive.slice.html#method.rchunks_exact_mut
/// [`into_remainder`]: ../../std/slice/struct.ChunksExactMut.html#method.into_remainder
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rchunks", since = "1.31.0")]
pub struct RChunksExactMut<'a, T:'a> {
    v: &'a mut [T],
    rem: &'a mut [T],
    chunk_size: usize
}

impl<'a, T> RChunksExactMut<'a, T> {
    /// Return the remainder of the original slice that is not going to be
    /// returned by the iterator. The returned slice has at most `chunk_size-1`
    /// elements.
    #[stable(feature = "rchunks", since = "1.31.0")]
    pub fn into_remainder(self) -> &'a mut [T] {
        self.rem
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> Iterator for RChunksExactMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (head, tail) = tmp.split_at_mut(tmp_len - self.chunk_size);
            self.v = head;
            Some(tail)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.v.len() / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<&'a mut [T]> {
        let (end, overflow) = n.overflowing_mul(self.chunk_size);
        if end >= self.v.len() || overflow {
            self.v = &mut [];
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (fst, _) = tmp.split_at_mut(tmp_len - end);
            self.v = fst;
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for RChunksExactMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(self.chunk_size);
            self.v = tail;
            Some(head)
        }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> ExactSizeIterator for RChunksExactMut<'a, T> {
    fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<'a, T> TrustedLen for RChunksExactMut<'a, T> {}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> FusedIterator for RChunksExactMut<'a, T> {}

#[doc(hidden)]
#[stable(feature = "rchunks", since = "1.31.0")]
unsafe impl<'a, T> TrustedRandomAccess for RChunksExactMut<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a mut [T] {
        let end = self.v.len() - i * self.chunk_size;
        let start = end - self.chunk_size;
        from_raw_parts_mut(self.v.as_mut_ptr().add(start), self.chunk_size)
    }
    fn may_have_side_effect() -> bool { false }
}

//
// Free functions
//

/// Forms a slice from a pointer and a length.
///
/// The `len` argument is the number of **elements**, not the number of bytes.
///
/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `len` elements, nor whether the lifetime inferred is a suitable
/// lifetime for the returned slice.
///
/// `data` must be non-null and aligned, even for zero-length slices. One
/// reason for this is that enum layout optimizations may rely on references
/// (including slices of any length) being aligned and non-null to distinguish
/// them from other data. You can obtain a pointer that is usable as `data`
/// for zero-length slices using [`NonNull::dangling()`].
///
/// The total size of the slice must be no larger than `isize::MAX` **bytes**
/// in memory. See the safety documentation of [`pointer::offset`].
///
/// # Caveat
///
/// The lifetime for the returned slice is inferred from its usage. To
/// prevent accidental misuse, it's suggested to tie the lifetime to whichever
/// source lifetime is safe in the context, such as by providing a helper
/// function taking the lifetime of a host value for the slice, or by explicit
/// annotation.
///
/// # Examples
///
/// ```
/// use std::slice;
///
/// // manifest a slice for a single element
/// let x = 42;
/// let ptr = &x as *const _;
/// let slice = unsafe { slice::from_raw_parts(ptr, 1) };
/// assert_eq!(slice[0], 42);
/// ```
///
/// [`NonNull::dangling()`]: ../../std/ptr/struct.NonNull.html#method.dangling
/// [`pointer::offset`]: ../../std/primitive.pointer.html#method.offset
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn from_raw_parts<'a, T>(data: *const T, len: usize) -> &'a [T] {
    debug_assert!(data as usize % mem::align_of::<T>() == 0, "attempt to create unaligned slice");
    debug_assert!(mem::size_of::<T>().saturating_mul(len) <= isize::MAX as usize,
                  "attempt to create slice covering half the address space");
    Repr { raw: FatPtr { data, len } }.rust
}

/// Performs the same functionality as [`from_raw_parts`], except that a
/// mutable slice is returned.
///
/// This function is unsafe for the same reasons as [`from_raw_parts`], as well
/// as not being able to provide a non-aliasing guarantee of the returned
/// mutable slice. `data` must be non-null and aligned even for zero-length
/// slices as with [`from_raw_parts`]. The total size of the slice must be no
/// larger than `isize::MAX` **bytes** in memory.
///
/// See the documentation of [`from_raw_parts`] for more details.
///
/// [`from_raw_parts`]: ../../std/slice/fn.from_raw_parts.html
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn from_raw_parts_mut<'a, T>(data: *mut T, len: usize) -> &'a mut [T] {
    debug_assert!(data as usize % mem::align_of::<T>() == 0, "attempt to create unaligned slice");
    debug_assert!(mem::size_of::<T>().saturating_mul(len) <= isize::MAX as usize,
                  "attempt to create slice covering half the address space");
    Repr { raw: FatPtr { data, len } }.rust_mut
}

/// Converts a reference to T into a slice of length 1 (without copying).
#[stable(feature = "from_ref", since = "1.28.0")]
pub fn from_ref<T>(s: &T) -> &[T] {
    unsafe {
        from_raw_parts(s, 1)
    }
}

/// Converts a reference to T into a slice of length 1 (without copying).
#[stable(feature = "from_ref", since = "1.28.0")]
pub fn from_mut<T>(s: &mut T) -> &mut [T] {
    unsafe {
        from_raw_parts_mut(s, 1)
    }
}

// This function is public only because there is no other way to unit test heapsort.
#[unstable(feature = "sort_internals", reason = "internal to sort module", issue = "0")]
#[doc(hidden)]
pub fn heapsort<T, F>(v: &mut [T], mut is_less: F)
    where F: FnMut(&T, &T) -> bool
{
    sort::heapsort(v, &mut is_less);
}

//
// Comparison traits
//

extern {
    /// Calls implementation provided memcmp.
    ///
    /// Interprets the data as u8.
    ///
    /// Returns 0 for equal, < 0 for less than and > 0 for greater
    /// than.
    // FIXME(#32610): Return type should be c_int
    fn memcmp(s1: *const u8, s2: *const u8, n: usize) -> i32;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> PartialEq<[B]> for [A] where A: PartialEq<B> {
    fn eq(&self, other: &[B]) -> bool {
        SlicePartialEq::equal(self, other)
    }

    fn ne(&self, other: &[B]) -> bool {
        SlicePartialEq::not_equal(self, other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Eq> Eq for [T] {}

/// Implements comparison of vectors lexicographically.
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> Ord for [T] {
    fn cmp(&self, other: &[T]) -> Ordering {
        SliceOrd::compare(self, other)
    }
}

/// Implements comparison of vectors lexicographically.
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: PartialOrd> PartialOrd for [T] {
    fn partial_cmp(&self, other: &[T]) -> Option<Ordering> {
        SlicePartialOrd::partial_compare(self, other)
    }
}

#[doc(hidden)]
// intermediate trait for specialization of slice's PartialEq
trait SlicePartialEq<B> {
    fn equal(&self, other: &[B]) -> bool;

    fn not_equal(&self, other: &[B]) -> bool { !self.equal(other) }
}

// Generic slice equality
impl<A, B> SlicePartialEq<B> for [A]
    where A: PartialEq<B>
{
    default fn equal(&self, other: &[B]) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for i in 0..self.len() {
            if !self[i].eq(&other[i]) {
                return false;
            }
        }

        true
    }
}

// Use memcmp for bytewise equality when the types allow
impl<A> SlicePartialEq<A> for [A]
    where A: PartialEq<A> + BytewiseEquality
{
    fn equal(&self, other: &[A]) -> bool {
        if self.len() != other.len() {
            return false;
        }
        if self.as_ptr() == other.as_ptr() {
            return true;
        }
        unsafe {
            let size = mem::size_of_val(self);
            memcmp(self.as_ptr() as *const u8,
                   other.as_ptr() as *const u8, size) == 0
        }
    }
}

#[doc(hidden)]
// intermediate trait for specialization of slice's PartialOrd
trait SlicePartialOrd<B> {
    fn partial_compare(&self, other: &[B]) -> Option<Ordering>;
}

impl<A> SlicePartialOrd<A> for [A]
    where A: PartialOrd
{
    default fn partial_compare(&self, other: &[A]) -> Option<Ordering> {
        let l = cmp::min(self.len(), other.len());

        // Slice to the loop iteration range to enable bound check
        // elimination in the compiler
        let lhs = &self[..l];
        let rhs = &other[..l];

        for i in 0..l {
            match lhs[i].partial_cmp(&rhs[i]) {
                Some(Ordering::Equal) => (),
                non_eq => return non_eq,
            }
        }

        self.len().partial_cmp(&other.len())
    }
}

impl<A> SlicePartialOrd<A> for [A]
    where A: Ord
{
    default fn partial_compare(&self, other: &[A]) -> Option<Ordering> {
        Some(SliceOrd::compare(self, other))
    }
}

#[doc(hidden)]
// intermediate trait for specialization of slice's Ord
trait SliceOrd<B> {
    fn compare(&self, other: &[B]) -> Ordering;
}

impl<A> SliceOrd<A> for [A]
    where A: Ord
{
    default fn compare(&self, other: &[A]) -> Ordering {
        let l = cmp::min(self.len(), other.len());

        // Slice to the loop iteration range to enable bound check
        // elimination in the compiler
        let lhs = &self[..l];
        let rhs = &other[..l];

        for i in 0..l {
            match lhs[i].cmp(&rhs[i]) {
                Ordering::Equal => (),
                non_eq => return non_eq,
            }
        }

        self.len().cmp(&other.len())
    }
}

// memcmp compares a sequence of unsigned bytes lexicographically.
// this matches the order we want for [u8], but no others (not even [i8]).
impl SliceOrd<u8> for [u8] {
    #[inline]
    fn compare(&self, other: &[u8]) -> Ordering {
        let order = unsafe {
            memcmp(self.as_ptr(), other.as_ptr(),
                   cmp::min(self.len(), other.len()))
        };
        if order == 0 {
            self.len().cmp(&other.len())
        } else if order < 0 {
            Less
        } else {
            Greater
        }
    }
}

#[doc(hidden)]
/// Trait implemented for types that can be compared for equality using
/// their bytewise representation
trait BytewiseEquality { }

macro_rules! impl_marker_for {
    ($traitname:ident, $($ty:ty),*) => {
        $(
            impl $traitname for $ty { }
        )*
    }
}

impl_marker_for!(BytewiseEquality,
                 u8, i8, u16, i16, u32, i32, u64, i64, usize, isize, char, bool);

#[doc(hidden)]
unsafe impl<'a, T> TrustedRandomAccess for Iter<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a T {
        &*self.ptr.add(i)
    }
    fn may_have_side_effect() -> bool { false }
}

#[doc(hidden)]
unsafe impl<'a, T> TrustedRandomAccess for IterMut<'a, T> {
    unsafe fn get_unchecked(&mut self, i: usize) -> &'a mut T {
        &mut *self.ptr.add(i)
    }
    fn may_have_side_effect() -> bool { false }
}

trait SliceContains: Sized {
    fn slice_contains(&self, x: &[Self]) -> bool;
}

impl<T> SliceContains for T where T: PartialEq {
    default fn slice_contains(&self, x: &[Self]) -> bool {
        x.iter().any(|y| *y == *self)
    }
}

impl SliceContains for u8 {
    fn slice_contains(&self, x: &[Self]) -> bool {
        memchr::memchr(*self, x).is_some()
    }
}

impl SliceContains for i8 {
    fn slice_contains(&self, x: &[Self]) -> bool {
        let byte = *self as u8;
        let bytes: &[u8] = unsafe { from_raw_parts(x.as_ptr() as *const u8, x.len()) };
        memchr::memchr(byte, bytes).is_some()
    }
}
