// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Utilities for slice manipulation
//!
//! The `slice` module contains useful code to help work with slice values.
//! Slices are a view into a block of memory represented as a pointer and a length.
//!
//! ```rust
//! // slicing a Vec
//! let vec = vec!(1i, 2, 3);
//! let int_slice = vec.as_slice();
//! // coercing an array to a slice
//! let str_slice: &[&str] = &["one", "two", "three"];
//! ```
//!
//! Slices are either mutable or shared. The shared slice type is `&[T]`,
//! while the mutable slice type is `&mut[T]`. For example, you can mutate the
//! block of memory that a mutable slice points to:
//!
//! ```rust
//! let x: &mut[int] = &mut [1i, 2, 3];
//! x[1] = 7;
//! assert_eq!(x[0], 1);
//! assert_eq!(x[1], 7);
//! assert_eq!(x[2], 3);
//! ```
//!
//! Here are some of the things this module contains:
//!
//! ## Structs
//!
//! There are several structs that are useful for slices, such as `Items`, which
//! represents iteration over a slice.
//!
//! ## Traits
//!
//! A number of traits add methods that allow you to accomplish tasks
//! with slices, the most important being `SliceExt`. Other traits
//! apply only to slices of elements satisfying certain bounds (like
//! `Ord`).
//!
//! An example is the `slice` method which enables slicing syntax `[a..b]` that
//! returns an immutable "view" into a `Vec` or another slice from the index
//! interval `[a, b)`:
//!
//! ```rust
//! #![feature(slicing_syntax)]
//! fn main() {
//!     let numbers = [0i, 1i, 2i];
//!     let last_numbers = numbers[1..3];
//!     // last_numbers is now &[1i, 2i]
//! }
//! ```
//!
//! ## Implementations of other traits
//!
//! There are several implementations of common traits for slices. Some examples
//! include:
//!
//! * `Clone`
//! * `Eq`, `Ord` - for immutable slices whose element type are `Eq` or `Ord`.
//! * `Hash` - for slices whose element type is `Hash`
//!
//! ## Iteration
//!
//! The method `iter()` returns an iteration value for a slice. The iterator
//! yields references to the slice's elements, so if the element
//! type of the slice is `int`, the element type of the iterator is `&int`.
//!
//! ```rust
//! let numbers = [0i, 1i, 2i];
//! for &x in numbers.iter() {
//!     println!("{} is a number!", x);
//! }
//! ```
//!
//! * `.iter_mut()` returns an iterator that allows modifying each value.
//! * Further iterators exist that split, chunk or permute the slice.

#![doc(primitive = "slice")]

use alloc::boxed::Box;
use core::borrow::{BorrowFrom, BorrowFromMut, ToOwned};
use core::cmp;
use core::iter::{range_step, MultiplicativeIterator};
use core::kinds::Sized;
use core::mem::size_of;
use core::mem;
use core::ops::FnMut;
use core::prelude::{Clone, Greater, Iterator, IteratorExt, Less, None, Option};
use core::prelude::{Ord, Ordering, RawPtr, Some, range};
use core::ptr;
use core::slice as core_slice;
use self::Direction::*;

use vec::Vec;

pub use core::slice::{Chunks, AsSlice, SplitsN, Windows};
pub use core::slice::{Items, MutItems, PartialEqSliceExt};
pub use core::slice::{ImmutableIntSlice, MutableIntSlice};
pub use core::slice::{MutSplits, MutChunks, Splits};
pub use core::slice::{bytes, mut_ref_slice, ref_slice};
pub use core::slice::{from_raw_buf, from_raw_mut_buf, BinarySearchResult};

// Functional utilities

#[allow(missing_docs)]
pub trait VectorVector<T> for Sized? {
    // FIXME #5898: calling these .concat and .connect conflicts with
    // StrVector::con{cat,nect}, since they have generic contents.
    /// Flattens a vector of vectors of `T` into a single `Vec<T>`.
    fn concat_vec(&self) -> Vec<T>;

    /// Concatenate a vector of vectors, placing a given separator between each.
    fn connect_vec(&self, sep: &T) -> Vec<T>;
}

impl<'a, T: Clone, V: AsSlice<T>> VectorVector<T> for [V] {
    fn concat_vec(&self) -> Vec<T> {
        let size = self.iter().fold(0u, |acc, v| acc + v.as_slice().len());
        let mut result = Vec::with_capacity(size);
        for v in self.iter() {
            result.push_all(v.as_slice())
        }
        result
    }

    fn connect_vec(&self, sep: &T) -> Vec<T> {
        let size = self.iter().fold(0u, |acc, v| acc + v.as_slice().len());
        let mut result = Vec::with_capacity(size + self.len());
        let mut first = true;
        for v in self.iter() {
            if first { first = false } else { result.push(sep.clone()) }
            result.push_all(v.as_slice())
        }
        result
    }
}

/// An iterator that yields the element swaps needed to produce
/// a sequence of all possible permutations for an indexed sequence of
/// elements. Each permutation is only a single swap apart.
///
/// The Steinhaus-Johnson-Trotter algorithm is used.
///
/// Generates even and odd permutations alternately.
///
/// The last generated swap is always (0, 1), and it returns the
/// sequence to its initial order.
pub struct ElementSwaps {
    sdir: Vec<SizeDirection>,
    /// If `true`, emit the last swap that returns the sequence to initial
    /// state.
    emit_reset: bool,
    swaps_made : uint,
}

impl ElementSwaps {
    /// Creates an `ElementSwaps` iterator for a sequence of `length` elements.
    pub fn new(length: uint) -> ElementSwaps {
        // Initialize `sdir` with a direction that position should move in
        // (all negative at the beginning) and the `size` of the
        // element (equal to the original index).
        ElementSwaps{
            emit_reset: true,
            sdir: range(0, length).map(|i| SizeDirection{ size: i, dir: Neg }).collect(),
            swaps_made: 0
        }
    }
}

#[deriving(Copy)]
enum Direction { Pos, Neg }

/// An `Index` and `Direction` together.
#[deriving(Copy)]
struct SizeDirection {
    size: uint,
    dir: Direction,
}

impl Iterator<(uint, uint)> for ElementSwaps {
    #[inline]
    fn next(&mut self) -> Option<(uint, uint)> {
        fn new_pos(i: uint, s: Direction) -> uint {
            i + match s { Pos => 1, Neg => -1 }
        }

        // Find the index of the largest mobile element:
        // The direction should point into the vector, and the
        // swap should be with a smaller `size` element.
        let max = self.sdir.iter().map(|&x| x).enumerate()
                           .filter(|&(i, sd)|
                                new_pos(i, sd.dir) < self.sdir.len() &&
                                self.sdir[new_pos(i, sd.dir)].size < sd.size)
                           .max_by(|&(_, sd)| sd.size);
        match max {
            Some((i, sd)) => {
                let j = new_pos(i, sd.dir);
                self.sdir.swap(i, j);

                // Swap the direction of each larger SizeDirection
                for x in self.sdir.iter_mut() {
                    if x.size > sd.size {
                        x.dir = match x.dir { Pos => Neg, Neg => Pos };
                    }
                }
                self.swaps_made += 1;
                Some((i, j))
            },
            None => if self.emit_reset {
                self.emit_reset = false;
                if self.sdir.len() > 1 {
                    // The last swap
                    self.swaps_made += 1;
                    Some((0, 1))
                } else {
                    // Vector is of the form [] or [x], and the only permutation is itself
                    self.swaps_made += 1;
                    Some((0,0))
                }
            } else { None }
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        // For a vector of size n, there are exactly n! permutations.
        let n = range(2, self.sdir.len() + 1).product();
        (n - self.swaps_made, Some(n - self.swaps_made))
    }
}

/// An iterator that uses `ElementSwaps` to iterate through
/// all possible permutations of a vector.
///
/// The first iteration yields a clone of the vector as it is,
/// then each successive element is the vector with one
/// swap applied.
///
/// Generates even and odd permutations alternately.
pub struct Permutations<T> {
    swaps: ElementSwaps,
    v: Vec<T>,
}

impl<T: Clone> Iterator<Vec<T>> for Permutations<T> {
    #[inline]
    fn next(&mut self) -> Option<Vec<T>> {
        match self.swaps.next() {
            None => None,
            Some((0,0)) => Some(self.v.clone()),
            Some((a, b)) => {
                let elt = self.v.clone();
                self.v.swap(a, b);
                Some(elt)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.swaps.size_hint()
    }
}

/// Extension methods for boxed slices.
pub trait BoxedSliceExt<T> {
    /// Convert `self` into a vector without clones or allocation.
    fn into_vec(self) -> Vec<T>;
}

impl<T> BoxedSliceExt<T> for Box<[T]> {
    #[experimental]
    fn into_vec(mut self) -> Vec<T> {
        unsafe {
            let xs = Vec::from_raw_parts(self.as_mut_ptr(), self.len(), self.len());
            mem::forget(self);
            xs
        }
    }
}

/// Allocating extension methods for slices containing `Clone` elements.
pub trait CloneSliceExt<T> for Sized? {
    /// Copies `self` into a new `Vec`.
    fn to_vec(&self) -> Vec<T>;

    /// Partitions the vector into two vectors `(a, b)`, where all
    /// elements of `a` satisfy `f` and all elements of `b` do not.
    fn partitioned<F>(&self, f: F) -> (Vec<T>, Vec<T>) where F: FnMut(&T) -> bool;

    /// Creates an iterator that yields every possible permutation of the
    /// vector in succession.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let v = [1i, 2, 3];
    /// let mut perms = v.permutations();
    ///
    /// for p in perms {
    ///   println!("{}", p);
    /// }
    /// ```
    ///
    /// Iterating through permutations one by one.
    ///
    /// ```rust
    /// let v = [1i, 2, 3];
    /// let mut perms = v.permutations();
    ///
    /// assert_eq!(Some(vec![1i, 2, 3]), perms.next());
    /// assert_eq!(Some(vec![1i, 3, 2]), perms.next());
    /// assert_eq!(Some(vec![3i, 1, 2]), perms.next());
    /// ```
    fn permutations(&self) -> Permutations<T>;

    /// Copies as many elements from `src` as it can into `self` (the
    /// shorter of `self.len()` and `src.len()`). Returns the number
    /// of elements copied.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut dst = [0i, 0, 0];
    /// let src = [1i, 2];
    ///
    /// assert!(dst.clone_from_slice(&src) == 2);
    /// assert!(dst == [1, 2, 0]);
    ///
    /// let src2 = [3i, 4, 5, 6];
    /// assert!(dst.clone_from_slice(&src2) == 3);
    /// assert!(dst == [3i, 4, 5]);
    /// ```
    fn clone_from_slice(&mut self, &[T]) -> uint;
}

impl<T: Clone> CloneSliceExt<T> for [T] {
    /// Returns a copy of `v`.
    #[inline]
    fn to_vec(&self) -> Vec<T> {
        let mut vector = Vec::with_capacity(self.len());
        vector.push_all(self);
        vector
    }


    #[inline]
    fn partitioned<F>(&self, mut f: F) -> (Vec<T>, Vec<T>) where F: FnMut(&T) -> bool {
        let mut lefts  = Vec::new();
        let mut rights = Vec::new();

        for elt in self.iter() {
            if f(elt) {
                lefts.push((*elt).clone());
            } else {
                rights.push((*elt).clone());
            }
        }

        (lefts, rights)
    }

    /// Returns an iterator over all permutations of a vector.
    fn permutations(&self) -> Permutations<T> {
        Permutations{
            swaps: ElementSwaps::new(self.len()),
            v: self.to_vec(),
        }
    }

    fn clone_from_slice(&mut self, src: &[T]) -> uint {
        core_slice::CloneSliceExt::clone_from_slice(self, src)
    }
}

fn insertion_sort<T, F>(v: &mut [T], mut compare: F) where F: FnMut(&T, &T) -> Ordering {
    let len = v.len() as int;
    let buf_v = v.as_mut_ptr();

    // 1 <= i < len;
    for i in range(1, len) {
        // j satisfies: 0 <= j <= i;
        let mut j = i;
        unsafe {
            // `i` is in bounds.
            let read_ptr = buf_v.offset(i) as *const T;

            // find where to insert, we need to do strict <,
            // rather than <=, to maintain stability.

            // 0 <= j - 1 < len, so .offset(j - 1) is in bounds.
            while j > 0 &&
                    compare(&*read_ptr, &*buf_v.offset(j - 1)) == Less {
                j -= 1;
            }

            // shift everything to the right, to make space to
            // insert this value.

            // j + 1 could be `len` (for the last `i`), but in
            // that case, `i == j` so we don't copy. The
            // `.offset(j)` is always in bounds.

            if i != j {
                let tmp = ptr::read(read_ptr);
                ptr::copy_memory(buf_v.offset(j + 1),
                                 &*buf_v.offset(j),
                                 (i - j) as uint);
                ptr::copy_nonoverlapping_memory(buf_v.offset(j),
                                                &tmp as *const T,
                                                1);
                mem::forget(tmp);
            }
        }
    }
}

fn merge_sort<T, F>(v: &mut [T], mut compare: F) where F: FnMut(&T, &T) -> Ordering {
    // warning: this wildly uses unsafe.
    static BASE_INSERTION: uint = 32;
    static LARGE_INSERTION: uint = 16;

    // FIXME #12092: smaller insertion runs seems to make sorting
    // vectors of large elements a little faster on some platforms,
    // but hasn't been tested/tuned extensively
    let insertion = if size_of::<T>() <= 16 {
        BASE_INSERTION
    } else {
        LARGE_INSERTION
    };

    let len = v.len();

    // short vectors get sorted in-place via insertion sort to avoid allocations
    if len <= insertion {
        insertion_sort(v, compare);
        return;
    }

    // allocate some memory to use as scratch memory, we keep the
    // length 0 so we can keep shallow copies of the contents of `v`
    // without risking the dtors running on an object twice if
    // `compare` panics.
    let mut working_space = Vec::with_capacity(2 * len);
    // these both are buffers of length `len`.
    let mut buf_dat = working_space.as_mut_ptr();
    let mut buf_tmp = unsafe {buf_dat.offset(len as int)};

    // length `len`.
    let buf_v = v.as_ptr();

    // step 1. sort short runs with insertion sort. This takes the
    // values from `v` and sorts them into `buf_dat`, leaving that
    // with sorted runs of length INSERTION.

    // We could hardcode the sorting comparisons here, and we could
    // manipulate/step the pointers themselves, rather than repeatedly
    // .offset-ing.
    for start in range_step(0, len, insertion) {
        // start <= i < len;
        for i in range(start, cmp::min(start + insertion, len)) {
            // j satisfies: start <= j <= i;
            let mut j = i as int;
            unsafe {
                // `i` is in bounds.
                let read_ptr = buf_v.offset(i as int);

                // find where to insert, we need to do strict <,
                // rather than <=, to maintain stability.

                // start <= j - 1 < len, so .offset(j - 1) is in
                // bounds.
                while j > start as int &&
                        compare(&*read_ptr, &*buf_dat.offset(j - 1)) == Less {
                    j -= 1;
                }

                // shift everything to the right, to make space to
                // insert this value.

                // j + 1 could be `len` (for the last `i`), but in
                // that case, `i == j` so we don't copy. The
                // `.offset(j)` is always in bounds.
                ptr::copy_memory(buf_dat.offset(j + 1),
                                 &*buf_dat.offset(j),
                                 i - j as uint);
                ptr::copy_nonoverlapping_memory(buf_dat.offset(j), read_ptr, 1);
            }
        }
    }

    // step 2. merge the sorted runs.
    let mut width = insertion;
    while width < len {
        // merge the sorted runs of length `width` in `buf_dat` two at
        // a time, placing the result in `buf_tmp`.

        // 0 <= start <= len.
        for start in range_step(0, len, 2 * width) {
            // manipulate pointers directly for speed (rather than
            // using a `for` loop with `range` and `.offset` inside
            // that loop).
            unsafe {
                // the end of the first run & start of the
                // second. Offset of `len` is defined, since this is
                // precisely one byte past the end of the object.
                let right_start = buf_dat.offset(cmp::min(start + width, len) as int);
                // end of the second. Similar reasoning to the above re safety.
                let right_end_idx = cmp::min(start + 2 * width, len);
                let right_end = buf_dat.offset(right_end_idx as int);

                // the pointers to the elements under consideration
                // from the two runs.

                // both of these are in bounds.
                let mut left = buf_dat.offset(start as int);
                let mut right = right_start;

                // where we're putting the results, it is a run of
                // length `2*width`, so we step it once for each step
                // of either `left` or `right`.  `buf_tmp` has length
                // `len`, so these are in bounds.
                let mut out = buf_tmp.offset(start as int);
                let out_end = buf_tmp.offset(right_end_idx as int);

                while out < out_end {
                    // Either the left or the right run are exhausted,
                    // so just copy the remainder from the other run
                    // and move on; this gives a huge speed-up (order
                    // of 25%) for mostly sorted vectors (the best
                    // case).
                    if left == right_start {
                        // the number remaining in this run.
                        let elems = (right_end as uint - right as uint) / mem::size_of::<T>();
                        ptr::copy_nonoverlapping_memory(out, &*right, elems);
                        break;
                    } else if right == right_end {
                        let elems = (right_start as uint - left as uint) / mem::size_of::<T>();
                        ptr::copy_nonoverlapping_memory(out, &*left, elems);
                        break;
                    }

                    // check which side is smaller, and that's the
                    // next element for the new run.

                    // `left < right_start` and `right < right_end`,
                    // so these are valid.
                    let to_copy = if compare(&*left, &*right) == Greater {
                        step(&mut right)
                    } else {
                        step(&mut left)
                    };
                    ptr::copy_nonoverlapping_memory(out, &*to_copy, 1);
                    step(&mut out);
                }
            }
        }

        mem::swap(&mut buf_dat, &mut buf_tmp);

        width *= 2;
    }

    // write the result to `v` in one go, so that there are never two copies
    // of the same object in `v`.
    unsafe {
        ptr::copy_nonoverlapping_memory(v.as_mut_ptr(), &*buf_dat, len);
    }

    // increment the pointer, returning the old pointer.
    #[inline(always)]
    unsafe fn step<T>(ptr: &mut *mut T) -> *mut T {
        let old = *ptr;
        *ptr = ptr.offset(1);
        old
    }
}

/// Allocating extension methods for slices on Ord values.
#[experimental = "likely to merge with other traits"]
pub trait OrdSliceExt<T> for Sized? {
    /// Sorts the slice, in place.
    ///
    /// This is equivalent to `self.sort_by(|a, b| a.cmp(b))`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut v = [-5i, 4, 1, -3, 2];
    ///
    /// v.sort();
    /// assert!(v == [-5i, -3, 1, 2, 4]);
    /// ```
    #[experimental]
    fn sort(&mut self);

    /// Binary search a sorted slice for a given element.
    ///
    /// If the value is found then `Found` is returned, containing the
    /// index of the matching element; if the value is not found then
    /// `NotFound` is returned, containing the index where a matching
    /// element could be inserted while maintaining sorted order.
    ///
    /// # Example
    ///
    /// Looks up a series of four elements. The first is found, with a
    /// uniquely determined position; the second and third are not
    /// found; the fourth could match any position in `[1,4]`.
    ///
    /// ```rust
    /// use std::slice::BinarySearchResult::{Found, NotFound};
    /// let s = [0i, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
    /// let s = s.as_slice();
    ///
    /// assert_eq!(s.binary_search_elem(&13),  Found(9));
    /// assert_eq!(s.binary_search_elem(&4),   NotFound(7));
    /// assert_eq!(s.binary_search_elem(&100), NotFound(13));
    /// let r = s.binary_search_elem(&1);
    /// assert!(match r { Found(1...4) => true, _ => false, });
    /// ```
    #[unstable = "name likely to change"]
    fn binary_search_elem(&self, x: &T) -> BinarySearchResult;

    /// Mutates the slice to the next lexicographic permutation.
    ///
    /// Returns `true` if successful and `false` if the slice is at the
    /// last-ordered permutation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: &mut [_] = &mut [0i, 1, 2];
    /// v.next_permutation();
    /// let b: &mut [_] = &mut [0i, 2, 1];
    /// assert!(v == b);
    /// v.next_permutation();
    /// let b: &mut [_] = &mut [1i, 0, 2];
    /// assert!(v == b);
    /// ```
    #[experimental]
    fn next_permutation(&mut self) -> bool;

    /// Mutates the slice to the previous lexicographic permutation.
    ///
    /// Returns `true` if successful and `false` if the slice is at the
    /// first-ordered permutation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: &mut [_] = &mut [1i, 0, 2];
    /// v.prev_permutation();
    /// let b: &mut [_] = &mut [0i, 2, 1];
    /// assert!(v == b);
    /// v.prev_permutation();
    /// let b: &mut [_] = &mut [0i, 1, 2];
    /// assert!(v == b);
    /// ```
    #[experimental]
    fn prev_permutation(&mut self) -> bool;
}

impl<T: Ord> OrdSliceExt<T> for [T] {
    #[inline]
    fn sort(&mut self) {
        self.sort_by(|a, b| a.cmp(b))
    }

    fn binary_search_elem(&self, x: &T) -> BinarySearchResult {
        core_slice::OrdSliceExt::binary_search_elem(self, x)
    }

    fn next_permutation(&mut self) -> bool {
        core_slice::OrdSliceExt::next_permutation(self)
    }

    fn prev_permutation(&mut self) -> bool {
        core_slice::OrdSliceExt::prev_permutation(self)
    }
}

/// Allocating extension methods for slices.
#[experimental = "likely to merge with other traits"]
pub trait SliceExt<T> for Sized? {
    /// Sorts the slice, in place, using `compare` to compare
    /// elements.
    ///
    /// This sort is `O(n log n)` worst-case and stable, but allocates
    /// approximately `2 * n`, where `n` is the length of `self`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut v = [5i, 4, 1, 3, 2];
    /// v.sort_by(|a, b| a.cmp(b));
    /// assert!(v == [1, 2, 3, 4, 5]);
    ///
    /// // reverse sorting
    /// v.sort_by(|a, b| b.cmp(a));
    /// assert!(v == [5, 4, 3, 2, 1]);
    /// ```
    fn sort_by<F>(&mut self, compare: F) where F: FnMut(&T, &T) -> Ordering;

    /// Consumes `src` and moves as many elements as it can into `self`
    /// from the range [start,end).
    ///
    /// Returns the number of elements copied (the shorter of `self.len()`
    /// and `end - start`).
    ///
    /// # Arguments
    ///
    /// * src - A mutable vector of `T`
    /// * start - The index into `src` to start copying from
    /// * end - The index into `src` to stop copying from
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut a = [1i, 2, 3, 4, 5];
    /// let b = vec![6i, 7, 8];
    /// let num_moved = a.move_from(b, 0, 3);
    /// assert_eq!(num_moved, 3);
    /// assert!(a == [6i, 7, 8, 4, 5]);
    /// ```
    fn move_from(&mut self, src: Vec<T>, start: uint, end: uint) -> uint;

    /// Returns a subslice spanning the interval [`start`, `end`).
    ///
    /// Panics when the end of the new slice lies beyond the end of the
    /// original slice (i.e. when `end > self.len()`) or when `start > end`.
    ///
    /// Slicing with `start` equal to `end` yields an empty slice.
    #[unstable = "waiting on final error conventions/slicing syntax"]
    fn slice(&self, start: uint, end: uint) -> &[T];

    /// Returns a subslice from `start` to the end of the slice.
    ///
    /// Panics when `start` is strictly greater than the length of the original slice.
    ///
    /// Slicing from `self.len()` yields an empty slice.
    #[unstable = "waiting on final error conventions/slicing syntax"]
    fn slice_from(&self, start: uint) -> &[T];

    /// Returns a subslice from the start of the slice to `end`.
    ///
    /// Panics when `end` is strictly greater than the length of the original slice.
    ///
    /// Slicing to `0` yields an empty slice.
    #[unstable = "waiting on final error conventions/slicing syntax"]
    fn slice_to(&self, end: uint) -> &[T];

    /// Divides one slice into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// Panics if `mid > len`.
    #[unstable = "waiting on final error conventions"]
    fn split_at(&self, mid: uint) -> (&[T], &[T]);

    /// Returns an iterator over the slice
    #[unstable = "iterator type may change"]
    fn iter(&self) -> Items<T>;

    /// Returns an iterator over subslices separated by elements that match
    /// `pred`.  The matched element is not contained in the subslices.
    #[unstable = "iterator type may change, waiting on unboxed closures"]
    fn split<F>(&self, pred: F) -> Splits<T, F>
                where F: FnMut(&T) -> bool;

    /// Returns an iterator over subslices separated by elements that match
    /// `pred`, limited to splitting at most `n` times.  The matched element is
    /// not contained in the subslices.
    #[unstable = "iterator type may change"]
    fn splitn<F>(&self, n: uint, pred: F) -> SplitsN<Splits<T, F>>
                 where F: FnMut(&T) -> bool;

    /// Returns an iterator over subslices separated by elements that match
    /// `pred` limited to splitting at most `n` times. This starts at the end of
    /// the slice and works backwards.  The matched element is not contained in
    /// the subslices.
    #[unstable = "iterator type may change"]
    fn rsplitn<F>(&self, n: uint, pred: F) -> SplitsN<Splits<T, F>>
                  where F: FnMut(&T) -> bool;

    /// Returns an iterator over all contiguous windows of length
    /// `size`. The windows overlap. If the slice is shorter than
    /// `size`, the iterator returns no values.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    ///
    /// # Example
    ///
    /// Print the adjacent pairs of a slice (i.e. `[1,2]`, `[2,3]`,
    /// `[3,4]`):
    ///
    /// ```rust
    /// let v = &[1i, 2, 3, 4];
    /// for win in v.windows(2) {
    ///     println!("{}", win);
    /// }
    /// ```
    #[unstable = "iterator type may change"]
    fn windows(&self, size: uint) -> Windows<T>;

    /// Returns an iterator over `size` elements of the slice at a
    /// time. The chunks do not overlap. If `size` does not divide the
    /// length of the slice, then the last chunk will not have length
    /// `size`.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    ///
    /// # Example
    ///
    /// Print the slice two elements at a time (i.e. `[1,2]`,
    /// `[3,4]`, `[5]`):
    ///
    /// ```rust
    /// let v = &[1i, 2, 3, 4, 5];
    /// for win in v.chunks(2) {
    ///     println!("{}", win);
    /// }
    /// ```
    #[unstable = "iterator type may change"]
    fn chunks(&self, size: uint) -> Chunks<T>;

    /// Returns the element of a slice at the given index, or `None` if the
    /// index is out of bounds.
    #[unstable = "waiting on final collection conventions"]
    fn get(&self, index: uint) -> Option<&T>;

    /// Returns the first element of a slice, or `None` if it is empty.
    #[unstable = "name may change"]
    fn head(&self) -> Option<&T>;

    /// Returns all but the first element of a slice.
    #[unstable = "name may change"]
    fn tail(&self) -> &[T];

    /// Returns all but the last element of a slice.
    #[unstable = "name may change"]
    fn init(&self) -> &[T];

    /// Returns the last element of a slice, or `None` if it is empty.
    #[unstable = "name may change"]
    fn last(&self) -> Option<&T>;

    /// Returns a pointer to the element at the given index, without doing
    /// bounds checking.
    #[unstable]
    unsafe fn unsafe_get(&self, index: uint) -> &T;

    /// Returns an unsafe pointer to the slice's buffer
    ///
    /// The caller must ensure that the slice outlives the pointer this
    /// function returns, or else it will end up pointing to garbage.
    ///
    /// Modifying the slice may cause its buffer to be reallocated, which
    /// would also make any pointers to it invalid.
    #[unstable]
    fn as_ptr(&self) -> *const T;

    /// Binary search a sorted slice with a comparator function.
    ///
    /// The comparator function should implement an order consistent
    /// with the sort order of the underlying slice, returning an
    /// order code that indicates whether its argument is `Less`,
    /// `Equal` or `Greater` the desired target.
    ///
    /// If a matching value is found then returns `Found`, containing
    /// the index for the matched element; if no match is found then
    /// `NotFound` is returned, containing the index where a matching
    /// element could be inserted while maintaining sorted order.
    ///
    /// # Example
    ///
    /// Looks up a series of four elements. The first is found, with a
    /// uniquely determined position; the second and third are not
    /// found; the fourth could match any position in `[1,4]`.
    ///
    /// ```rust
    /// use std::slice::BinarySearchResult::{Found, NotFound};
    /// let s = [0i, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
    /// let s = s.as_slice();
    ///
    /// let seek = 13;
    /// assert_eq!(s.binary_search(|probe| probe.cmp(&seek)), Found(9));
    /// let seek = 4;
    /// assert_eq!(s.binary_search(|probe| probe.cmp(&seek)), NotFound(7));
    /// let seek = 100;
    /// assert_eq!(s.binary_search(|probe| probe.cmp(&seek)), NotFound(13));
    /// let seek = 1;
    /// let r = s.binary_search(|probe| probe.cmp(&seek));
    /// assert!(match r { Found(1...4) => true, _ => false, });
    /// ```
    #[unstable = "waiting on unboxed closures"]
    fn binary_search<F>(&self, f: F) -> BinarySearchResult
                        where F: FnMut(&T) -> Ordering;

    /// Return the number of elements in the slice
    ///
    /// # Example
    ///
    /// ```
    /// let a = [1i, 2, 3];
    /// assert_eq!(a.len(), 3);
    /// ```
    #[experimental = "not triaged yet"]
    fn len(&self) -> uint;

    /// Returns true if the slice has a length of 0
    ///
    /// # Example
    ///
    /// ```
    /// let a = [1i, 2, 3];
    /// assert!(!a.is_empty());
    /// ```
    #[inline]
    #[experimental = "not triaged yet"]
    fn is_empty(&self) -> bool { self.len() == 0 }
    /// Returns a mutable reference to the element at the given index,
    /// or `None` if the index is out of bounds
    #[unstable = "waiting on final error conventions"]
    fn get_mut(&mut self, index: uint) -> Option<&mut T>;

    /// Work with `self` as a mut slice.
    /// Primarily intended for getting a &mut [T] from a [T, ..N].
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Returns a mutable subslice spanning the interval [`start`, `end`).
    ///
    /// Panics when the end of the new slice lies beyond the end of the
    /// original slice (i.e. when `end > self.len()`) or when `start > end`.
    ///
    /// Slicing with `start` equal to `end` yields an empty slice.
    #[unstable = "waiting on final error conventions"]
    fn slice_mut(&mut self, start: uint, end: uint) -> &mut [T];

    /// Returns a mutable subslice from `start` to the end of the slice.
    ///
    /// Panics when `start` is strictly greater than the length of the original slice.
    ///
    /// Slicing from `self.len()` yields an empty slice.
    #[unstable = "waiting on final error conventions"]
    fn slice_from_mut(&mut self, start: uint) -> &mut [T];

    /// Returns a mutable subslice from the start of the slice to `end`.
    ///
    /// Panics when `end` is strictly greater than the length of the original slice.
    ///
    /// Slicing to `0` yields an empty slice.
    #[unstable = "waiting on final error conventions"]
    fn slice_to_mut(&mut self, end: uint) -> &mut [T];

    /// Returns an iterator that allows modifying each value
    #[unstable = "waiting on iterator type name conventions"]
    fn iter_mut(&mut self) -> MutItems<T>;

    /// Returns a mutable pointer to the first element of a slice, or `None` if it is empty
    #[unstable = "name may change"]
    fn head_mut(&mut self) -> Option<&mut T>;

    /// Returns all but the first element of a mutable slice
    #[unstable = "name may change"]
    fn tail_mut(&mut self) -> &mut [T];

    /// Returns all but the last element of a mutable slice
    #[unstable = "name may change"]
    fn init_mut(&mut self) -> &mut [T];

    /// Returns a mutable pointer to the last item in the slice.
    #[unstable = "name may change"]
    fn last_mut(&mut self) -> Option<&mut T>;

    /// Returns an iterator over mutable subslices separated by elements that
    /// match `pred`.  The matched element is not contained in the subslices.
    #[unstable = "waiting on unboxed closures, iterator type name conventions"]
    fn split_mut<F>(&mut self, pred: F) -> MutSplits<T, F>
                    where F: FnMut(&T) -> bool;

    /// Returns an iterator over subslices separated by elements that match
    /// `pred`, limited to splitting at most `n` times.  The matched element is
    /// not contained in the subslices.
    #[unstable = "waiting on unboxed closures, iterator type name conventions"]
    fn splitn_mut<F>(&mut self, n: uint, pred: F) -> SplitsN<MutSplits<T, F>>
                     where F: FnMut(&T) -> bool;

    /// Returns an iterator over subslices separated by elements that match
    /// `pred` limited to splitting at most `n` times. This starts at the end of
    /// the slice and works backwards.  The matched element is not contained in
    /// the subslices.
    #[unstable = "waiting on unboxed closures, iterator type name conventions"]
    fn rsplitn_mut<F>(&mut self,  n: uint, pred: F) -> SplitsN<MutSplits<T, F>>
                      where F: FnMut(&T) -> bool;

    /// Returns an iterator over `chunk_size` elements of the slice at a time.
    /// The chunks are mutable and do not overlap. If `chunk_size` does
    /// not divide the length of the slice, then the last chunk will not
    /// have length `chunk_size`.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    #[unstable = "waiting on iterator type name conventions"]
    fn chunks_mut(&mut self, chunk_size: uint) -> MutChunks<T>;

    /// Swaps two elements in a slice.
    ///
    /// Panics if `a` or `b` are out of bounds.
    ///
    /// # Arguments
    ///
    /// * a - The index of the first element
    /// * b - The index of the second element
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = ["a", "b", "c", "d"];
    /// v.swap(1, 3);
    /// assert!(v == ["a", "d", "c", "b"]);
    /// ```
    #[unstable = "waiting on final error conventions"]
    fn swap(&mut self, a: uint, b: uint);

    /// Divides one `&mut` into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// Panics if `mid > len`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = [1i, 2, 3, 4, 5, 6];
    ///
    /// // scoped to restrict the lifetime of the borrows
    /// {
    ///    let (left, right) = v.split_at_mut(0);
    ///    assert!(left == []);
    ///    assert!(right == [1i, 2, 3, 4, 5, 6]);
    /// }
    ///
    /// {
    ///     let (left, right) = v.split_at_mut(2);
    ///     assert!(left == [1i, 2]);
    ///     assert!(right == [3i, 4, 5, 6]);
    /// }
    ///
    /// {
    ///     let (left, right) = v.split_at_mut(6);
    ///     assert!(left == [1i, 2, 3, 4, 5, 6]);
    ///     assert!(right == []);
    /// }
    /// ```
    #[unstable = "waiting on final error conventions"]
    fn split_at_mut(&mut self, mid: uint) -> (&mut [T], &mut [T]);

    /// Reverse the order of elements in a slice, in place.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = [1i, 2, 3];
    /// v.reverse();
    /// assert!(v == [3i, 2, 1]);
    /// ```
    #[experimental = "may be moved to iterators instead"]
    fn reverse(&mut self);

    /// Returns an unsafe mutable pointer to the element in index
    #[experimental = "waiting on unsafe conventions"]
    unsafe fn unsafe_mut(&mut self, index: uint) -> &mut T;

    /// Return an unsafe mutable pointer to the slice's buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this
    /// function returns, or else it will end up pointing to garbage.
    ///
    /// Modifying the slice may cause its buffer to be reallocated, which
    /// would also make any pointers to it invalid.
    #[inline]
    #[unstable]
    fn as_mut_ptr(&mut self) -> *mut T;
}

impl<T> SliceExt<T> for [T] {
    #[inline]
    fn sort_by<F>(&mut self, compare: F) where F: FnMut(&T, &T) -> Ordering {
        merge_sort(self, compare)
    }

    #[inline]
    fn move_from(&mut self, mut src: Vec<T>, start: uint, end: uint) -> uint {
        for (a, b) in self.iter_mut().zip(src[mut start..end].iter_mut()) {
            mem::swap(a, b);
        }
        cmp::min(self.len(), end-start)
    }

    #[inline]
    fn slice<'a>(&'a self, start: uint, end: uint) -> &'a [T] {
        core_slice::SliceExt::slice(self, start, end)
    }

    #[inline]
    fn slice_from<'a>(&'a self, start: uint) -> &'a [T] {
        core_slice::SliceExt::slice_from(self, start)
    }

    #[inline]
    fn slice_to<'a>(&'a self, end: uint) -> &'a [T] {
        core_slice::SliceExt::slice_to(self, end)
    }

    #[inline]
    fn split_at<'a>(&'a self, mid: uint) -> (&'a [T], &'a [T]) {
        core_slice::SliceExt::split_at(self, mid)
    }

    #[inline]
    fn iter<'a>(&'a self) -> Items<'a, T> {
        core_slice::SliceExt::iter(self)
    }

    #[inline]
    fn split<F>(&self, pred: F) -> Splits<T, F>
                where F: FnMut(&T) -> bool {
        core_slice::SliceExt::split(self, pred)
    }

    #[inline]
    fn splitn<F>(&self, n: uint, pred: F) -> SplitsN<Splits<T, F>>
                 where F: FnMut(&T) -> bool {
        core_slice::SliceExt::splitn(self, n, pred)
    }

    #[inline]
    fn rsplitn<F>(&self, n: uint, pred: F) -> SplitsN<Splits<T, F>>
                  where F: FnMut(&T) -> bool {
        core_slice::SliceExt::rsplitn(self, n, pred)
    }

    #[inline]
    fn windows<'a>(&'a self, size: uint) -> Windows<'a, T> {
        core_slice::SliceExt::windows(self, size)
    }

    #[inline]
    fn chunks<'a>(&'a self, size: uint) -> Chunks<'a, T> {
        core_slice::SliceExt::chunks(self, size)
    }

    #[inline]
    fn get<'a>(&'a self, index: uint) -> Option<&'a T> {
        core_slice::SliceExt::get(self, index)
    }

    #[inline]
    fn head<'a>(&'a self) -> Option<&'a T> {
        core_slice::SliceExt::head(self)
    }

    #[inline]
    fn tail<'a>(&'a self) -> &'a [T] {
        core_slice::SliceExt::tail(self)
    }

    #[inline]
    fn init<'a>(&'a self) -> &'a [T] {
        core_slice::SliceExt::init(self)
    }

    #[inline]
    fn last<'a>(&'a self) -> Option<&'a T> {
        core_slice::SliceExt::last(self)
    }

    #[inline]
    unsafe fn unsafe_get<'a>(&'a self, index: uint) -> &'a T {
        core_slice::SliceExt::unsafe_get(self, index)
    }

    #[inline]
    fn as_ptr(&self) -> *const T {
        core_slice::SliceExt::as_ptr(self)
    }

    #[inline]
    fn binary_search<F>(&self, f: F) -> BinarySearchResult
                        where F: FnMut(&T) -> Ordering {
        core_slice::SliceExt::binary_search(self, f)
    }

    #[inline]
    fn len(&self) -> uint {
        core_slice::SliceExt::len(self)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        core_slice::SliceExt::is_empty(self)
    }

    #[inline]
    fn get_mut<'a>(&'a mut self, index: uint) -> Option<&'a mut T> {
        core_slice::SliceExt::get_mut(self, index)
    }

    #[inline]
    fn as_mut_slice<'a>(&'a mut self) -> &'a mut [T] {
        core_slice::SliceExt::as_mut_slice(self)
    }

    #[inline]
    fn slice_mut<'a>(&'a mut self, start: uint, end: uint) -> &'a mut [T] {
        core_slice::SliceExt::slice_mut(self, start, end)
    }

    #[inline]
    fn slice_from_mut<'a>(&'a mut self, start: uint) -> &'a mut [T] {
        core_slice::SliceExt::slice_from_mut(self, start)
    }

    #[inline]
    fn slice_to_mut<'a>(&'a mut self, end: uint) -> &'a mut [T] {
        core_slice::SliceExt::slice_to_mut(self, end)
    }

    #[inline]
    fn iter_mut<'a>(&'a mut self) -> MutItems<'a, T> {
        core_slice::SliceExt::iter_mut(self)
    }

    #[inline]
    fn head_mut<'a>(&'a mut self) -> Option<&'a mut T> {
        core_slice::SliceExt::head_mut(self)
    }

    #[inline]
    fn tail_mut<'a>(&'a mut self) -> &'a mut [T] {
        core_slice::SliceExt::tail_mut(self)
    }

    #[inline]
    fn init_mut<'a>(&'a mut self) -> &'a mut [T] {
        core_slice::SliceExt::init_mut(self)
    }

    #[inline]
    fn last_mut<'a>(&'a mut self) -> Option<&'a mut T> {
        core_slice::SliceExt::last_mut(self)
    }

    #[inline]
    fn split_mut<F>(&mut self, pred: F) -> MutSplits<T, F>
                    where F: FnMut(&T) -> bool {
        core_slice::SliceExt::split_mut(self, pred)
    }

    #[inline]
    fn splitn_mut<F>(&mut self, n: uint, pred: F) -> SplitsN<MutSplits<T, F>>
                     where F: FnMut(&T) -> bool {
        core_slice::SliceExt::splitn_mut(self, n, pred)
    }

    #[inline]
    fn rsplitn_mut<F>(&mut self,  n: uint, pred: F) -> SplitsN<MutSplits<T, F>>
                      where F: FnMut(&T) -> bool {
        core_slice::SliceExt::rsplitn_mut(self, n, pred)
    }

    #[inline]
    fn chunks_mut<'a>(&'a mut self, chunk_size: uint) -> MutChunks<'a, T> {
        core_slice::SliceExt::chunks_mut(self, chunk_size)
    }

    #[inline]
    fn swap(&mut self, a: uint, b: uint) {
        core_slice::SliceExt::swap(self, a, b)
    }

    #[inline]
    fn split_at_mut<'a>(&'a mut self, mid: uint) -> (&'a mut [T], &'a mut [T]) {
        core_slice::SliceExt::split_at_mut(self, mid)
    }

    #[inline]
    fn reverse(&mut self) {
        core_slice::SliceExt::reverse(self)
    }

    #[inline]
    unsafe fn unsafe_mut<'a>(&'a mut self, index: uint) -> &'a mut T {
        core_slice::SliceExt::unsafe_mut(self, index)
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        core_slice::SliceExt::as_mut_ptr(self)
    }
}

#[unstable = "trait is unstable"]
impl<T> BorrowFrom<Vec<T>> for [T] {
    fn borrow_from(owned: &Vec<T>) -> &[T] { owned[] }
}

#[unstable = "trait is unstable"]
impl<T> BorrowFromMut<Vec<T>> for [T] {
    fn borrow_from_mut(owned: &mut Vec<T>) -> &mut [T] { owned[mut] }
}

#[unstable = "trait is unstable"]
impl<T: Clone> ToOwned<Vec<T>> for [T] {
    fn to_owned(&self) -> Vec<T> { self.to_vec() }
}

/// Unsafe operations
pub mod raw {
    pub use core::slice::raw::{buf_as_slice, mut_buf_as_slice};
    pub use core::slice::raw::{shift_ptr, pop_ptr};
}

#[cfg(test)]
mod tests {
    use std::boxed::Box;
    use prelude::*;
    use core::cell::Cell;
    use core::default::Default;
    use core::mem;
    use std::rand::{Rng, task_rng};
    use std::rc::Rc;
    use super::ElementSwaps;

    fn square(n: uint) -> uint { n * n }

    fn is_odd(n: &uint) -> bool { *n % 2u == 1u }

    #[test]
    fn test_from_fn() {
        // Test on-stack from_fn.
        let mut v = Vec::from_fn(3u, square);
        {
            let v = v.as_slice();
            assert_eq!(v.len(), 3u);
            assert_eq!(v[0], 0u);
            assert_eq!(v[1], 1u);
            assert_eq!(v[2], 4u);
        }

        // Test on-heap from_fn.
        v = Vec::from_fn(5u, square);
        {
            let v = v.as_slice();
            assert_eq!(v.len(), 5u);
            assert_eq!(v[0], 0u);
            assert_eq!(v[1], 1u);
            assert_eq!(v[2], 4u);
            assert_eq!(v[3], 9u);
            assert_eq!(v[4], 16u);
        }
    }

    #[test]
    fn test_from_elem() {
        // Test on-stack from_elem.
        let mut v = Vec::from_elem(2u, 10u);
        {
            let v = v.as_slice();
            assert_eq!(v.len(), 2u);
            assert_eq!(v[0], 10u);
            assert_eq!(v[1], 10u);
        }

        // Test on-heap from_elem.
        v = Vec::from_elem(6u, 20u);
        {
            let v = v.as_slice();
            assert_eq!(v[0], 20u);
            assert_eq!(v[1], 20u);
            assert_eq!(v[2], 20u);
            assert_eq!(v[3], 20u);
            assert_eq!(v[4], 20u);
            assert_eq!(v[5], 20u);
        }
    }

    #[test]
    fn test_is_empty() {
        let xs: [int, ..0] = [];
        assert!(xs.is_empty());
        assert!(![0i].is_empty());
    }

    #[test]
    fn test_len_divzero() {
        type Z = [i8, ..0];
        let v0 : &[Z] = &[];
        let v1 : &[Z] = &[[]];
        let v2 : &[Z] = &[[], []];
        assert_eq!(mem::size_of::<Z>(), 0);
        assert_eq!(v0.len(), 0);
        assert_eq!(v1.len(), 1);
        assert_eq!(v2.len(), 2);
    }

    #[test]
    fn test_get() {
        let mut a = vec![11i];
        assert_eq!(a.as_slice().get(1), None);
        a = vec![11i, 12];
        assert_eq!(a.as_slice().get(1).unwrap(), &12);
        a = vec![11i, 12, 13];
        assert_eq!(a.as_slice().get(1).unwrap(), &12);
    }

    #[test]
    fn test_head() {
        let mut a = vec![];
        assert_eq!(a.as_slice().head(), None);
        a = vec![11i];
        assert_eq!(a.as_slice().head().unwrap(), &11);
        a = vec![11i, 12];
        assert_eq!(a.as_slice().head().unwrap(), &11);
    }

    #[test]
    fn test_head_mut() {
        let mut a = vec![];
        assert_eq!(a.head_mut(), None);
        a = vec![11i];
        assert_eq!(*a.head_mut().unwrap(), 11);
        a = vec![11i, 12];
        assert_eq!(*a.head_mut().unwrap(), 11);
    }

    #[test]
    fn test_tail() {
        let mut a = vec![11i];
        let b: &[int] = &[];
        assert_eq!(a.tail(), b);
        a = vec![11i, 12];
        let b: &[int] = &[12];
        assert_eq!(a.tail(), b);
    }

    #[test]
    fn test_tail_mut() {
        let mut a = vec![11i];
        let b: &mut [int] = &mut [];
        assert!(a.tail_mut() == b);
        a = vec![11i, 12];
        let b: &mut [int] = &mut [12];
        assert!(a.tail_mut() == b);
    }

    #[test]
    #[should_fail]
    fn test_tail_empty() {
        let a: Vec<int> = vec![];
        a.tail();
    }

    #[test]
    #[should_fail]
    fn test_tail_mut_empty() {
        let mut a: Vec<int> = vec![];
        a.tail_mut();
    }

    #[test]
    fn test_init() {
        let mut a = vec![11i];
        let b: &[int] = &[];
        assert_eq!(a.init(), b);
        a = vec![11i, 12];
        let b: &[int] = &[11];
        assert_eq!(a.init(), b);
    }

    #[test]
    fn test_init_mut() {
        let mut a = vec![11i];
        let b: &mut [int] = &mut [];
        assert!(a.init_mut() == b);
        a = vec![11i, 12];
        let b: &mut [int] = &mut [11];
        assert!(a.init_mut() == b);
    }

    #[test]
    #[should_fail]
    fn test_init_empty() {
        let a: Vec<int> = vec![];
        a.init();
    }

    #[test]
    #[should_fail]
    fn test_init_mut_empty() {
        let mut a: Vec<int> = vec![];
        a.init_mut();
    }

    #[test]
    fn test_last() {
        let mut a = vec![];
        assert_eq!(a.as_slice().last(), None);
        a = vec![11i];
        assert_eq!(a.as_slice().last().unwrap(), &11);
        a = vec![11i, 12];
        assert_eq!(a.as_slice().last().unwrap(), &12);
    }

    #[test]
    fn test_last_mut() {
        let mut a = vec![];
        assert_eq!(a.last_mut(), None);
        a = vec![11i];
        assert_eq!(*a.last_mut().unwrap(), 11);
        a = vec![11i, 12];
        assert_eq!(*a.last_mut().unwrap(), 12);
    }

    #[test]
    fn test_slice() {
        // Test fixed length vector.
        let vec_fixed = [1i, 2, 3, 4];
        let v_a = vec_fixed[1u..vec_fixed.len()].to_vec();
        assert_eq!(v_a.len(), 3u);
        let v_a = v_a.as_slice();
        assert_eq!(v_a[0], 2);
        assert_eq!(v_a[1], 3);
        assert_eq!(v_a[2], 4);

        // Test on stack.
        let vec_stack: &[_] = &[1i, 2, 3];
        let v_b = vec_stack[1u..3u].to_vec();
        assert_eq!(v_b.len(), 2u);
        let v_b = v_b.as_slice();
        assert_eq!(v_b[0], 2);
        assert_eq!(v_b[1], 3);

        // Test `Box<[T]>`
        let vec_unique = vec![1i, 2, 3, 4, 5, 6];
        let v_d = vec_unique[1u..6u].to_vec();
        assert_eq!(v_d.len(), 5u);
        let v_d = v_d.as_slice();
        assert_eq!(v_d[0], 2);
        assert_eq!(v_d[1], 3);
        assert_eq!(v_d[2], 4);
        assert_eq!(v_d[3], 5);
        assert_eq!(v_d[4], 6);
    }

    #[test]
    fn test_slice_from() {
        let vec: &[int] = &[1, 2, 3, 4];
        assert_eq!(vec[0..], vec);
        let b: &[int] = &[3, 4];
        assert_eq!(vec[2..], b);
        let b: &[int] = &[];
        assert_eq!(vec[4..], b);
    }

    #[test]
    fn test_slice_to() {
        let vec: &[int] = &[1, 2, 3, 4];
        assert_eq!(vec[..4], vec);
        let b: &[int] = &[1, 2];
        assert_eq!(vec[..2], b);
        let b: &[int] = &[];
        assert_eq!(vec[..0], b);
    }


    #[test]
    fn test_pop() {
        let mut v = vec![5i];
        let e = v.pop();
        assert_eq!(v.len(), 0);
        assert_eq!(e, Some(5));
        let f = v.pop();
        assert_eq!(f, None);
        let g = v.pop();
        assert_eq!(g, None);
    }

    #[test]
    fn test_swap_remove() {
        let mut v = vec![1i, 2, 3, 4, 5];
        let mut e = v.swap_remove(0);
        assert_eq!(e, Some(1));
        assert_eq!(v, vec![5i, 2, 3, 4]);
        e = v.swap_remove(3);
        assert_eq!(e, Some(4));
        assert_eq!(v, vec![5i, 2, 3]);

        e = v.swap_remove(3);
        assert_eq!(e, None);
        assert_eq!(v, vec![5i, 2, 3]);
    }

    #[test]
    fn test_swap_remove_noncopyable() {
        // Tests that we don't accidentally run destructors twice.
        let mut v = Vec::new();
        v.push(box 0u8);
        v.push(box 0u8);
        v.push(box 0u8);
        let mut _e = v.swap_remove(0);
        assert_eq!(v.len(), 2);
        _e = v.swap_remove(1);
        assert_eq!(v.len(), 1);
        _e = v.swap_remove(0);
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn test_push() {
        // Test on-stack push().
        let mut v = vec![];
        v.push(1i);
        assert_eq!(v.len(), 1u);
        assert_eq!(v.as_slice()[0], 1);

        // Test on-heap push().
        v.push(2i);
        assert_eq!(v.len(), 2u);
        assert_eq!(v.as_slice()[0], 1);
        assert_eq!(v.as_slice()[1], 2);
    }

    #[test]
    fn test_grow() {
        // Test on-stack grow().
        let mut v = vec![];
        v.grow(2u, 1i);
        {
            let v = v.as_slice();
            assert_eq!(v.len(), 2u);
            assert_eq!(v[0], 1);
            assert_eq!(v[1], 1);
        }

        // Test on-heap grow().
        v.grow(3u, 2i);
        {
            let v = v.as_slice();
            assert_eq!(v.len(), 5u);
            assert_eq!(v[0], 1);
            assert_eq!(v[1], 1);
            assert_eq!(v[2], 2);
            assert_eq!(v[3], 2);
            assert_eq!(v[4], 2);
        }
    }

    #[test]
    fn test_grow_fn() {
        let mut v = vec![];
        v.grow_fn(3u, square);
        let v = v.as_slice();
        assert_eq!(v.len(), 3u);
        assert_eq!(v[0], 0u);
        assert_eq!(v[1], 1u);
        assert_eq!(v[2], 4u);
    }

    #[test]
    fn test_truncate() {
        let mut v = vec![box 6i,box 5,box 4];
        v.truncate(1);
        let v = v.as_slice();
        assert_eq!(v.len(), 1);
        assert_eq!(*(v[0]), 6);
        // If the unsafe block didn't drop things properly, we blow up here.
    }

    #[test]
    fn test_clear() {
        let mut v = vec![box 6i,box 5,box 4];
        v.clear();
        assert_eq!(v.len(), 0);
        // If the unsafe block didn't drop things properly, we blow up here.
    }

    #[test]
    fn test_dedup() {
        fn case(a: Vec<uint>, b: Vec<uint>) {
            let mut v = a;
            v.dedup();
            assert_eq!(v, b);
        }
        case(vec![], vec![]);
        case(vec![1u], vec![1]);
        case(vec![1u,1], vec![1]);
        case(vec![1u,2,3], vec![1,2,3]);
        case(vec![1u,1,2,3], vec![1,2,3]);
        case(vec![1u,2,2,3], vec![1,2,3]);
        case(vec![1u,2,3,3], vec![1,2,3]);
        case(vec![1u,1,2,2,2,3,3], vec![1,2,3]);
    }

    #[test]
    fn test_dedup_unique() {
        let mut v0 = vec![box 1i, box 1, box 2, box 3];
        v0.dedup();
        let mut v1 = vec![box 1i, box 2, box 2, box 3];
        v1.dedup();
        let mut v2 = vec![box 1i, box 2, box 3, box 3];
        v2.dedup();
        /*
         * If the boxed pointers were leaked or otherwise misused, valgrind
         * and/or rt should raise errors.
         */
    }

    #[test]
    fn test_dedup_shared() {
        let mut v0 = vec![box 1i, box 1, box 2, box 3];
        v0.dedup();
        let mut v1 = vec![box 1i, box 2, box 2, box 3];
        v1.dedup();
        let mut v2 = vec![box 1i, box 2, box 3, box 3];
        v2.dedup();
        /*
         * If the pointers were leaked or otherwise misused, valgrind and/or
         * rt should raise errors.
         */
    }

    #[test]
    fn test_retain() {
        let mut v = vec![1u, 2, 3, 4, 5];
        v.retain(is_odd);
        assert_eq!(v, vec![1u, 3, 5]);
    }

    #[test]
    fn test_element_swaps() {
        let mut v = [1i, 2, 3];
        for (i, (a, b)) in ElementSwaps::new(v.len()).enumerate() {
            v.swap(a, b);
            match i {
                0 => assert!(v == [1, 3, 2]),
                1 => assert!(v == [3, 1, 2]),
                2 => assert!(v == [3, 2, 1]),
                3 => assert!(v == [2, 3, 1]),
                4 => assert!(v == [2, 1, 3]),
                5 => assert!(v == [1, 2, 3]),
                _ => panic!(),
            }
        }
    }

    #[test]
    fn test_permutations() {
        {
            let v: [int, ..0] = [];
            let mut it = v.permutations();
            let (min_size, max_opt) = it.size_hint();
            assert_eq!(min_size, 1);
            assert_eq!(max_opt.unwrap(), 1);
            assert_eq!(it.next(), Some(v.as_slice().to_vec()));
            assert_eq!(it.next(), None);
        }
        {
            let v = ["Hello".to_string()];
            let mut it = v.permutations();
            let (min_size, max_opt) = it.size_hint();
            assert_eq!(min_size, 1);
            assert_eq!(max_opt.unwrap(), 1);
            assert_eq!(it.next(), Some(v.as_slice().to_vec()));
            assert_eq!(it.next(), None);
        }
        {
            let v = [1i, 2, 3];
            let mut it = v.permutations();
            let (min_size, max_opt) = it.size_hint();
            assert_eq!(min_size, 3*2);
            assert_eq!(max_opt.unwrap(), 3*2);
            assert_eq!(it.next(), Some(vec![1,2,3]));
            assert_eq!(it.next(), Some(vec![1,3,2]));
            assert_eq!(it.next(), Some(vec![3,1,2]));
            let (min_size, max_opt) = it.size_hint();
            assert_eq!(min_size, 3);
            assert_eq!(max_opt.unwrap(), 3);
            assert_eq!(it.next(), Some(vec![3,2,1]));
            assert_eq!(it.next(), Some(vec![2,3,1]));
            assert_eq!(it.next(), Some(vec![2,1,3]));
            assert_eq!(it.next(), None);
        }
        {
            // check that we have N! permutations
            let v = ['A', 'B', 'C', 'D', 'E', 'F'];
            let mut amt = 0;
            let mut it = v.permutations();
            let (min_size, max_opt) = it.size_hint();
            for _perm in it {
                amt += 1;
            }
            assert_eq!(amt, it.swaps.swaps_made);
            assert_eq!(amt, min_size);
            assert_eq!(amt, 2 * 3 * 4 * 5 * 6);
            assert_eq!(amt, max_opt.unwrap());
        }
    }

    #[test]
    fn test_lexicographic_permutations() {
        let v : &mut[int] = &mut[1i, 2, 3, 4, 5];
        assert!(v.prev_permutation() == false);
        assert!(v.next_permutation());
        let b: &mut[int] = &mut[1, 2, 3, 5, 4];
        assert!(v == b);
        assert!(v.prev_permutation());
        let b: &mut[int] = &mut[1, 2, 3, 4, 5];
        assert!(v == b);
        assert!(v.next_permutation());
        assert!(v.next_permutation());
        let b: &mut[int] = &mut[1, 2, 4, 3, 5];
        assert!(v == b);
        assert!(v.next_permutation());
        let b: &mut[int] = &mut[1, 2, 4, 5, 3];
        assert!(v == b);

        let v : &mut[int] = &mut[1i, 0, 0, 0];
        assert!(v.next_permutation() == false);
        assert!(v.prev_permutation());
        let b: &mut[int] = &mut[0, 1, 0, 0];
        assert!(v == b);
        assert!(v.prev_permutation());
        let b: &mut[int] = &mut[0, 0, 1, 0];
        assert!(v == b);
        assert!(v.prev_permutation());
        let b: &mut[int] = &mut[0, 0, 0, 1];
        assert!(v == b);
        assert!(v.prev_permutation() == false);
    }

    #[test]
    fn test_lexicographic_permutations_empty_and_short() {
        let empty : &mut[int] = &mut[];
        assert!(empty.next_permutation() == false);
        let b: &mut[int] = &mut[];
        assert!(empty == b);
        assert!(empty.prev_permutation() == false);
        assert!(empty == b);

        let one_elem : &mut[int] = &mut[4i];
        assert!(one_elem.prev_permutation() == false);
        let b: &mut[int] = &mut[4];
        assert!(one_elem == b);
        assert!(one_elem.next_permutation() == false);
        assert!(one_elem == b);

        let two_elem : &mut[int] = &mut[1i, 2];
        assert!(two_elem.prev_permutation() == false);
        let b : &mut[int] = &mut[1, 2];
        let c : &mut[int] = &mut[2, 1];
        assert!(two_elem == b);
        assert!(two_elem.next_permutation());
        assert!(two_elem == c);
        assert!(two_elem.next_permutation() == false);
        assert!(two_elem == c);
        assert!(two_elem.prev_permutation());
        assert!(two_elem == b);
        assert!(two_elem.prev_permutation() == false);
        assert!(two_elem == b);
    }

    #[test]
    fn test_position_elem() {
        assert!([].position_elem(&1i).is_none());

        let v1 = vec![1i, 2, 3, 3, 2, 5];
        assert_eq!(v1.as_slice().position_elem(&1), Some(0u));
        assert_eq!(v1.as_slice().position_elem(&2), Some(1u));
        assert_eq!(v1.as_slice().position_elem(&5), Some(5u));
        assert!(v1.as_slice().position_elem(&4).is_none());
    }

    #[test]
    fn test_binary_search_elem() {
        assert_eq!([1i,2,3,4,5].binary_search_elem(&5).found(), Some(4));
        assert_eq!([1i,2,3,4,5].binary_search_elem(&4).found(), Some(3));
        assert_eq!([1i,2,3,4,5].binary_search_elem(&3).found(), Some(2));
        assert_eq!([1i,2,3,4,5].binary_search_elem(&2).found(), Some(1));
        assert_eq!([1i,2,3,4,5].binary_search_elem(&1).found(), Some(0));

        assert_eq!([2i,4,6,8,10].binary_search_elem(&1).found(), None);
        assert_eq!([2i,4,6,8,10].binary_search_elem(&5).found(), None);
        assert_eq!([2i,4,6,8,10].binary_search_elem(&4).found(), Some(1));
        assert_eq!([2i,4,6,8,10].binary_search_elem(&10).found(), Some(4));

        assert_eq!([2i,4,6,8].binary_search_elem(&1).found(), None);
        assert_eq!([2i,4,6,8].binary_search_elem(&5).found(), None);
        assert_eq!([2i,4,6,8].binary_search_elem(&4).found(), Some(1));
        assert_eq!([2i,4,6,8].binary_search_elem(&8).found(), Some(3));

        assert_eq!([2i,4,6].binary_search_elem(&1).found(), None);
        assert_eq!([2i,4,6].binary_search_elem(&5).found(), None);
        assert_eq!([2i,4,6].binary_search_elem(&4).found(), Some(1));
        assert_eq!([2i,4,6].binary_search_elem(&6).found(), Some(2));

        assert_eq!([2i,4].binary_search_elem(&1).found(), None);
        assert_eq!([2i,4].binary_search_elem(&5).found(), None);
        assert_eq!([2i,4].binary_search_elem(&2).found(), Some(0));
        assert_eq!([2i,4].binary_search_elem(&4).found(), Some(1));

        assert_eq!([2i].binary_search_elem(&1).found(), None);
        assert_eq!([2i].binary_search_elem(&5).found(), None);
        assert_eq!([2i].binary_search_elem(&2).found(), Some(0));

        assert_eq!([].binary_search_elem(&1i).found(), None);
        assert_eq!([].binary_search_elem(&5i).found(), None);

        assert!([1i,1,1,1,1].binary_search_elem(&1).found() != None);
        assert!([1i,1,1,1,2].binary_search_elem(&1).found() != None);
        assert!([1i,1,1,2,2].binary_search_elem(&1).found() != None);
        assert!([1i,1,2,2,2].binary_search_elem(&1).found() != None);
        assert_eq!([1i,2,2,2,2].binary_search_elem(&1).found(), Some(0));

        assert_eq!([1i,2,3,4,5].binary_search_elem(&6).found(), None);
        assert_eq!([1i,2,3,4,5].binary_search_elem(&0).found(), None);
    }

    #[test]
    fn test_reverse() {
        let mut v: Vec<int> = vec![10i, 20];
        assert_eq!(v[0], 10);
        assert_eq!(v[1], 20);
        v.reverse();
        assert_eq!(v[0], 20);
        assert_eq!(v[1], 10);

        let mut v3: Vec<int> = vec![];
        v3.reverse();
        assert!(v3.is_empty());
    }

    #[test]
    fn test_sort() {
        for len in range(4u, 25) {
            for _ in range(0i, 100) {
                let mut v = task_rng().gen_iter::<uint>().take(len)
                                      .collect::<Vec<uint>>();
                let mut v1 = v.clone();

                v.sort();
                assert!(v.as_slice().windows(2).all(|w| w[0] <= w[1]));

                v1.sort_by(|a, b| a.cmp(b));
                assert!(v1.as_slice().windows(2).all(|w| w[0] <= w[1]));

                v1.sort_by(|a, b| b.cmp(a));
                assert!(v1.as_slice().windows(2).all(|w| w[0] >= w[1]));
            }
        }

        // shouldn't panic
        let mut v: [uint, .. 0] = [];
        v.sort();

        let mut v = [0xDEADBEEFu];
        v.sort();
        assert!(v == [0xDEADBEEF]);
    }

    #[test]
    fn test_sort_stability() {
        for len in range(4i, 25) {
            for _ in range(0u, 10) {
                let mut counts = [0i, .. 10];

                // create a vector like [(6, 1), (5, 1), (6, 2), ...],
                // where the first item of each tuple is random, but
                // the second item represents which occurrence of that
                // number this element is, i.e. the second elements
                // will occur in sorted order.
                let mut v = range(0, len).map(|_| {
                        let n = task_rng().gen::<uint>() % 10;
                        counts[n] += 1;
                        (n, counts[n])
                    }).collect::<Vec<(uint, int)>>();

                // only sort on the first element, so an unstable sort
                // may mix up the counts.
                v.sort_by(|&(a,_), &(b,_)| a.cmp(&b));

                // this comparison includes the count (the second item
                // of the tuple), so elements with equal first items
                // will need to be ordered with increasing
                // counts... i.e. exactly asserting that this sort is
                // stable.
                assert!(v.as_slice().windows(2).all(|w| w[0] <= w[1]));
            }
        }
    }

    #[test]
    fn test_partition() {
        assert_eq!((vec![]).partition(|x: &int| *x < 3), (vec![], vec![]));
        assert_eq!((vec![1i, 2, 3]).partition(|x: &int| *x < 4), (vec![1, 2, 3], vec![]));
        assert_eq!((vec![1i, 2, 3]).partition(|x: &int| *x < 2), (vec![1], vec![2, 3]));
        assert_eq!((vec![1i, 2, 3]).partition(|x: &int| *x < 0), (vec![], vec![1, 2, 3]));
    }

    #[test]
    fn test_partitioned() {
        assert_eq!(([]).partitioned(|x: &int| *x < 3), (vec![], vec![]));
        assert_eq!(([1i, 2, 3]).partitioned(|x: &int| *x < 4), (vec![1, 2, 3], vec![]));
        assert_eq!(([1i, 2, 3]).partitioned(|x: &int| *x < 2), (vec![1], vec![2, 3]));
        assert_eq!(([1i, 2, 3]).partitioned(|x: &int| *x < 0), (vec![], vec![1, 2, 3]));
    }

    #[test]
    fn test_concat() {
        let v: [Vec<int>, ..0] = [];
        assert_eq!(v.concat_vec(), vec![]);
        assert_eq!([vec![1i], vec![2i,3i]].concat_vec(), vec![1, 2, 3]);

        let v: [&[int], ..2] = [&[1], &[2, 3]];
        assert_eq!(v.connect_vec(&0), vec![1, 0, 2, 3]);
        let v: [&[int], ..3] = [&[1], &[2], &[3]];
        assert_eq!(v.connect_vec(&0), vec![1, 0, 2, 0, 3]);
    }

    #[test]
    fn test_connect() {
        let v: [Vec<int>, ..0] = [];
        assert_eq!(v.connect_vec(&0), vec![]);
        assert_eq!([vec![1i], vec![2i, 3]].connect_vec(&0), vec![1, 0, 2, 3]);
        assert_eq!([vec![1i], vec![2i], vec![3i]].connect_vec(&0), vec![1, 0, 2, 0, 3]);

        let v: [&[int], ..2] = [&[1], &[2, 3]];
        assert_eq!(v.connect_vec(&0), vec![1, 0, 2, 3]);
        let v: [&[int], ..3] = [&[1], &[2], &[3]];
        assert_eq!(v.connect_vec(&0), vec![1, 0, 2, 0, 3]);
    }

    #[test]
    fn test_insert() {
        let mut a = vec![1i, 2, 4];
        a.insert(2, 3);
        assert_eq!(a, vec![1, 2, 3, 4]);

        let mut a = vec![1i, 2, 3];
        a.insert(0, 0);
        assert_eq!(a, vec![0, 1, 2, 3]);

        let mut a = vec![1i, 2, 3];
        a.insert(3, 4);
        assert_eq!(a, vec![1, 2, 3, 4]);

        let mut a = vec![];
        a.insert(0, 1i);
        assert_eq!(a, vec![1]);
    }

    #[test]
    #[should_fail]
    fn test_insert_oob() {
        let mut a = vec![1i, 2, 3];
        a.insert(4, 5);
    }

    #[test]
    fn test_remove() {
        let mut a = vec![1i,2,3,4];

        assert_eq!(a.remove(2), Some(3));
        assert_eq!(a, vec![1i,2,4]);

        assert_eq!(a.remove(2), Some(4));
        assert_eq!(a, vec![1i,2]);

        assert_eq!(a.remove(2), None);
        assert_eq!(a, vec![1i,2]);

        assert_eq!(a.remove(0), Some(1));
        assert_eq!(a, vec![2i]);

        assert_eq!(a.remove(0), Some(2));
        assert_eq!(a, vec![]);

        assert_eq!(a.remove(0), None);
        assert_eq!(a.remove(10), None);
    }

    #[test]
    fn test_capacity() {
        let mut v = vec![0u64];
        v.reserve_exact(10u);
        assert!(v.capacity() >= 11u);
        let mut v = vec![0u32];
        v.reserve_exact(10u);
        assert!(v.capacity() >= 11u);
    }

    #[test]
    fn test_slice_2() {
        let v = vec![1i, 2, 3, 4, 5];
        let v = v.slice(1u, 3u);
        assert_eq!(v.len(), 2u);
        assert_eq!(v[0], 2);
        assert_eq!(v[1], 3);
    }


    #[test]
    #[should_fail]
    fn test_from_fn_fail() {
        Vec::from_fn(100, |v| {
            if v == 50 { panic!() }
            box 0i
        });
    }

    #[test]
    #[should_fail]
    fn test_from_elem_fail() {

        struct S {
            f: Cell<int>,
            boxes: (Box<int>, Rc<int>)
        }

        impl Clone for S {
            fn clone(&self) -> S {
                self.f.set(self.f.get() + 1);
                if self.f.get() == 10 { panic!() }
                S {
                    f: self.f.clone(),
                    boxes: self.boxes.clone(),
                }
            }
        }

        let s = S {
            f: Cell::new(0),
            boxes: (box 0, Rc::new(0)),
        };
        let _ = Vec::from_elem(100, s);
    }

    #[test]
    #[should_fail]
    fn test_grow_fn_fail() {
        let mut v = vec![];
        v.grow_fn(100, |i| {
            if i == 50 {
                panic!()
            }
            (box 0i, Rc::new(0i))
        })
    }

    #[test]
    #[should_fail]
    fn test_permute_fail() {
        let v = [(box 0i, Rc::new(0i)), (box 0i, Rc::new(0i)),
                 (box 0i, Rc::new(0i)), (box 0i, Rc::new(0i))];
        let mut i = 0u;
        for _ in v.permutations() {
            if i == 2 {
                panic!()
            }
            i += 1;
        }
    }

    #[test]
    fn test_total_ord() {
        let c: &[int] = &[1, 2, 3];
        [1, 2, 3, 4][].cmp(c) == Greater;
        let c: &[int] = &[1, 2, 3, 4];
        [1, 2, 3][].cmp(c) == Less;
        let c: &[int] = &[1, 2, 3, 6];
        [1, 2, 3, 4][].cmp(c) == Equal;
        let c: &[int] = &[1, 2, 3, 4, 5, 6];
        [1, 2, 3, 4, 5, 5, 5, 5][].cmp(c) == Less;
        let c: &[int] = &[1, 2, 3, 4];
        [2, 2][].cmp(c) == Greater;
    }

    #[test]
    fn test_iterator() {
        let xs = [1i, 2, 5, 10, 11];
        let mut it = xs.iter();
        assert_eq!(it.size_hint(), (5, Some(5)));
        assert_eq!(it.next().unwrap(), &1);
        assert_eq!(it.size_hint(), (4, Some(4)));
        assert_eq!(it.next().unwrap(), &2);
        assert_eq!(it.size_hint(), (3, Some(3)));
        assert_eq!(it.next().unwrap(), &5);
        assert_eq!(it.size_hint(), (2, Some(2)));
        assert_eq!(it.next().unwrap(), &10);
        assert_eq!(it.size_hint(), (1, Some(1)));
        assert_eq!(it.next().unwrap(), &11);
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_random_access_iterator() {
        let xs = [1i, 2, 5, 10, 11];
        let mut it = xs.iter();

        assert_eq!(it.indexable(), 5);
        assert_eq!(it.idx(0).unwrap(), &1);
        assert_eq!(it.idx(2).unwrap(), &5);
        assert_eq!(it.idx(4).unwrap(), &11);
        assert!(it.idx(5).is_none());

        assert_eq!(it.next().unwrap(), &1);
        assert_eq!(it.indexable(), 4);
        assert_eq!(it.idx(0).unwrap(), &2);
        assert_eq!(it.idx(3).unwrap(), &11);
        assert!(it.idx(4).is_none());

        assert_eq!(it.next().unwrap(), &2);
        assert_eq!(it.indexable(), 3);
        assert_eq!(it.idx(1).unwrap(), &10);
        assert!(it.idx(3).is_none());

        assert_eq!(it.next().unwrap(), &5);
        assert_eq!(it.indexable(), 2);
        assert_eq!(it.idx(1).unwrap(), &11);

        assert_eq!(it.next().unwrap(), &10);
        assert_eq!(it.indexable(), 1);
        assert_eq!(it.idx(0).unwrap(), &11);
        assert!(it.idx(1).is_none());

        assert_eq!(it.next().unwrap(), &11);
        assert_eq!(it.indexable(), 0);
        assert!(it.idx(0).is_none());

        assert!(it.next().is_none());
    }

    #[test]
    fn test_iter_size_hints() {
        let mut xs = [1i, 2, 5, 10, 11];
        assert_eq!(xs.iter().size_hint(), (5, Some(5)));
        assert_eq!(xs.iter_mut().size_hint(), (5, Some(5)));
    }

    #[test]
    fn test_iter_clone() {
        let xs = [1i, 2, 5];
        let mut it = xs.iter();
        it.next();
        let mut jt = it.clone();
        assert_eq!(it.next(), jt.next());
        assert_eq!(it.next(), jt.next());
        assert_eq!(it.next(), jt.next());
    }

    #[test]
    fn test_mut_iterator() {
        let mut xs = [1i, 2, 3, 4, 5];
        for x in xs.iter_mut() {
            *x += 1;
        }
        assert!(xs == [2, 3, 4, 5, 6])
    }

    #[test]
    fn test_rev_iterator() {

        let xs = [1i, 2, 5, 10, 11];
        let ys = [11, 10, 5, 2, 1];
        let mut i = 0;
        for &x in xs.iter().rev() {
            assert_eq!(x, ys[i]);
            i += 1;
        }
        assert_eq!(i, 5);
    }

    #[test]
    fn test_mut_rev_iterator() {
        let mut xs = [1u, 2, 3, 4, 5];
        for (i,x) in xs.iter_mut().rev().enumerate() {
            *x += i;
        }
        assert!(xs == [5, 5, 5, 5, 5])
    }

    #[test]
    fn test_move_iterator() {
        let xs = vec![1u,2,3,4,5];
        assert_eq!(xs.into_iter().fold(0, |a: uint, b: uint| 10*a + b), 12345);
    }

    #[test]
    fn test_move_rev_iterator() {
        let xs = vec![1u,2,3,4,5];
        assert_eq!(xs.into_iter().rev().fold(0, |a: uint, b: uint| 10*a + b), 54321);
    }

    #[test]
    fn test_splitator() {
        let xs = &[1i,2,3,4,5];

        let splits: &[&[int]] = &[&[1], &[3], &[5]];
        assert_eq!(xs.split(|x| *x % 2 == 0).collect::<Vec<&[int]>>(),
                   splits);
        let splits: &[&[int]] = &[&[], &[2,3,4,5]];
        assert_eq!(xs.split(|x| *x == 1).collect::<Vec<&[int]>>(),
                   splits);
        let splits: &[&[int]] = &[&[1,2,3,4], &[]];
        assert_eq!(xs.split(|x| *x == 5).collect::<Vec<&[int]>>(),
                   splits);
        let splits: &[&[int]] = &[&[1,2,3,4,5]];
        assert_eq!(xs.split(|x| *x == 10).collect::<Vec<&[int]>>(),
                   splits);
        let splits: &[&[int]] = &[&[], &[], &[], &[], &[], &[]];
        assert_eq!(xs.split(|_| true).collect::<Vec<&[int]>>(),
                   splits);

        let xs: &[int] = &[];
        let splits: &[&[int]] = &[&[]];
        assert_eq!(xs.split(|x| *x == 5).collect::<Vec<&[int]>>(), splits);
    }

    #[test]
    fn test_splitnator() {
        let xs = &[1i,2,3,4,5];

        let splits: &[&[int]] = &[&[1,2,3,4,5]];
        assert_eq!(xs.splitn(0, |x| *x % 2 == 0).collect::<Vec<&[int]>>(),
                   splits);
        let splits: &[&[int]] = &[&[1], &[3,4,5]];
        assert_eq!(xs.splitn(1, |x| *x % 2 == 0).collect::<Vec<&[int]>>(),
                   splits);
        let splits: &[&[int]] = &[&[], &[], &[], &[4,5]];
        assert_eq!(xs.splitn(3, |_| true).collect::<Vec<&[int]>>(),
                   splits);

        let xs: &[int] = &[];
        let splits: &[&[int]] = &[&[]];
        assert_eq!(xs.splitn(1, |x| *x == 5).collect::<Vec<&[int]>>(), splits);
    }

    #[test]
    fn test_splitnator_mut() {
        let xs = &mut [1i,2,3,4,5];

        let splits: &[&mut [int]] = &[&mut [1,2,3,4,5]];
        assert_eq!(xs.splitn_mut(0, |x| *x % 2 == 0).collect::<Vec<&mut [int]>>(),
                   splits);
        let splits: &[&mut [int]] = &[&mut [1], &mut [3,4,5]];
        assert_eq!(xs.splitn_mut(1, |x| *x % 2 == 0).collect::<Vec<&mut [int]>>(),
                   splits);
        let splits: &[&mut [int]] = &[&mut [], &mut [], &mut [], &mut [4,5]];
        assert_eq!(xs.splitn_mut(3, |_| true).collect::<Vec<&mut [int]>>(),
                   splits);

        let xs: &mut [int] = &mut [];
        let splits: &[&mut [int]] = &[&mut []];
        assert_eq!(xs.splitn_mut(1, |x| *x == 5).collect::<Vec<&mut [int]>>(),
                   splits);
    }

    #[test]
    fn test_rsplitator() {
        let xs = &[1i,2,3,4,5];

        let splits: &[&[int]] = &[&[5], &[3], &[1]];
        assert_eq!(xs.split(|x| *x % 2 == 0).rev().collect::<Vec<&[int]>>(),
                   splits);
        let splits: &[&[int]] = &[&[2,3,4,5], &[]];
        assert_eq!(xs.split(|x| *x == 1).rev().collect::<Vec<&[int]>>(),
                   splits);
        let splits: &[&[int]] = &[&[], &[1,2,3,4]];
        assert_eq!(xs.split(|x| *x == 5).rev().collect::<Vec<&[int]>>(),
                   splits);
        let splits: &[&[int]] = &[&[1,2,3,4,5]];
        assert_eq!(xs.split(|x| *x == 10).rev().collect::<Vec<&[int]>>(),
                   splits);

        let xs: &[int] = &[];
        let splits: &[&[int]] = &[&[]];
        assert_eq!(xs.split(|x| *x == 5).rev().collect::<Vec<&[int]>>(), splits);
    }

    #[test]
    fn test_rsplitnator() {
        let xs = &[1,2,3,4,5];

        let splits: &[&[int]] = &[&[1,2,3,4,5]];
        assert_eq!(xs.rsplitn(0, |x| *x % 2 == 0).collect::<Vec<&[int]>>(),
                   splits);
        let splits: &[&[int]] = &[&[5], &[1,2,3]];
        assert_eq!(xs.rsplitn(1, |x| *x % 2 == 0).collect::<Vec<&[int]>>(),
                   splits);
        let splits: &[&[int]] = &[&[], &[], &[], &[1,2]];
        assert_eq!(xs.rsplitn(3, |_| true).collect::<Vec<&[int]>>(),
                   splits);

        let xs: &[int] = &[];
        let splits: &[&[int]] = &[&[]];
        assert_eq!(xs.rsplitn(1, |x| *x == 5).collect::<Vec<&[int]>>(), splits);
    }

    #[test]
    fn test_windowsator() {
        let v = &[1i,2,3,4];

        let wins: &[&[int]] = &[&[1,2], &[2,3], &[3,4]];
        assert_eq!(v.windows(2).collect::<Vec<&[int]>>(), wins);
        let wins: &[&[int]] = &[&[1i,2,3], &[2,3,4]];
        assert_eq!(v.windows(3).collect::<Vec<&[int]>>(), wins);
        assert!(v.windows(6).next().is_none());
    }

    #[test]
    #[should_fail]
    fn test_windowsator_0() {
        let v = &[1i,2,3,4];
        let _it = v.windows(0);
    }

    #[test]
    fn test_chunksator() {
        let v = &[1i,2,3,4,5];

        let chunks: &[&[int]] = &[&[1i,2], &[3,4], &[5]];
        assert_eq!(v.chunks(2).collect::<Vec<&[int]>>(), chunks);
        let chunks: &[&[int]] = &[&[1i,2,3], &[4,5]];
        assert_eq!(v.chunks(3).collect::<Vec<&[int]>>(), chunks);
        let chunks: &[&[int]] = &[&[1i,2,3,4,5]];
        assert_eq!(v.chunks(6).collect::<Vec<&[int]>>(), chunks);

        let chunks: &[&[int]] = &[&[5i], &[3,4], &[1,2]];
        assert_eq!(v.chunks(2).rev().collect::<Vec<&[int]>>(), chunks);
        let mut it = v.chunks(2);
        assert_eq!(it.indexable(), 3);
        let chunk: &[int] = &[1,2];
        assert_eq!(it.idx(0).unwrap(), chunk);
        let chunk: &[int] = &[3,4];
        assert_eq!(it.idx(1).unwrap(), chunk);
        let chunk: &[int] = &[5];
        assert_eq!(it.idx(2).unwrap(), chunk);
        assert_eq!(it.idx(3), None);
    }

    #[test]
    #[should_fail]
    fn test_chunksator_0() {
        let v = &[1i,2,3,4];
        let _it = v.chunks(0);
    }

    #[test]
    fn test_move_from() {
        let mut a = [1i,2,3,4,5];
        let b = vec![6i,7,8];
        assert_eq!(a.move_from(b, 0, 3), 3);
        assert!(a == [6i,7,8,4,5]);
        let mut a = [7i,2,8,1];
        let b = vec![3i,1,4,1,5,9];
        assert_eq!(a.move_from(b, 0, 6), 4);
        assert!(a == [3i,1,4,1]);
        let mut a = [1i,2,3,4];
        let b = vec![5i,6,7,8,9,0];
        assert_eq!(a.move_from(b, 2, 3), 1);
        assert!(a == [7i,2,3,4]);
        let mut a = [1i,2,3,4,5];
        let b = vec![5i,6,7,8,9,0];
        assert_eq!(a[mut 2..4].move_from(b,1,6), 2);
        assert!(a == [1i,2,6,7,5]);
    }

    #[test]
    fn test_reverse_part() {
        let mut values = [1i,2,3,4,5];
        values[mut 1..4].reverse();
        assert!(values == [1,4,3,2,5]);
    }

    #[test]
    fn test_show() {
        macro_rules! test_show_vec(
            ($x:expr, $x_str:expr) => ({
                let (x, x_str) = ($x, $x_str);
                assert_eq!(format!("{}", x), x_str);
                assert_eq!(format!("{}", x.as_slice()), x_str);
            })
        );
        let empty: Vec<int> = vec![];
        test_show_vec!(empty, "[]");
        test_show_vec!(vec![1i], "[1]");
        test_show_vec!(vec![1i, 2, 3], "[1, 2, 3]");
        test_show_vec!(vec![vec![], vec![1u], vec![1u, 1u]],
                       "[[], [1], [1, 1]]");

        let empty_mut: &mut [int] = &mut[];
        test_show_vec!(empty_mut, "[]");
        let v: &mut[int] = &mut[1];
        test_show_vec!(v, "[1]");
        let v: &mut[int] = &mut[1, 2, 3];
        test_show_vec!(v, "[1, 2, 3]");
        let v: &mut [&mut[uint]] = &mut[&mut[], &mut[1u], &mut[1u, 1u]];
        test_show_vec!(v, "[[], [1], [1, 1]]");
    }

    #[test]
    fn test_vec_default() {
        macro_rules! t (
            ($ty:ty) => {{
                let v: $ty = Default::default();
                assert!(v.is_empty());
            }}
        );

        t!(&[int]);
        t!(Vec<int>);
    }

    #[test]
    fn test_bytes_set_memory() {
        use slice::bytes::MutableByteVector;
        let mut values = [1u8,2,3,4,5];
        values[mut 0..5].set_memory(0xAB);
        assert!(values == [0xAB, 0xAB, 0xAB, 0xAB, 0xAB]);
        values[mut 2..4].set_memory(0xFF);
        assert!(values == [0xAB, 0xAB, 0xFF, 0xFF, 0xAB]);
    }

    #[test]
    #[should_fail]
    fn test_overflow_does_not_cause_segfault() {
        let mut v = vec![];
        v.reserve_exact(-1);
        v.push(1i);
        v.push(2);
    }

    #[test]
    #[should_fail]
    fn test_overflow_does_not_cause_segfault_managed() {
        let mut v = vec![Rc::new(1i)];
        v.reserve_exact(-1);
        v.push(Rc::new(2i));
    }

    #[test]
    fn test_mut_split_at() {
        let mut values = [1u8,2,3,4,5];
        {
            let (left, right) = values.split_at_mut(2);
            {
                let left: &[_] = left;
                assert!(left[0..left.len()] == [1, 2][]);
            }
            for p in left.iter_mut() {
                *p += 1;
            }

            {
                let right: &[_] = right;
                assert!(right[0..right.len()] == [3, 4, 5][]);
            }
            for p in right.iter_mut() {
                *p += 2;
            }
        }

        assert!(values == [2, 3, 5, 6, 7]);
    }

    #[deriving(Clone, PartialEq)]
    struct Foo;

    #[test]
    fn test_iter_zero_sized() {
        let mut v = vec![Foo, Foo, Foo];
        assert_eq!(v.len(), 3);
        let mut cnt = 0u;

        for f in v.iter() {
            assert!(*f == Foo);
            cnt += 1;
        }
        assert_eq!(cnt, 3);

        for f in v[1..3].iter() {
            assert!(*f == Foo);
            cnt += 1;
        }
        assert_eq!(cnt, 5);

        for f in v.iter_mut() {
            assert!(*f == Foo);
            cnt += 1;
        }
        assert_eq!(cnt, 8);

        for f in v.into_iter() {
            assert!(f == Foo);
            cnt += 1;
        }
        assert_eq!(cnt, 11);

        let xs: [Foo, ..3] = [Foo, Foo, Foo];
        cnt = 0;
        for f in xs.iter() {
            assert!(*f == Foo);
            cnt += 1;
        }
        assert!(cnt == 3);
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut xs = vec![0, 1, 2, 3];
        for i in range(4i, 100) {
            xs.push(i)
        }
        assert_eq!(xs.capacity(), 128);
        xs.shrink_to_fit();
        assert_eq!(xs.capacity(), 100);
        assert_eq!(xs, range(0i, 100i).collect::<Vec<_>>());
    }

    #[test]
    fn test_starts_with() {
        assert!(b"foobar".starts_with(b"foo"));
        assert!(!b"foobar".starts_with(b"oob"));
        assert!(!b"foobar".starts_with(b"bar"));
        assert!(!b"foo".starts_with(b"foobar"));
        assert!(!b"bar".starts_with(b"foobar"));
        assert!(b"foobar".starts_with(b"foobar"));
        let empty: &[u8] = &[];
        assert!(empty.starts_with(empty));
        assert!(!empty.starts_with(b"foo"));
        assert!(b"foobar".starts_with(empty));
    }

    #[test]
    fn test_ends_with() {
        assert!(b"foobar".ends_with(b"bar"));
        assert!(!b"foobar".ends_with(b"oba"));
        assert!(!b"foobar".ends_with(b"foo"));
        assert!(!b"foo".ends_with(b"foobar"));
        assert!(!b"bar".ends_with(b"foobar"));
        assert!(b"foobar".ends_with(b"foobar"));
        let empty: &[u8] = &[];
        assert!(empty.ends_with(empty));
        assert!(!empty.ends_with(b"foo"));
        assert!(b"foobar".ends_with(empty));
    }

    #[test]
    fn test_mut_splitator() {
        let mut xs = [0i,1,0,2,3,0,0,4,5,0];
        assert_eq!(xs.split_mut(|x| *x == 0).count(), 6);
        for slice in xs.split_mut(|x| *x == 0) {
            slice.reverse();
        }
        assert!(xs == [0,1,0,3,2,0,0,5,4,0]);

        let mut xs = [0i,1,0,2,3,0,0,4,5,0,6,7];
        for slice in xs.split_mut(|x| *x == 0).take(5) {
            slice.reverse();
        }
        assert!(xs == [0,1,0,3,2,0,0,5,4,0,6,7]);
    }

    #[test]
    fn test_mut_splitator_rev() {
        let mut xs = [1i,2,0,3,4,0,0,5,6,0];
        for slice in xs.split_mut(|x| *x == 0).rev().take(4) {
            slice.reverse();
        }
        assert!(xs == [1,2,0,4,3,0,0,6,5,0]);
    }

    #[test]
    fn test_get_mut() {
        let mut v = [0i,1,2];
        assert_eq!(v.get_mut(3), None);
        v.get_mut(1).map(|e| *e = 7);
        assert_eq!(v[1], 7);
        let mut x = 2;
        assert_eq!(v.get_mut(2), Some(&mut x));
    }

    #[test]
    fn test_mut_chunks() {
        let mut v = [0u8, 1, 2, 3, 4, 5, 6];
        for (i, chunk) in v.chunks_mut(3).enumerate() {
            for x in chunk.iter_mut() {
                *x = i as u8;
            }
        }
        let result = [0u8, 0, 0, 1, 1, 1, 2];
        assert!(v == result);
    }

    #[test]
    fn test_mut_chunks_rev() {
        let mut v = [0u8, 1, 2, 3, 4, 5, 6];
        for (i, chunk) in v.chunks_mut(3).rev().enumerate() {
            for x in chunk.iter_mut() {
                *x = i as u8;
            }
        }
        let result = [2u8, 2, 2, 1, 1, 1, 0];
        assert!(v == result);
    }

    #[test]
    #[should_fail]
    fn test_mut_chunks_0() {
        let mut v = [1i, 2, 3, 4];
        let _it = v.chunks_mut(0);
    }

    #[test]
    fn test_mut_last() {
        let mut x = [1i, 2, 3, 4, 5];
        let h = x.last_mut();
        assert_eq!(*h.unwrap(), 5);

        let y: &mut [int] = &mut [];
        assert!(y.last_mut().is_none());
    }

    #[test]
    fn test_to_vec() {
        let xs = box [1u, 2, 3];
        let ys = xs.to_vec();
        assert_eq!(ys, [1u, 2, 3]);
    }
}

#[cfg(test)]
mod bench {
    use prelude::*;
    use core::mem;
    use core::ptr;
    use std::rand::{weak_rng, Rng};
    use test::{Bencher, black_box};

    #[bench]
    fn iterator(b: &mut Bencher) {
        // peculiar numbers to stop LLVM from optimising the summation
        // out.
        let v = Vec::from_fn(100, |i| i ^ (i << 1) ^ (i >> 1));

        b.iter(|| {
            let mut sum = 0;
            for x in v.iter() {
                sum += *x;
            }
            // sum == 11806, to stop dead code elimination.
            if sum == 0 {panic!()}
        })
    }

    #[bench]
    fn mut_iterator(b: &mut Bencher) {
        let mut v = Vec::from_elem(100, 0i);

        b.iter(|| {
            let mut i = 0i;
            for x in v.iter_mut() {
                *x = i;
                i += 1;
            }
        })
    }

    #[bench]
    fn concat(b: &mut Bencher) {
        let xss: Vec<Vec<uint>> =
            Vec::from_fn(100, |i| range(0u, i).collect());
        b.iter(|| {
            xss.as_slice().concat_vec()
        });
    }

    #[bench]
    fn connect(b: &mut Bencher) {
        let xss: Vec<Vec<uint>> =
            Vec::from_fn(100, |i| range(0u, i).collect());
        b.iter(|| {
            xss.as_slice().connect_vec(&0)
        });
    }

    #[bench]
    fn push(b: &mut Bencher) {
        let mut vec: Vec<uint> = vec![];
        b.iter(|| {
            vec.push(0);
            black_box(&vec);
        });
    }

    #[bench]
    fn starts_with_same_vector(b: &mut Bencher) {
        let vec: Vec<uint> = Vec::from_fn(100, |i| i);
        b.iter(|| {
            vec.as_slice().starts_with(vec.as_slice())
        })
    }

    #[bench]
    fn starts_with_single_element(b: &mut Bencher) {
        let vec: Vec<uint> = vec![0];
        b.iter(|| {
            vec.as_slice().starts_with(vec.as_slice())
        })
    }

    #[bench]
    fn starts_with_diff_one_element_at_end(b: &mut Bencher) {
        let vec: Vec<uint> = Vec::from_fn(100, |i| i);
        let mut match_vec: Vec<uint> = Vec::from_fn(99, |i| i);
        match_vec.push(0);
        b.iter(|| {
            vec.as_slice().starts_with(match_vec.as_slice())
        })
    }

    #[bench]
    fn ends_with_same_vector(b: &mut Bencher) {
        let vec: Vec<uint> = Vec::from_fn(100, |i| i);
        b.iter(|| {
            vec.as_slice().ends_with(vec.as_slice())
        })
    }

    #[bench]
    fn ends_with_single_element(b: &mut Bencher) {
        let vec: Vec<uint> = vec![0];
        b.iter(|| {
            vec.as_slice().ends_with(vec.as_slice())
        })
    }

    #[bench]
    fn ends_with_diff_one_element_at_beginning(b: &mut Bencher) {
        let vec: Vec<uint> = Vec::from_fn(100, |i| i);
        let mut match_vec: Vec<uint> = Vec::from_fn(100, |i| i);
        match_vec.as_mut_slice()[0] = 200;
        b.iter(|| {
            vec.as_slice().starts_with(match_vec.as_slice())
        })
    }

    #[bench]
    fn contains_last_element(b: &mut Bencher) {
        let vec: Vec<uint> = Vec::from_fn(100, |i| i);
        b.iter(|| {
            vec.contains(&99u)
        })
    }

    #[bench]
    fn zero_1kb_from_elem(b: &mut Bencher) {
        b.iter(|| {
            Vec::from_elem(1024, 0u8)
        });
    }

    #[bench]
    fn zero_1kb_set_memory(b: &mut Bencher) {
        b.iter(|| {
            let mut v: Vec<uint> = Vec::with_capacity(1024);
            unsafe {
                let vp = v.as_mut_ptr();
                ptr::set_memory(vp, 0, 1024);
                v.set_len(1024);
            }
            v
        });
    }

    #[bench]
    fn zero_1kb_loop_set(b: &mut Bencher) {
        b.iter(|| {
            let mut v: Vec<uint> = Vec::with_capacity(1024);
            unsafe {
                v.set_len(1024);
            }
            for i in range(0u, 1024) {
                v[i] = 0;
            }
        });
    }

    #[bench]
    fn zero_1kb_mut_iter(b: &mut Bencher) {
        b.iter(|| {
            let mut v = Vec::with_capacity(1024);
            unsafe {
                v.set_len(1024);
            }
            for x in v.iter_mut() {
                *x = 0i;
            }
            v
        });
    }

    #[bench]
    fn random_inserts(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
                let mut v = Vec::from_elem(30, (0u, 0u));
                for _ in range(0u, 100) {
                    let l = v.len();
                    v.insert(rng.gen::<uint>() % (l + 1),
                             (1, 1));
                }
            })
    }
    #[bench]
    fn random_removes(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
                let mut v = Vec::from_elem(130, (0u, 0u));
                for _ in range(0u, 100) {
                    let l = v.len();
                    v.remove(rng.gen::<uint>() % l);
                }
            })
    }

    #[bench]
    fn sort_random_small(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v = rng.gen_iter::<u64>().take(5).collect::<Vec<u64>>();
            v.as_mut_slice().sort();
        });
        b.bytes = 5 * mem::size_of::<u64>() as u64;
    }

    #[bench]
    fn sort_random_medium(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v = rng.gen_iter::<u64>().take(100).collect::<Vec<u64>>();
            v.as_mut_slice().sort();
        });
        b.bytes = 100 * mem::size_of::<u64>() as u64;
    }

    #[bench]
    fn sort_random_large(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v = rng.gen_iter::<u64>().take(10000).collect::<Vec<u64>>();
            v.as_mut_slice().sort();
        });
        b.bytes = 10000 * mem::size_of::<u64>() as u64;
    }

    #[bench]
    fn sort_sorted(b: &mut Bencher) {
        let mut v = Vec::from_fn(10000, |i| i);
        b.iter(|| {
            v.sort();
        });
        b.bytes = (v.len() * mem::size_of_val(&v[0])) as u64;
    }

    type BigSortable = (u64,u64,u64,u64);

    #[bench]
    fn sort_big_random_small(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v = rng.gen_iter::<BigSortable>().take(5)
                           .collect::<Vec<BigSortable>>();
            v.sort();
        });
        b.bytes = 5 * mem::size_of::<BigSortable>() as u64;
    }

    #[bench]
    fn sort_big_random_medium(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v = rng.gen_iter::<BigSortable>().take(100)
                           .collect::<Vec<BigSortable>>();
            v.sort();
        });
        b.bytes = 100 * mem::size_of::<BigSortable>() as u64;
    }

    #[bench]
    fn sort_big_random_large(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v = rng.gen_iter::<BigSortable>().take(10000)
                           .collect::<Vec<BigSortable>>();
            v.sort();
        });
        b.bytes = 10000 * mem::size_of::<BigSortable>() as u64;
    }

    #[bench]
    fn sort_big_sorted(b: &mut Bencher) {
        let mut v = Vec::from_fn(10000u, |i| (i, i, i, i));
        b.iter(|| {
            v.sort();
        });
        b.bytes = (v.len() * mem::size_of_val(&v[0])) as u64;
    }
}
