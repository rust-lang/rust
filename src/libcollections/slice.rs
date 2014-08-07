// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Utilities for slice manipulation

The `slice` module contains useful code to help work with slice values.
Slices are a view into a block of memory represented as a pointer and a length.

```rust
// slicing a Vec
let vec = vec!(1i, 2, 3);
let int_slice = vec.as_slice();
// coercing an array to a slice
let str_slice: &[&str] = ["one", "two", "three"];
```

Slices are either mutable or shared. The shared slice type is `&[T]`,
while the mutable slice type is `&mut[T]`. For example, you can mutate the
block of memory that a mutable slice points to:

```rust
let x: &mut[int] = [1i, 2, 3];
x[1] = 7;
assert_eq!(x[0], 1);
assert_eq!(x[1], 7);
assert_eq!(x[2], 3);
```

Here are some of the things this module contains:

## Structs

There are several structs that are useful for slices, such as `Items`, which
represents iteration over a slice.

## Traits

A number of traits add methods that allow you to accomplish tasks with slices.
These traits include `ImmutableSlice`, which is defined for `&[T]` types,
and `MutableSlice`, defined for `&mut [T]` types.

An example is the method `.slice(a, b)` that returns an immutable "view" into
a `Vec` or another slice from the index interval `[a, b)`:

```rust
let numbers = [0i, 1i, 2i];
let last_numbers = numbers.slice(1, 3);
// last_numbers is now &[1i, 2i]
```

## Implementations of other traits

There are several implementations of common traits for slices. Some examples
include:

* `Clone`
* `Eq`, `Ord` - for immutable slices whose element type are `Eq` or `Ord`.
* `Hash` - for slices whose element type is `Hash`

## Iteration

The method `iter()` returns an iteration value for a slice. The iterator
yields references to the slice's elements, so if the element
type of the slice is `int`, the element type of the iterator is `&int`.

```rust
let numbers = [0i, 1i, 2i];
for &x in numbers.iter() {
    println!("{} is a number!", x);
}
```

* `.mut_iter()` returns an iterator that allows modifying each value.
* Further iterators exist that split, chunk or permute the slice.

*/

#![doc(primitive = "slice")]

use core::prelude::*;

use core::cmp;
use core::mem::size_of;
use core::mem;
use core::ptr;
use core::iter::{range_step, MultiplicativeIterator};

use {Collection, MutableSeq};
use vec::Vec;

pub use core::slice::{ref_slice, mut_ref_slice, Splits, Windows};
pub use core::slice::{Chunks, Slice, ImmutableSlice, ImmutablePartialEqSlice};
pub use core::slice::{ImmutableOrdSlice, MutableSlice, Items, MutItems};
pub use core::slice::{MutSplits, MutChunks};
pub use core::slice::{bytes, MutableCloneableSlice};
pub use core::slice::{BinarySearchResult, Found, NotFound};

// Functional utilities

#[allow(missing_doc)]
pub trait VectorVector<T> {
    // FIXME #5898: calling these .concat and .connect conflicts with
    // StrVector::con{cat,nect}, since they have generic contents.
    /// Flattens a vector of vectors of T into a single vector of T.
    fn concat_vec(&self) -> Vec<T>;

    /// Concatenate a vector of vectors, placing a given separator between each.
    fn connect_vec(&self, sep: &T) -> Vec<T>;
}

impl<'a, T: Clone, V: Slice<T>> VectorVector<T> for &'a [V] {
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

/// An Iterator that yields the element swaps needed to produce
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
    /// If true, emit the last swap that returns the sequence to initial state
    emit_reset: bool,
    swaps_made : uint,
}

impl ElementSwaps {
    /// Create an `ElementSwaps` iterator for a sequence of `length` elements
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

enum Direction { Pos, Neg }

/// An Index and Direction together
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
                self.sdir.as_mut_slice().swap(i, j);

                // Swap the direction of each larger SizeDirection
                for x in self.sdir.mut_iter() {
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

/// An Iterator that uses `ElementSwaps` to iterate through
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
                self.v.as_mut_slice().swap(a, b);
                Some(elt)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.swaps.size_hint()
    }
}

/// Extension methods for vector slices with cloneable elements
pub trait CloneableVector<T> {
    /// Copy `self` into a new vector
    fn to_vec(&self) -> Vec<T>;

    /// Deprecated. Use `to_vec`
    #[deprecated = "Replaced by `to_vec`"]
    fn to_owned(&self) -> Vec<T> {
        self.to_vec()
    }

    /// Convert `self` into an owned vector, not making a copy if possible.
    fn into_vec(self) -> Vec<T>;

    /// Deprecated. Use `into_vec`
    #[deprecated = "Replaced by `into_vec`"]
    fn into_owned(self) -> Vec<T> {
        self.into_vec()
    }
}

/// Extension methods for vector slices
impl<'a, T: Clone> CloneableVector<T> for &'a [T] {
    /// Returns a copy of `v`.
    #[inline]
    fn to_vec(&self) -> Vec<T> { Vec::from_slice(*self) }

    #[inline(always)]
    fn into_vec(self) -> Vec<T> { self.to_vec() }
}

/// Extension methods for vectors containing `Clone` elements.
pub trait ImmutableCloneableVector<T> {
    /// Partitions the vector into two vectors `(A,B)`, where all
    /// elements of `A` satisfy `f` and all elements of `B` do not.
    fn partitioned(&self, f: |&T| -> bool) -> (Vec<T>, Vec<T>);

    /// Create an iterator that yields every possible permutation of the
    /// vector in succession.
    ///
    /// # Example
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
    /// # Example 2: iterating through permutations one by one.
    ///
    /// ```rust
    /// let v = [1i, 2, 3];
    /// let mut perms = v.permutations();
    ///
    /// assert_eq!(Some(vec![1i, 2, 3]), perms.next());
    /// assert_eq!(Some(vec![1i, 3, 2]), perms.next());
    /// assert_eq!(Some(vec![3i, 1, 2]), perms.next());
    /// ```
    fn permutations(self) -> Permutations<T>;
}

impl<'a,T:Clone> ImmutableCloneableVector<T> for &'a [T] {
    #[inline]
    fn partitioned(&self, f: |&T| -> bool) -> (Vec<T>, Vec<T>) {
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
    fn permutations(self) -> Permutations<T> {
        Permutations{
            swaps: ElementSwaps::new(self.len()),
            v: self.to_vec(),
        }
    }

}

fn insertion_sort<T>(v: &mut [T], compare: |&T, &T| -> Ordering) {
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

fn merge_sort<T>(v: &mut [T], compare: |&T, &T| -> Ordering) {
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
    // `compare` fails.
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

/// Extension methods for vectors such that their elements are
/// mutable.
pub trait MutableSliceAllocating<'a, T> {
    /// Sort the vector, in place, using `compare` to compare
    /// elements.
    ///
    /// This sort is `O(n log n)` worst-case and stable, but allocates
    /// approximately `2 * n`, where `n` is the length of `self`.
    ///
    /// # Example
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
    fn sort_by(self, compare: |&T, &T| -> Ordering);

    /**
     * Consumes `src` and moves as many elements as it can into `self`
     * from the range [start,end).
     *
     * Returns the number of elements copied (the shorter of self.len()
     * and end - start).
     *
     * # Arguments
     *
     * * src - A mutable vector of `T`
     * * start - The index into `src` to start copying from
     * * end - The index into `src` to stop copying from
     *
     * # Example
     *
     * ```rust
     * let mut a = [1i, 2, 3, 4, 5];
     * let b = vec![6i, 7, 8];
     * let num_moved = a.move_from(b, 0, 3);
     * assert_eq!(num_moved, 3);
     * assert!(a == [6i, 7, 8, 4, 5]);
     * ```
     */
    fn move_from(self, src: Vec<T>, start: uint, end: uint) -> uint;
}

impl<'a,T> MutableSliceAllocating<'a, T> for &'a mut [T] {
    #[inline]
    fn sort_by(self, compare: |&T, &T| -> Ordering) {
        merge_sort(self, compare)
    }

    #[inline]
    fn move_from(self, mut src: Vec<T>, start: uint, end: uint) -> uint {
        for (a, b) in self.mut_iter().zip(src.mut_slice(start, end).mut_iter()) {
            mem::swap(a, b);
        }
        cmp::min(self.len(), end-start)
    }
}

/// Methods for mutable vectors with orderable elements, such as
/// in-place sorting.
pub trait MutableOrdSlice<T> {
    /// Sort the vector, in place.
    ///
    /// This is equivalent to `self.sort_by(|a, b| a.cmp(b))`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = [-5i, 4, 1, -3, 2];
    ///
    /// v.sort();
    /// assert!(v == [-5i, -3, 1, 2, 4]);
    /// ```
    fn sort(self);

    /// Mutates the slice to the next lexicographic permutation.
    ///
    /// Returns `true` if successful, `false` if the slice is at the last-ordered permutation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v = &mut [0i, 1, 2];
    /// v.next_permutation();
    /// assert_eq!(v, &mut [0i, 2, 1]);
    /// v.next_permutation();
    /// assert_eq!(v, &mut [1i, 0, 2]);
    /// ```
    fn next_permutation(self) -> bool;

    /// Mutates the slice to the previous lexicographic permutation.
    ///
    /// Returns `true` if successful, `false` if the slice is at the first-ordered permutation.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v = &mut [1i, 0, 2];
    /// v.prev_permutation();
    /// assert_eq!(v, &mut [0i, 2, 1]);
    /// v.prev_permutation();
    /// assert_eq!(v, &mut [0i, 1, 2]);
    /// ```
    fn prev_permutation(self) -> bool;
}

impl<'a, T: Ord> MutableOrdSlice<T> for &'a mut [T] {
    #[inline]
    fn sort(self) {
        self.sort_by(|a,b| a.cmp(b))
    }

    fn next_permutation(self) -> bool {
        // These cases only have 1 permutation each, so we can't do anything.
        if self.len() < 2 { return false; }

        // Step 1: Identify the longest, rightmost weakly decreasing part of the vector
        let mut i = self.len() - 1;
        while i > 0 && self[i-1] >= self[i] {
            i -= 1;
        }

        // If that is the entire vector, this is the last-ordered permutation.
        if i == 0 {
            return false;
        }

        // Step 2: Find the rightmost element larger than the pivot (i-1)
        let mut j = self.len() - 1;
        while j >= i && self[j] <= self[i-1]  {
            j -= 1;
        }

        // Step 3: Swap that element with the pivot
        self.swap(j, i-1);

        // Step 4: Reverse the (previously) weakly decreasing part
        self.mut_slice_from(i).reverse();

        true
    }

    fn prev_permutation(self) -> bool {
        // These cases only have 1 permutation each, so we can't do anything.
        if self.len() < 2 { return false; }

        // Step 1: Identify the longest, rightmost weakly increasing part of the vector
        let mut i = self.len() - 1;
        while i > 0 && self[i-1] <= self[i] {
            i -= 1;
        }

        // If that is the entire vector, this is the first-ordered permutation.
        if i == 0 {
            return false;
        }

        // Step 2: Reverse the weakly increasing part
        self.mut_slice_from(i).reverse();

        // Step 3: Find the rightmost element equal to or bigger than the pivot (i-1)
        let mut j = self.len() - 1;
        while j >= i && self[j-1] < self[i-1]  {
            j -= 1;
        }

        // Step 4: Swap that element with the pivot
        self.swap(i-1, j);

        true
    }
}

/// Unsafe operations
pub mod raw {
    pub use core::slice::raw::{buf_as_slice, mut_buf_as_slice};
    pub use core::slice::raw::{shift_ptr, pop_ptr};
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::default::Default;
    use std::mem;
    use std::prelude::*;
    use std::rand::{Rng, task_rng};
    use std::rc::Rc;
    use std::rt;
    use slice::*;

    use {Mutable, MutableSeq};
    use vec::Vec;

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
    fn test_tail() {
        let mut a = vec![11i];
        assert_eq!(a.tail(), &[]);
        a = vec![11i, 12];
        assert_eq!(a.tail(), &[12]);
    }

    #[test]
    #[should_fail]
    fn test_tail_empty() {
        let a: Vec<int> = vec![];
        a.tail();
    }

    #[test]
    fn test_tailn() {
        let mut a = vec![11i, 12, 13];
        assert_eq!(a.tailn(0), &[11, 12, 13]);
        a = vec![11i, 12, 13];
        assert_eq!(a.tailn(2), &[13]);
    }

    #[test]
    #[should_fail]
    fn test_tailn_empty() {
        let a: Vec<int> = vec![];
        a.tailn(2);
    }

    #[test]
    fn test_init() {
        let mut a = vec![11i];
        assert_eq!(a.init(), &[]);
        a = vec![11i, 12];
        assert_eq!(a.init(), &[11]);
    }

    #[test]
    #[should_fail]
    fn test_init_empty() {
        let a: Vec<int> = vec![];
        a.init();
    }

    #[test]
    fn test_initn() {
        let mut a = vec![11i, 12, 13];
        assert_eq!(a.as_slice().initn(0), &[11, 12, 13]);
        a = vec![11i, 12, 13];
        assert_eq!(a.as_slice().initn(2), &[11]);
    }

    #[test]
    #[should_fail]
    fn test_initn_empty() {
        let a: Vec<int> = vec![];
        a.as_slice().initn(2);
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
    fn test_slice() {
        // Test fixed length vector.
        let vec_fixed = [1i, 2, 3, 4];
        let v_a = vec_fixed.slice(1u, vec_fixed.len()).to_vec();
        assert_eq!(v_a.len(), 3u);
        let v_a = v_a.as_slice();
        assert_eq!(v_a[0], 2);
        assert_eq!(v_a[1], 3);
        assert_eq!(v_a[2], 4);

        // Test on stack.
        let vec_stack = &[1i, 2, 3];
        let v_b = vec_stack.slice(1u, 3u).to_vec();
        assert_eq!(v_b.len(), 2u);
        let v_b = v_b.as_slice();
        assert_eq!(v_b[0], 2);
        assert_eq!(v_b[1], 3);

        // Test `Box<[T]>`
        let vec_unique = vec![1i, 2, 3, 4, 5, 6];
        let v_d = vec_unique.slice(1u, 6u).to_vec();
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
        let vec = &[1i, 2, 3, 4];
        assert_eq!(vec.slice_from(0), vec);
        assert_eq!(vec.slice_from(2), &[3, 4]);
        assert_eq!(vec.slice_from(4), &[]);
    }

    #[test]
    fn test_slice_to() {
        let vec = &[1i, 2, 3, 4];
        assert_eq!(vec.slice_to(4), vec);
        assert_eq!(vec.slice_to(2), &[1, 2]);
        assert_eq!(vec.slice_to(0), &[]);
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
        let mut v = vec![rt::exclusive::Exclusive::new(()),
                         rt::exclusive::Exclusive::new(()),
                         rt::exclusive::Exclusive::new(())];
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
        v.grow(2u, &1i);
        {
            let v = v.as_slice();
            assert_eq!(v.len(), 2u);
            assert_eq!(v[0], 1);
            assert_eq!(v[1], 1);
        }

        // Test on-heap grow().
        v.grow(3u, &2i);
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
    fn test_grow_set() {
        let mut v = vec![1i, 2, 3];
        v.grow_set(4u, &4, 5);
        let v = v.as_slice();
        assert_eq!(v.len(), 5u);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);
        assert_eq!(v[3], 4);
        assert_eq!(v[4], 5);
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
         * and/or rustrt should raise errors.
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
         * rustrt should raise errors.
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
                _ => fail!(),
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
        assert_eq!(v, &mut[1, 2, 3, 5, 4]);
        assert!(v.prev_permutation());
        assert_eq!(v, &mut[1, 2, 3, 4, 5]);
        assert!(v.next_permutation());
        assert!(v.next_permutation());
        assert_eq!(v, &mut[1, 2, 4, 3, 5]);
        assert!(v.next_permutation());
        assert_eq!(v, &mut[1, 2, 4, 5, 3]);

        let v : &mut[int] = &mut[1i, 0, 0, 0];
        assert!(v.next_permutation() == false);
        assert!(v.prev_permutation());
        assert_eq!(v, &mut[0, 1, 0, 0]);
        assert!(v.prev_permutation());
        assert_eq!(v, &mut[0, 0, 1, 0]);
        assert!(v.prev_permutation());
        assert_eq!(v, &mut[0, 0, 0, 1]);
        assert!(v.prev_permutation() == false);
    }

    #[test]
    fn test_lexicographic_permutations_empty_and_short() {
        let empty : &mut[int] = &mut[];
        assert!(empty.next_permutation() == false);
        assert_eq!(empty, &mut[]);
        assert!(empty.prev_permutation() == false);
        assert_eq!(empty, &mut[]);

        let one_elem : &mut[int] = &mut[4i];
        assert!(one_elem.prev_permutation() == false);
        assert_eq!(one_elem, &mut[4]);
        assert!(one_elem.next_permutation() == false);
        assert_eq!(one_elem, &mut[4]);

        let two_elem : &mut[int] = &mut[1i, 2];
        assert!(two_elem.prev_permutation() == false);
        assert_eq!(two_elem, &mut[1, 2]);
        assert!(two_elem.next_permutation());
        assert_eq!(two_elem, &mut[2, 1]);
        assert!(two_elem.next_permutation() == false);
        assert_eq!(two_elem, &mut[2, 1]);
        assert!(two_elem.prev_permutation());
        assert_eq!(two_elem, &mut[1, 2]);
        assert!(two_elem.prev_permutation() == false);
        assert_eq!(two_elem, &mut[1, 2]);
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
    fn test_bsearch_elem() {
        assert_eq!([1i,2,3,4,5].bsearch_elem(&5), Some(4));
        assert_eq!([1i,2,3,4,5].bsearch_elem(&4), Some(3));
        assert_eq!([1i,2,3,4,5].bsearch_elem(&3), Some(2));
        assert_eq!([1i,2,3,4,5].bsearch_elem(&2), Some(1));
        assert_eq!([1i,2,3,4,5].bsearch_elem(&1), Some(0));

        assert_eq!([2i,4,6,8,10].bsearch_elem(&1), None);
        assert_eq!([2i,4,6,8,10].bsearch_elem(&5), None);
        assert_eq!([2i,4,6,8,10].bsearch_elem(&4), Some(1));
        assert_eq!([2i,4,6,8,10].bsearch_elem(&10), Some(4));

        assert_eq!([2i,4,6,8].bsearch_elem(&1), None);
        assert_eq!([2i,4,6,8].bsearch_elem(&5), None);
        assert_eq!([2i,4,6,8].bsearch_elem(&4), Some(1));
        assert_eq!([2i,4,6,8].bsearch_elem(&8), Some(3));

        assert_eq!([2i,4,6].bsearch_elem(&1), None);
        assert_eq!([2i,4,6].bsearch_elem(&5), None);
        assert_eq!([2i,4,6].bsearch_elem(&4), Some(1));
        assert_eq!([2i,4,6].bsearch_elem(&6), Some(2));

        assert_eq!([2i,4].bsearch_elem(&1), None);
        assert_eq!([2i,4].bsearch_elem(&5), None);
        assert_eq!([2i,4].bsearch_elem(&2), Some(0));
        assert_eq!([2i,4].bsearch_elem(&4), Some(1));

        assert_eq!([2i].bsearch_elem(&1), None);
        assert_eq!([2i].bsearch_elem(&5), None);
        assert_eq!([2i].bsearch_elem(&2), Some(0));

        assert_eq!([].bsearch_elem(&1i), None);
        assert_eq!([].bsearch_elem(&5i), None);

        assert!([1i,1,1,1,1].bsearch_elem(&1) != None);
        assert!([1i,1,1,1,2].bsearch_elem(&1) != None);
        assert!([1i,1,1,2,2].bsearch_elem(&1) != None);
        assert!([1i,1,2,2,2].bsearch_elem(&1) != None);
        assert_eq!([1i,2,2,2,2].bsearch_elem(&1), Some(0));

        assert_eq!([1i,2,3,4,5].bsearch_elem(&6), None);
        assert_eq!([1i,2,3,4,5].bsearch_elem(&0), None);
    }

    #[test]
    fn test_reverse() {
        let mut v: Vec<int> = vec![10i, 20];
        assert_eq!(*v.get(0), 10);
        assert_eq!(*v.get(1), 20);
        v.reverse();
        assert_eq!(*v.get(0), 20);
        assert_eq!(*v.get(1), 10);

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

                v.as_mut_slice().sort();
                assert!(v.as_slice().windows(2).all(|w| w[0] <= w[1]));

                v1.as_mut_slice().sort_by(|a, b| a.cmp(b));
                assert!(v1.as_slice().windows(2).all(|w| w[0] <= w[1]));

                v1.as_mut_slice().sort_by(|a, b| b.cmp(a));
                assert!(v1.as_slice().windows(2).all(|w| w[0] >= w[1]));
            }
        }

        // shouldn't fail/crash
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

        assert_eq!([&[1i], &[2i,3i]].concat_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn test_connect() {
        let v: [Vec<int>, ..0] = [];
        assert_eq!(v.connect_vec(&0), vec![]);
        assert_eq!([vec![1i], vec![2i, 3]].connect_vec(&0), vec![1, 0, 2, 3]);
        assert_eq!([vec![1i], vec![2i], vec![3i]].connect_vec(&0), vec![1, 0, 2, 0, 3]);

        assert_eq!([&[1i], &[2i, 3]].connect_vec(&0), vec![1, 0, 2, 3]);
        assert_eq!([&[1i], &[2i], &[3]].connect_vec(&0), vec![1, 0, 2, 0, 3]);
    }

    #[test]
    fn test_shift() {
        let mut x = vec![1i, 2, 3];
        assert_eq!(x.shift(), Some(1));
        assert_eq!(&x, &vec![2i, 3]);
        assert_eq!(x.shift(), Some(2));
        assert_eq!(x.shift(), Some(3));
        assert_eq!(x.shift(), None);
        assert_eq!(x.len(), 0);
    }

    #[test]
    fn test_unshift() {
        let mut x = vec![1i, 2, 3];
        x.unshift(0);
        assert_eq!(x, vec![0, 1, 2, 3]);
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
        assert_eq!(v.capacity(), 10u);
        let mut v = vec![0u32];
        v.reserve_exact(10u);
        assert_eq!(v.capacity(), 10u);
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
            if v == 50 { fail!() }
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
                if self.f.get() == 10 { fail!() }
                S { f: self.f, boxes: self.boxes.clone() }
            }
        }

        let s = S { f: Cell::new(0), boxes: (box 0, Rc::new(0)) };
        let _ = Vec::from_elem(100, s);
    }

    #[test]
    #[should_fail]
    fn test_grow_fn_fail() {
        let mut v = vec![];
        v.grow_fn(100, |i| {
            if i == 50 {
                fail!()
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
                fail!()
            }
            i += 1;
        }
    }

    #[test]
    #[should_fail]
    fn test_copy_memory_oob() {
        unsafe {
            let mut a = [1i, 2, 3, 4];
            let b = [1i, 2, 3, 4, 5];
            a.copy_memory(b);
        }
    }

    #[test]
    fn test_total_ord() {
        [1i, 2, 3, 4].cmp(& &[1, 2, 3]) == Greater;
        [1i, 2, 3].cmp(& &[1, 2, 3, 4]) == Less;
        [1i, 2, 3, 4].cmp(& &[1, 2, 3, 4]) == Equal;
        [1i, 2, 3, 4, 5, 5, 5, 5].cmp(& &[1, 2, 3, 4, 5, 6]) == Less;
        [2i, 2].cmp(& &[1, 2, 3, 4]) == Greater;
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
        assert_eq!(xs.mut_iter().size_hint(), (5, Some(5)));
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
        for x in xs.mut_iter() {
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
        for (i,x) in xs.mut_iter().rev().enumerate() {
            *x += i;
        }
        assert!(xs == [5, 5, 5, 5, 5])
    }

    #[test]
    fn test_move_iterator() {
        let xs = vec![1u,2,3,4,5];
        assert_eq!(xs.move_iter().fold(0, |a: uint, b: uint| 10*a + b), 12345);
    }

    #[test]
    fn test_move_rev_iterator() {
        let xs = vec![1u,2,3,4,5];
        assert_eq!(xs.move_iter().rev().fold(0, |a: uint, b: uint| 10*a + b), 54321);
    }

    #[test]
    fn test_splitator() {
        let xs = &[1i,2,3,4,5];

        assert_eq!(xs.split(|x| *x % 2 == 0).collect::<Vec<&[int]>>().as_slice(),
                   &[&[1], &[3], &[5]]);
        assert_eq!(xs.split(|x| *x == 1).collect::<Vec<&[int]>>().as_slice(),
                   &[&[], &[2,3,4,5]]);
        assert_eq!(xs.split(|x| *x == 5).collect::<Vec<&[int]>>().as_slice(),
                   &[&[1,2,3,4], &[]]);
        assert_eq!(xs.split(|x| *x == 10).collect::<Vec<&[int]>>().as_slice(),
                   &[&[1,2,3,4,5]]);
        assert_eq!(xs.split(|_| true).collect::<Vec<&[int]>>().as_slice(),
                   &[&[], &[], &[], &[], &[], &[]]);

        let xs: &[int] = &[];
        assert_eq!(xs.split(|x| *x == 5).collect::<Vec<&[int]>>().as_slice(), &[&[]]);
    }

    #[test]
    fn test_splitnator() {
        let xs = &[1i,2,3,4,5];

        assert_eq!(xs.splitn(0, |x| *x % 2 == 0).collect::<Vec<&[int]>>().as_slice(),
                   &[&[1,2,3,4,5]]);
        assert_eq!(xs.splitn(1, |x| *x % 2 == 0).collect::<Vec<&[int]>>().as_slice(),
                   &[&[1], &[3,4,5]]);
        assert_eq!(xs.splitn(3, |_| true).collect::<Vec<&[int]>>().as_slice(),
                   &[&[], &[], &[], &[4,5]]);

        let xs: &[int] = &[];
        assert_eq!(xs.splitn(1, |x| *x == 5).collect::<Vec<&[int]>>().as_slice(), &[&[]]);
    }

    #[test]
    fn test_rsplitator() {
        let xs = &[1i,2,3,4,5];

        assert_eq!(xs.split(|x| *x % 2 == 0).rev().collect::<Vec<&[int]>>().as_slice(),
                   &[&[5], &[3], &[1]]);
        assert_eq!(xs.split(|x| *x == 1).rev().collect::<Vec<&[int]>>().as_slice(),
                   &[&[2,3,4,5], &[]]);
        assert_eq!(xs.split(|x| *x == 5).rev().collect::<Vec<&[int]>>().as_slice(),
                   &[&[], &[1,2,3,4]]);
        assert_eq!(xs.split(|x| *x == 10).rev().collect::<Vec<&[int]>>().as_slice(),
                   &[&[1,2,3,4,5]]);

        let xs: &[int] = &[];
        assert_eq!(xs.split(|x| *x == 5).rev().collect::<Vec<&[int]>>().as_slice(), &[&[]]);
    }

    #[test]
    fn test_rsplitnator() {
        let xs = &[1,2,3,4,5];

        assert_eq!(xs.rsplitn(0, |x| *x % 2 == 0).collect::<Vec<&[int]>>().as_slice(),
                   &[&[1,2,3,4,5]]);
        assert_eq!(xs.rsplitn(1, |x| *x % 2 == 0).collect::<Vec<&[int]>>().as_slice(),
                   &[&[5], &[1,2,3]]);
        assert_eq!(xs.rsplitn(3, |_| true).collect::<Vec<&[int]>>().as_slice(),
                   &[&[], &[], &[], &[1,2]]);

        let xs: &[int] = &[];
        assert_eq!(xs.rsplitn(1, |x| *x == 5).collect::<Vec<&[int]>>().as_slice(), &[&[]]);
    }

    #[test]
    fn test_windowsator() {
        let v = &[1i,2,3,4];

        assert_eq!(v.windows(2).collect::<Vec<&[int]>>().as_slice(), &[&[1,2], &[2,3], &[3,4]]);
        assert_eq!(v.windows(3).collect::<Vec<&[int]>>().as_slice(), &[&[1i,2,3], &[2,3,4]]);
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

        assert_eq!(v.chunks(2).collect::<Vec<&[int]>>().as_slice(), &[&[1i,2], &[3,4], &[5]]);
        assert_eq!(v.chunks(3).collect::<Vec<&[int]>>().as_slice(), &[&[1i,2,3], &[4,5]]);
        assert_eq!(v.chunks(6).collect::<Vec<&[int]>>().as_slice(), &[&[1i,2,3,4,5]]);

        assert_eq!(v.chunks(2).rev().collect::<Vec<&[int]>>().as_slice(), &[&[5i], &[3,4], &[1,2]]);
        let mut it = v.chunks(2);
        assert_eq!(it.indexable(), 3);
        assert_eq!(it.idx(0).unwrap(), &[1,2]);
        assert_eq!(it.idx(1).unwrap(), &[3,4]);
        assert_eq!(it.idx(2).unwrap(), &[5]);
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
        assert_eq!(a.mut_slice(2,4).move_from(b,1,6), 2);
        assert!(a == [1i,2,6,7,5]);
    }

    #[test]
    fn test_copy_from() {
        let mut a = [1i,2,3,4,5];
        let b = [6i,7,8];
        assert_eq!(a.copy_from(b), 3);
        assert!(a == [6i,7,8,4,5]);
        let mut c = [7i,2,8,1];
        let d = [3i,1,4,1,5,9];
        assert_eq!(c.copy_from(d), 4);
        assert!(c == [3i,1,4,1]);
    }

    #[test]
    fn test_reverse_part() {
        let mut values = [1i,2,3,4,5];
        values.mut_slice(1, 4).reverse();
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
        )
        let empty: Vec<int> = vec![];
        test_show_vec!(empty, "[]".to_string());
        test_show_vec!(vec![1i], "[1]".to_string());
        test_show_vec!(vec![1i, 2, 3], "[1, 2, 3]".to_string());
        test_show_vec!(vec![vec![], vec![1u], vec![1u, 1u]],
                       "[[], [1], [1, 1]]".to_string());

        let empty_mut: &mut [int] = &mut[];
        test_show_vec!(empty_mut, "[]".to_string());
        test_show_vec!(&mut[1i], "[1]".to_string());
        test_show_vec!(&mut[1i, 2, 3], "[1, 2, 3]".to_string());
        test_show_vec!(&mut[&mut[], &mut[1u], &mut[1u, 1u]],
                       "[[], [1], [1, 1]]".to_string());
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
        values.mut_slice(0,5).set_memory(0xAB);
        assert!(values == [0xAB, 0xAB, 0xAB, 0xAB, 0xAB]);
        values.mut_slice(2,4).set_memory(0xFF);
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
            let (left, right) = values.mut_split_at(2);
            assert!(left.slice(0, left.len()) == [1, 2]);
            for p in left.mut_iter() {
                *p += 1;
            }

            assert!(right.slice(0, right.len()) == [3, 4, 5]);
            for p in right.mut_iter() {
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

        for f in v.slice(1, 3).iter() {
            assert!(*f == Foo);
            cnt += 1;
        }
        assert_eq!(cnt, 5);

        for f in v.mut_iter() {
            assert!(*f == Foo);
            cnt += 1;
        }
        assert_eq!(cnt, 8);

        for f in v.move_iter() {
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
        let empty: &[u8] = [];
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
        let empty: &[u8] = [];
        assert!(empty.ends_with(empty));
        assert!(!empty.ends_with(b"foo"));
        assert!(b"foobar".ends_with(empty));
    }

    #[test]
    fn test_shift_ref() {
        let mut x: &[int] = [1, 2, 3, 4, 5];
        let h = x.shift_ref();
        assert_eq!(*h.unwrap(), 1);
        assert_eq!(x.len(), 4);
        assert_eq!(x[0], 2);
        assert_eq!(x[3], 5);

        let mut y: &[int] = [];
        assert_eq!(y.shift_ref(), None);
    }

    #[test]
    fn test_pop_ref() {
        let mut x: &[int] = [1, 2, 3, 4, 5];
        let h = x.pop_ref();
        assert_eq!(*h.unwrap(), 5);
        assert_eq!(x.len(), 4);
        assert_eq!(x[0], 1);
        assert_eq!(x[3], 4);

        let mut y: &[int] = [];
        assert!(y.pop_ref().is_none());
    }

    #[test]
    fn test_mut_splitator() {
        let mut xs = [0i,1,0,2,3,0,0,4,5,0];
        assert_eq!(xs.mut_split(|x| *x == 0).count(), 6);
        for slice in xs.mut_split(|x| *x == 0) {
            slice.reverse();
        }
        assert!(xs == [0,1,0,3,2,0,0,5,4,0]);

        let mut xs = [0i,1,0,2,3,0,0,4,5,0,6,7];
        for slice in xs.mut_split(|x| *x == 0).take(5) {
            slice.reverse();
        }
        assert!(xs == [0,1,0,3,2,0,0,5,4,0,6,7]);
    }

    #[test]
    fn test_mut_splitator_rev() {
        let mut xs = [1i,2,0,3,4,0,0,5,6,0];
        for slice in xs.mut_split(|x| *x == 0).rev().take(4) {
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
        for (i, chunk) in v.mut_chunks(3).enumerate() {
            for x in chunk.mut_iter() {
                *x = i as u8;
            }
        }
        let result = [0u8, 0, 0, 1, 1, 1, 2];
        assert!(v == result);
    }

    #[test]
    fn test_mut_chunks_rev() {
        let mut v = [0u8, 1, 2, 3, 4, 5, 6];
        for (i, chunk) in v.mut_chunks(3).rev().enumerate() {
            for x in chunk.mut_iter() {
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
        let _it = v.mut_chunks(0);
    }

    #[test]
    fn test_mut_shift_ref() {
        let mut x: &mut [int] = [1, 2, 3, 4, 5];
        let h = x.mut_shift_ref();
        assert_eq!(*h.unwrap(), 1);
        assert_eq!(x.len(), 4);
        assert_eq!(x[0], 2);
        assert_eq!(x[3], 5);

        let mut y: &mut [int] = [];
        assert!(y.mut_shift_ref().is_none());
    }

    #[test]
    fn test_mut_pop_ref() {
        let mut x: &mut [int] = [1, 2, 3, 4, 5];
        let h = x.mut_pop_ref();
        assert_eq!(*h.unwrap(), 5);
        assert_eq!(x.len(), 4);
        assert_eq!(x[0], 1);
        assert_eq!(x[3], 4);

        let mut y: &mut [int] = [];
        assert!(y.mut_pop_ref().is_none());
    }

    #[test]
    fn test_mut_last() {
        let mut x = [1i, 2, 3, 4, 5];
        let h = x.mut_last();
        assert_eq!(*h.unwrap(), 5);

        let y: &mut [int] = [];
        assert!(y.mut_last().is_none());
    }
}

#[cfg(test)]
mod bench {
    use std::prelude::*;
    use std::rand::{weak_rng, Rng};
    use std::mem;
    use std::ptr;
    use test::Bencher;

    use vec::Vec;
    use MutableSeq;

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
            if sum == 0 {fail!()}
        })
    }

    #[bench]
    fn mut_iterator(b: &mut Bencher) {
        let mut v = Vec::from_elem(100, 0i);

        b.iter(|| {
            let mut i = 0i;
            for x in v.mut_iter() {
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
            &vec
        })
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
                *v.get_mut(i) = 0;
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
            for x in v.mut_iter() {
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
        b.bytes = (v.len() * mem::size_of_val(v.get(0))) as u64;
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
        b.bytes = (v.len() * mem::size_of_val(v.get(0))) as u64;
    }
}
