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

Utilities for vector manipulation

The `vec` module contains useful code to help work with vector values.
Vectors are Rust's list type. Vectors contain zero or more values of
homogeneous types:

```rust
let int_vector = [1,2,3];
let str_vector = ["one", "two", "three"];
 ```

This is a big module, but for a high-level overview:

## Structs

Several structs that are useful for vectors, such as `Items`, which
represents iteration over a vector.

## Traits

A number of traits add methods that allow you to accomplish tasks with vectors.

Traits defined for the `&[T]` type (a vector slice), have methods that can be
called on either owned vectors, denoted `~[T]`, or on vector slices themselves.
These traits include `ImmutableVector`, and `MutableVector` for the `&mut [T]`
case.

An example is the method `.slice(a, b)` that returns an immutable "view" into
a vector or a vector slice from the index interval `[a, b)`:

```rust
let numbers = [0, 1, 2];
let last_numbers = numbers.slice(1, 3);
// last_numbers is now &[1, 2]
 ```

Traits defined for the `~[T]` type, like `OwnedVector`, can only be called
on such vectors. These methods deal with adding elements or otherwise changing
the allocation of the vector.

An example is the method `.push(element)` that will add an element at the end
of the vector:

```rust
let mut numbers = vec![0, 1, 2];
numbers.push(7);
// numbers is now vec![0, 1, 2, 7];
 ```

## Implementations of other traits

Vectors are a very useful type, and so there's several implementations of
traits from other modules. Some notable examples:

* `Clone`
* `Eq`, `Ord`, `TotalEq`, `TotalOrd` -- vectors can be compared,
  if the element type defines the corresponding trait.

## Iteration

The method `iter()` returns an iteration value for a vector or a vector slice.
The iterator yields references to the vector's elements, so if the element
type of the vector is `int`, the element type of the iterator is `&int`.

```rust
let numbers = [0, 1, 2];
for &x in numbers.iter() {
    println!("{} is a number!", x);
}
 ```

* `.rev_iter()` returns an iterator with the same values as `.iter()`,
  but going in the reverse order, starting with the back element.
* `.mut_iter()` returns an iterator that allows modifying each value.
* `.move_iter()` converts an owned vector into an iterator that
  moves out a value from the vector each iteration.
* Further iterators exist that split, chunk or permute the vector.

## Function definitions

There are a number of free functions that create or take vectors, for example:

* Creating a vector, like `from_elem` and `from_fn`
* Creating a vector with a given size: `with_capacity`
* Modifying a vector and returning it, like `append`
* Operations on paired elements, like `unzip`.

*/

use cast;
use cast::transmute;
use ops::Drop;
use clone::Clone;
use container::Container;
use cmp::{Eq, TotalOrd, Ordering, Less, Equal, Greater};
use cmp;
use default::Default;
use fmt;
use iter::*;
use num::{CheckedAdd, Saturating, div_rem};
use num::CheckedMul;
use option::{None, Option, Some};
use ptr;
use ptr::RawPtr;
use rt::global_heap::{malloc_raw, exchange_free};
use result::{Ok, Err};
use mem;
use mem::size_of;
use kinds::marker;
use uint;
use unstable::finally::try_finally;
use raw::{Repr, Slice};
use RawVec = raw::Vec;
use vec::Vec;

/**
 * Converts a pointer to A into a slice of length 1 (without copying).
 */
pub fn ref_slice<'a, A>(s: &'a A) -> &'a [A] {
    unsafe {
        transmute(Slice { data: s, len: 1 })
    }
}

/**
 * Converts a pointer to A into a slice of length 1 (without copying).
 */
pub fn mut_ref_slice<'a, A>(s: &'a mut A) -> &'a mut [A] {
    unsafe {
        let ptr: *A = transmute(s);
        transmute(Slice { data: ptr, len: 1 })
    }
}

/// An iterator over the slices of a vector separated by elements that
/// match a predicate function.
pub struct Splits<'a, T> {
    v: &'a [T],
    n: uint,
    pred: |t: &T|: 'a -> bool,
    finished: bool
}

impl<'a, T> Iterator<&'a [T]> for Splits<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.finished { return None; }

        if self.n == 0 {
            self.finished = true;
            return Some(self.v);
        }

        match self.v.iter().position(|x| (self.pred)(x)) {
            None => {
                self.finished = true;
                Some(self.v)
            }
            Some(idx) => {
                let ret = Some(self.v.slice(0, idx));
                self.v = self.v.slice(idx + 1, self.v.len());
                self.n -= 1;
                ret
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        if self.finished {
            return (0, Some(0))
        }
        // if the predicate doesn't match anything, we yield one slice
        // if it matches every element, we yield N+1 empty slices where
        // N is either the number of elements or the number of splits.
        match (self.v.len(), self.n) {
            (0,_) => (1, Some(1)),
            (_,0) => (1, Some(1)),
            (l,n) => (1, cmp::min(l,n).checked_add(&1u))
        }
    }
}

/// An iterator over the slices of a vector separated by elements that
/// match a predicate function, from back to front.
pub struct RevSplits<'a, T> {
    v: &'a [T],
    n: uint,
    pred: |t: &T|: 'a -> bool,
    finished: bool
}

impl<'a, T> Iterator<&'a [T]> for RevSplits<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.finished { return None; }

        if self.n == 0 {
            self.finished = true;
            return Some(self.v);
        }

        match self.v.iter().rposition(|x| (self.pred)(x)) {
            None => {
                self.finished = true;
                Some(self.v)
            }
            Some(idx) => {
                let ret = Some(self.v.slice(idx + 1, self.v.len()));
                self.v = self.v.slice(0, idx);
                self.n -= 1;
                ret
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        if self.finished {
            return (0, Some(0))
        }
        match (self.v.len(), self.n) {
            (0,_) => (1, Some(1)),
            (_,0) => (1, Some(1)),
            (l,n) => (1, cmp::min(l,n).checked_add(&1u))
        }
    }
}

// Functional utilities

#[allow(missing_doc)]
pub trait VectorVector<T> {
    // FIXME #5898: calling these .concat and .connect conflicts with
    // StrVector::con{cat,nect}, since they have generic contents.
    /// Flattens a vector of vectors of T into a single vector of T.
    fn concat_vec(&self) -> ~[T];

    /// Concatenate a vector of vectors, placing a given separator between each.
    fn connect_vec(&self, sep: &T) -> ~[T];
}

impl<'a, T: Clone, V: Vector<T>> VectorVector<T> for &'a [V] {
    fn concat_vec(&self) -> ~[T] {
        let size = self.iter().fold(0u, |acc, v| acc + v.as_slice().len());
        let mut result = Vec::with_capacity(size);
        for v in self.iter() {
            result.push_all(v.as_slice())
        }
        result.move_iter().collect()
    }

    fn connect_vec(&self, sep: &T) -> ~[T] {
        let size = self.iter().fold(0u, |acc, v| acc + v.as_slice().len());
        let mut result = Vec::with_capacity(size + self.len());
        let mut first = true;
        for v in self.iter() {
            if first { first = false } else { result.push(sep.clone()) }
            result.push_all(v.as_slice())
        }
        result.move_iter().collect()
    }
}

/**
 * Convert an iterator of pairs into a pair of vectors.
 *
 * Returns a tuple containing two vectors where the i-th element of the first
 * vector contains the first element of the i-th tuple of the input iterator,
 * and the i-th element of the second vector contains the second element
 * of the i-th tuple of the input iterator.
 */
pub fn unzip<T, U, V: Iterator<(T, U)>>(mut iter: V) -> (~[T], ~[U]) {
    let (lo, _) = iter.size_hint();
    let mut ts = Vec::with_capacity(lo);
    let mut us = Vec::with_capacity(lo);
    for (t, u) in iter {
        ts.push(t);
        us.push(u);
    }
    (ts.move_iter().collect(), us.move_iter().collect())
}

/// An Iterator that yields the element swaps needed to produce
/// a sequence of all possible permutations for an indexed sequence of
/// elements. Each permutation is only a single swap apart.
///
/// The Steinhaus–Johnson–Trotter algorithm is used.
///
/// Generates even and odd permutations alternately.
///
/// The last generated swap is always (0, 1), and it returns the
/// sequence to its initial order.
pub struct ElementSwaps {
    sdir: ~[SizeDirection],
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
            sdir: range(0, length)
                    .map(|i| SizeDirection{ size: i, dir: Neg })
                    .collect::<~[_]>(),
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
                self.sdir.swap(i, j);

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
    v: ~[T],
}

impl<T: Clone> Iterator<~[T]> for Permutations<T> {
    #[inline]
    fn next(&mut self) -> Option<~[T]> {
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

/// An iterator over the (overlapping) slices of length `size` within
/// a vector.
#[deriving(Clone)]
pub struct Windows<'a, T> {
    v: &'a [T],
    size: uint
}

impl<'a, T> Iterator<&'a [T]> for Windows<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.size > self.v.len() {
            None
        } else {
            let ret = Some(self.v.slice(0, self.size));
            self.v = self.v.slice(1, self.v.len());
            ret
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        if self.size > self.v.len() {
            (0, Some(0))
        } else {
            let x = self.v.len() - self.size;
            (x.saturating_add(1), x.checked_add(&1u))
        }
    }
}

/// An iterator over a vector in (non-overlapping) chunks (`size`
/// elements at a time).
///
/// When the vector len is not evenly divided by the chunk size,
/// the last slice of the iteration will be the remainder.
#[deriving(Clone)]
pub struct Chunks<'a, T> {
    v: &'a [T],
    size: uint
}

impl<'a, T> Iterator<&'a [T]> for Chunks<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.v.len() == 0 {
            None
        } else {
            let chunksz = cmp::min(self.v.len(), self.size);
            let (fst, snd) = (self.v.slice_to(chunksz),
                              self.v.slice_from(chunksz));
            self.v = snd;
            Some(fst)
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        if self.v.len() == 0 {
            (0, Some(0))
        } else {
            let (n, rem) = div_rem(self.v.len(), self.size);
            let n = if rem > 0 { n+1 } else { n };
            (n, Some(n))
        }
    }
}

impl<'a, T> DoubleEndedIterator<&'a [T]> for Chunks<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.v.len() == 0 {
            None
        } else {
            let remainder = self.v.len() % self.size;
            let chunksz = if remainder != 0 { remainder } else { self.size };
            let (fst, snd) = (self.v.slice_to(self.v.len() - chunksz),
                              self.v.slice_from(self.v.len() - chunksz));
            self.v = fst;
            Some(snd)
        }
    }
}

impl<'a, T> RandomAccessIterator<&'a [T]> for Chunks<'a, T> {
    #[inline]
    fn indexable(&self) -> uint {
        self.v.len()/self.size + if self.v.len() % self.size != 0 { 1 } else { 0 }
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<&'a [T]> {
        if index < self.indexable() {
            let lo = index * self.size;
            let mut hi = lo + self.size;
            if hi < lo || hi > self.v.len() { hi = self.v.len(); }

            Some(self.v.slice(lo, hi))
        } else {
            None
        }
    }
}

// Equality

#[cfg(not(test))]
#[allow(missing_doc)]
pub mod traits {
    use super::*;

    use container::Container;
    use clone::Clone;
    use cmp::{Eq, Ord, TotalEq, TotalOrd, Ordering, Equiv};
    use iter::{order, Iterator};
    use ops::Add;
    use vec::Vec;

    impl<'a,T:Eq> Eq for &'a [T] {
        fn eq(&self, other: & &'a [T]) -> bool {
            self.len() == other.len() &&
                order::eq(self.iter(), other.iter())
        }
        fn ne(&self, other: & &'a [T]) -> bool {
            self.len() != other.len() ||
                order::ne(self.iter(), other.iter())
        }
    }

    impl<T:Eq> Eq for ~[T] {
        #[inline]
        fn eq(&self, other: &~[T]) -> bool { self.as_slice() == *other }
        #[inline]
        fn ne(&self, other: &~[T]) -> bool { !self.eq(other) }
    }

    impl<'a,T:TotalEq> TotalEq for &'a [T] {}

    impl<T:TotalEq> TotalEq for ~[T] {}

    impl<'a,T:Eq, V: Vector<T>> Equiv<V> for &'a [T] {
        #[inline]
        fn equiv(&self, other: &V) -> bool { self.as_slice() == other.as_slice() }
    }

    impl<'a,T:Eq, V: Vector<T>> Equiv<V> for ~[T] {
        #[inline]
        fn equiv(&self, other: &V) -> bool { self.as_slice() == other.as_slice() }
    }

    impl<'a,T:TotalOrd> TotalOrd for &'a [T] {
        fn cmp(&self, other: & &'a [T]) -> Ordering {
            order::cmp(self.iter(), other.iter())
        }
    }

    impl<T: TotalOrd> TotalOrd for ~[T] {
        #[inline]
        fn cmp(&self, other: &~[T]) -> Ordering { self.as_slice().cmp(&other.as_slice()) }
    }

    impl<'a, T: Ord> Ord for &'a [T] {
        fn lt(&self, other: & &'a [T]) -> bool {
            order::lt(self.iter(), other.iter())
        }
        #[inline]
        fn le(&self, other: & &'a [T]) -> bool {
            order::le(self.iter(), other.iter())
        }
        #[inline]
        fn ge(&self, other: & &'a [T]) -> bool {
            order::ge(self.iter(), other.iter())
        }
        #[inline]
        fn gt(&self, other: & &'a [T]) -> bool {
            order::gt(self.iter(), other.iter())
        }
    }

    impl<T: Ord> Ord for ~[T] {
        #[inline]
        fn lt(&self, other: &~[T]) -> bool { self.as_slice() < other.as_slice() }
        #[inline]
        fn le(&self, other: &~[T]) -> bool { self.as_slice() <= other.as_slice() }
        #[inline]
        fn ge(&self, other: &~[T]) -> bool { self.as_slice() >= other.as_slice() }
        #[inline]
        fn gt(&self, other: &~[T]) -> bool { self.as_slice() > other.as_slice() }
    }

    impl<'a,T:Clone, V: Vector<T>> Add<V, ~[T]> for &'a [T] {
        #[inline]
        fn add(&self, rhs: &V) -> ~[T] {
            let mut res = Vec::with_capacity(self.len() + rhs.as_slice().len());
            res.push_all(*self);
            res.push_all(rhs.as_slice());
            res.move_iter().collect()
        }
    }

    impl<T:Clone, V: Vector<T>> Add<V, ~[T]> for ~[T] {
        #[inline]
        fn add(&self, rhs: &V) -> ~[T] {
            self.as_slice() + rhs.as_slice()
        }
    }
}

#[cfg(test)]
pub mod traits {}

/// Any vector that can be represented as a slice.
pub trait Vector<T> {
    /// Work with `self` as a slice.
    fn as_slice<'a>(&'a self) -> &'a [T];
}

impl<'a,T> Vector<T> for &'a [T] {
    #[inline(always)]
    fn as_slice<'a>(&'a self) -> &'a [T] { *self }
}

impl<T> Vector<T> for ~[T] {
    #[inline(always)]
    fn as_slice<'a>(&'a self) -> &'a [T] { let v: &'a [T] = *self; v }
}

impl<'a, T> Container for &'a [T] {
    /// Returns the length of a vector
    #[inline]
    fn len(&self) -> uint {
        self.repr().len
    }
}

impl<T> Container for ~[T] {
    /// Returns the length of a vector
    #[inline]
    fn len(&self) -> uint {
        self.as_slice().len()
    }
}

/// Extension methods for vector slices with cloneable elements
pub trait CloneableVector<T> {
    /// Copy `self` into a new owned vector
    fn to_owned(&self) -> ~[T];

    /// Convert `self` into an owned vector, not making a copy if possible.
    fn into_owned(self) -> ~[T];
}

/// Extension methods for vector slices
impl<'a, T: Clone> CloneableVector<T> for &'a [T] {
    /// Returns a copy of `v`.
    #[inline]
    fn to_owned(&self) -> ~[T] {
        let len = self.len();
        let mut result = Vec::with_capacity(len);
        // Unsafe code so this can be optimised to a memcpy (or something
        // similarly fast) when T is Copy. LLVM is easily confused, so any
        // extra operations during the loop can prevent this optimisation
        unsafe {
            let mut i = 0;
            let p = result.as_mut_ptr();
            // Use try_finally here otherwise the write to length
            // inside the loop stops LLVM from optimising this.
            try_finally(
                &mut i, (),
                |i, ()| while *i < len {
                    mem::move_val_init(
                        &mut(*p.offset(*i as int)),
                        self.unsafe_ref(*i).clone());
                    *i += 1;
                },
                |i| result.set_len(*i));
        }
        result.move_iter().collect()
    }

    #[inline(always)]
    fn into_owned(self) -> ~[T] { self.to_owned() }
}

/// Extension methods for owned vectors
impl<T: Clone> CloneableVector<T> for ~[T] {
    #[inline]
    fn to_owned(&self) -> ~[T] { self.clone() }

    #[inline(always)]
    fn into_owned(self) -> ~[T] { self }
}

/// Extension methods for vectors
pub trait ImmutableVector<'a, T> {
    /**
     * Returns a slice of self between `start` and `end`.
     *
     * Fails when `start` or `end` point outside the bounds of self,
     * or when `start` > `end`.
     */
    fn slice(&self, start: uint, end: uint) -> &'a [T];

    /**
     * Returns a slice of self from `start` to the end of the vec.
     *
     * Fails when `start` points outside the bounds of self.
     */
    fn slice_from(&self, start: uint) -> &'a [T];

    /**
     * Returns a slice of self from the start of the vec to `end`.
     *
     * Fails when `end` points outside the bounds of self.
     */
    fn slice_to(&self, end: uint) -> &'a [T];
    /// Returns an iterator over the vector
    fn iter(self) -> Items<'a, T>;
    /// Returns a reversed iterator over a vector
    fn rev_iter(self) -> RevItems<'a, T>;
    /// Returns an iterator over the subslices of the vector which are
    /// separated by elements that match `pred`.  The matched element
    /// is not contained in the subslices.
    fn split(self, pred: |&T|: 'a -> bool) -> Splits<'a, T>;
    /// Returns an iterator over the subslices of the vector which are
    /// separated by elements that match `pred`, limited to splitting
    /// at most `n` times.  The matched element is not contained in
    /// the subslices.
    fn splitn(self, n: uint, pred: |&T|: 'a -> bool) -> Splits<'a, T>;
    /// Returns an iterator over the subslices of the vector which are
    /// separated by elements that match `pred`. This starts at the
    /// end of the vector and works backwards.  The matched element is
    /// not contained in the subslices.
    fn rsplit(self, pred: |&T|: 'a -> bool) -> RevSplits<'a, T>;
    /// Returns an iterator over the subslices of the vector which are
    /// separated by elements that match `pred` limited to splitting
    /// at most `n` times. This starts at the end of the vector and
    /// works backwards.  The matched element is not contained in the
    /// subslices.
    fn rsplitn(self,  n: uint, pred: |&T|: 'a -> bool) -> RevSplits<'a, T>;

    /**
     * Returns an iterator over all contiguous windows of length
     * `size`. The windows overlap. If the vector is shorter than
     * `size`, the iterator returns no values.
     *
     * # Failure
     *
     * Fails if `size` is 0.
     *
     * # Example
     *
     * Print the adjacent pairs of a vector (i.e. `[1,2]`, `[2,3]`,
     * `[3,4]`):
     *
     * ```rust
     * let v = &[1,2,3,4];
     * for win in v.windows(2) {
     *     println!("{:?}", win);
     * }
     * ```
     *
     */
    fn windows(self, size: uint) -> Windows<'a, T>;
    /**
     *
     * Returns an iterator over `size` elements of the vector at a
     * time. The chunks do not overlap. If `size` does not divide the
     * length of the vector, then the last chunk will not have length
     * `size`.
     *
     * # Failure
     *
     * Fails if `size` is 0.
     *
     * # Example
     *
     * Print the vector two elements at a time (i.e. `[1,2]`,
     * `[3,4]`, `[5]`):
     *
     * ```rust
     * let v = &[1,2,3,4,5];
     * for win in v.chunks(2) {
     *     println!("{:?}", win);
     * }
     * ```
     *
     */
    fn chunks(self, size: uint) -> Chunks<'a, T>;

    /// Returns the element of a vector at the given index, or `None` if the
    /// index is out of bounds
    fn get(&self, index: uint) -> Option<&'a T>;
    /// Returns the first element of a vector, or `None` if it is empty
    fn head(&self) -> Option<&'a T>;
    /// Returns all but the first element of a vector
    fn tail(&self) -> &'a [T];
    /// Returns all but the first `n' elements of a vector
    fn tailn(&self, n: uint) -> &'a [T];
    /// Returns all but the last element of a vector
    fn init(&self) -> &'a [T];
    /// Returns all but the last `n' elements of a vector
    fn initn(&self, n: uint) -> &'a [T];
    /// Returns the last element of a vector, or `None` if it is empty.
    fn last(&self) -> Option<&'a T>;

    /// Returns a pointer to the element at the given index, without doing
    /// bounds checking.
    unsafe fn unsafe_ref(self, index: uint) -> &'a T;

    /**
     * Returns an unsafe pointer to the vector's buffer
     *
     * The caller must ensure that the vector outlives the pointer this
     * function returns, or else it will end up pointing to garbage.
     *
     * Modifying the vector may cause its buffer to be reallocated, which
     * would also make any pointers to it invalid.
     */
    fn as_ptr(&self) -> *T;

    /**
     * Binary search a sorted vector with a comparator function.
     *
     * The comparator function should implement an order consistent
     * with the sort order of the underlying vector, returning an
     * order code that indicates whether its argument is `Less`,
     * `Equal` or `Greater` the desired target.
     *
     * Returns the index where the comparator returned `Equal`, or `None` if
     * not found.
     */
    fn bsearch(&self, f: |&T| -> Ordering) -> Option<uint>;

    /**
     * Returns a mutable reference to the first element in this slice
     * and adjusts the slice in place so that it no longer contains
     * that element. O(1).
     *
     * Equivalent to:
     *
     * ```ignore
     *     if self.len() == 0 { return None }
     *     let head = &self[0];
     *     *self = self.slice_from(1);
     *     Some(head)
     * ```
     *
     * Returns `None` if vector is empty
     */
    fn shift_ref(&mut self) -> Option<&'a T>;

    /**
     * Returns a mutable reference to the last element in this slice
     * and adjusts the slice in place so that it no longer contains
     * that element. O(1).
     *
     * Equivalent to:
     *
     * ```ignore
     *     if self.len() == 0 { return None; }
     *     let tail = &self[self.len() - 1];
     *     *self = self.slice_to(self.len() - 1);
     *     Some(tail)
     * ```
     *
     * Returns `None` if slice is empty.
     */
    fn pop_ref(&mut self) -> Option<&'a T>;
}

impl<'a,T> ImmutableVector<'a, T> for &'a [T] {
    #[inline]
    fn slice(&self, start: uint, end: uint) -> &'a [T] {
        assert!(start <= end);
        assert!(end <= self.len());
        unsafe {
            transmute(Slice {
                    data: self.as_ptr().offset(start as int),
                    len: (end - start)
                })
        }
    }

    #[inline]
    fn slice_from(&self, start: uint) -> &'a [T] {
        self.slice(start, self.len())
    }

    #[inline]
    fn slice_to(&self, end: uint) -> &'a [T] {
        self.slice(0, end)
    }

    #[inline]
    fn iter(self) -> Items<'a, T> {
        unsafe {
            let p = self.as_ptr();
            if mem::size_of::<T>() == 0 {
                Items{ptr: p,
                      end: (p as uint + self.len()) as *T,
                      marker: marker::ContravariantLifetime::<'a>}
            } else {
                Items{ptr: p,
                      end: p.offset(self.len() as int),
                      marker: marker::ContravariantLifetime::<'a>}
            }
        }
    }

    #[inline]
    fn rev_iter(self) -> RevItems<'a, T> {
        self.iter().rev()
    }

    #[inline]
    fn split(self, pred: |&T|: 'a -> bool) -> Splits<'a, T> {
        self.splitn(uint::MAX, pred)
    }

    #[inline]
    fn splitn(self, n: uint, pred: |&T|: 'a -> bool) -> Splits<'a, T> {
        Splits {
            v: self,
            n: n,
            pred: pred,
            finished: false
        }
    }

    #[inline]
    fn rsplit(self, pred: |&T|: 'a -> bool) -> RevSplits<'a, T> {
        self.rsplitn(uint::MAX, pred)
    }

    #[inline]
    fn rsplitn(self, n: uint, pred: |&T|: 'a -> bool) -> RevSplits<'a, T> {
        RevSplits {
            v: self,
            n: n,
            pred: pred,
            finished: false
        }
    }

    #[inline]
    fn windows(self, size: uint) -> Windows<'a, T> {
        assert!(size != 0);
        Windows { v: self, size: size }
    }

    #[inline]
    fn chunks(self, size: uint) -> Chunks<'a, T> {
        assert!(size != 0);
        Chunks { v: self, size: size }
    }

    #[inline]
    fn get(&self, index: uint) -> Option<&'a T> {
        if index < self.len() { Some(&self[index]) } else { None }
    }

    #[inline]
    fn head(&self) -> Option<&'a T> {
        if self.len() == 0 { None } else { Some(&self[0]) }
    }

    #[inline]
    fn tail(&self) -> &'a [T] { self.slice(1, self.len()) }

    #[inline]
    fn tailn(&self, n: uint) -> &'a [T] { self.slice(n, self.len()) }

    #[inline]
    fn init(&self) -> &'a [T] {
        self.slice(0, self.len() - 1)
    }

    #[inline]
    fn initn(&self, n: uint) -> &'a [T] {
        self.slice(0, self.len() - n)
    }

    #[inline]
    fn last(&self) -> Option<&'a T> {
            if self.len() == 0 { None } else { Some(&self[self.len() - 1]) }
    }

    #[inline]
    unsafe fn unsafe_ref(self, index: uint) -> &'a T {
        transmute(self.repr().data.offset(index as int))
    }

    #[inline]
    fn as_ptr(&self) -> *T {
        self.repr().data
    }


    fn bsearch(&self, f: |&T| -> Ordering) -> Option<uint> {
        let mut base : uint = 0;
        let mut lim : uint = self.len();

        while lim != 0 {
            let ix = base + (lim >> 1);
            match f(&self[ix]) {
                Equal => return Some(ix),
                Less => {
                    base = ix + 1;
                    lim -= 1;
                }
                Greater => ()
            }
            lim >>= 1;
        }
        return None;
    }

    fn shift_ref(&mut self) -> Option<&'a T> {
        if self.len() == 0 { return None; }
        unsafe {
            let s: &mut Slice<T> = transmute(self);
            Some(&*raw::shift_ptr(s))
        }
    }

    fn pop_ref(&mut self) -> Option<&'a T> {
        if self.len() == 0 { return None; }
        unsafe {
            let s: &mut Slice<T> = transmute(self);
            Some(&*raw::pop_ptr(s))
        }
    }
}

/// Extension methods for vectors contain `Eq` elements.
pub trait ImmutableEqVector<T:Eq> {
    /// Find the first index containing a matching value
    fn position_elem(&self, t: &T) -> Option<uint>;

    /// Find the last index containing a matching value
    fn rposition_elem(&self, t: &T) -> Option<uint>;

    /// Return true if a vector contains an element with the given value
    fn contains(&self, x: &T) -> bool;

    /// Returns true if `needle` is a prefix of the vector.
    fn starts_with(&self, needle: &[T]) -> bool;

    /// Returns true if `needle` is a suffix of the vector.
    fn ends_with(&self, needle: &[T]) -> bool;
}

impl<'a,T:Eq> ImmutableEqVector<T> for &'a [T] {
    #[inline]
    fn position_elem(&self, x: &T) -> Option<uint> {
        self.iter().position(|y| *x == *y)
    }

    #[inline]
    fn rposition_elem(&self, t: &T) -> Option<uint> {
        self.iter().rposition(|x| *x == *t)
    }

    #[inline]
    fn contains(&self, x: &T) -> bool {
        self.iter().any(|elt| *x == *elt)
    }

    #[inline]
    fn starts_with(&self, needle: &[T]) -> bool {
        let n = needle.len();
        self.len() >= n && needle == self.slice_to(n)
    }

    #[inline]
    fn ends_with(&self, needle: &[T]) -> bool {
        let (m, n) = (self.len(), needle.len());
        m >= n && needle == self.slice_from(m - n)
    }
}

/// Extension methods for vectors containing `TotalOrd` elements.
pub trait ImmutableTotalOrdVector<T: TotalOrd> {
    /**
     * Binary search a sorted vector for a given element.
     *
     * Returns the index of the element or None if not found.
     */
    fn bsearch_elem(&self, x: &T) -> Option<uint>;
}

impl<'a, T: TotalOrd> ImmutableTotalOrdVector<T> for &'a [T] {
    fn bsearch_elem(&self, x: &T) -> Option<uint> {
        self.bsearch(|p| p.cmp(x))
    }
}

/// Extension methods for vectors containing `Clone` elements.
pub trait ImmutableCloneableVector<T> {
    /// Partitions the vector into two vectors `(A,B)`, where all
    /// elements of `A` satisfy `f` and all elements of `B` do not.
    fn partitioned(&self, f: |&T| -> bool) -> (~[T], ~[T]);

    /// Create an iterator that yields every possible permutation of the
    /// vector in succession.
    fn permutations(self) -> Permutations<T>;
}

impl<'a,T:Clone> ImmutableCloneableVector<T> for &'a [T] {
    #[inline]
    fn partitioned(&self, f: |&T| -> bool) -> (~[T], ~[T]) {
        let mut lefts  = Vec::new();
        let mut rights = Vec::new();

        for elt in self.iter() {
            if f(elt) {
                lefts.push((*elt).clone());
            } else {
                rights.push((*elt).clone());
            }
        }

        (lefts.move_iter().collect(), rights.move_iter().collect())
    }

    fn permutations(self) -> Permutations<T> {
        Permutations{
            swaps: ElementSwaps::new(self.len()),
            v: self.to_owned(),
        }
    }

}

/// Extension methods for owned vectors.
pub trait OwnedVector<T> {
    /// Creates a consuming iterator, that is, one that moves each
    /// value out of the vector (from start to end). The vector cannot
    /// be used after calling this.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let v = ~["a".to_owned(), "b".to_owned()];
    /// for s in v.move_iter() {
    ///   // s has type ~str, not &~str
    ///   println!("{}", s);
    /// }
    /// ```
    fn move_iter(self) -> MoveItems<T>;
    /// Creates a consuming iterator that moves out of the vector in
    /// reverse order.
    fn move_rev_iter(self) -> RevMoveItems<T>;

    /**
     * Partitions the vector into two vectors `(A,B)`, where all
     * elements of `A` satisfy `f` and all elements of `B` do not.
     */
    fn partition(self, f: |&T| -> bool) -> (~[T], ~[T]);
}

impl<T> OwnedVector<T> for ~[T] {
    #[inline]
    fn move_iter(self) -> MoveItems<T> {
        unsafe {
            let iter = transmute(self.iter());
            let ptr = transmute(self);
            MoveItems { allocation: ptr, iter: iter }
        }
    }

    #[inline]
    fn move_rev_iter(self) -> RevMoveItems<T> {
        self.move_iter().rev()
    }

    #[inline]
    fn partition(self, f: |&T| -> bool) -> (~[T], ~[T]) {
        let mut lefts  = Vec::new();
        let mut rights = Vec::new();

        for elt in self.move_iter() {
            if f(&elt) {
                lefts.push(elt);
            } else {
                rights.push(elt);
            }
        }

        (lefts.move_iter().collect(), rights.move_iter().collect())
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
            let read_ptr = buf_v.offset(i) as *T;

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
                                                &tmp as *T,
                                                1);
                cast::forget(tmp);
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
pub trait MutableVector<'a, T> {
    /// Work with `self` as a mut slice.
    /// Primarily intended for getting a &mut [T] from a [T, ..N].
    fn as_mut_slice(self) -> &'a mut [T];

    /// Return a slice that points into another slice.
    fn mut_slice(self, start: uint, end: uint) -> &'a mut [T];

    /**
     * Returns a slice of self from `start` to the end of the vec.
     *
     * Fails when `start` points outside the bounds of self.
     */
    fn mut_slice_from(self, start: uint) -> &'a mut [T];

    /**
     * Returns a slice of self from the start of the vec to `end`.
     *
     * Fails when `end` points outside the bounds of self.
     */
    fn mut_slice_to(self, end: uint) -> &'a mut [T];

    /// Returns an iterator that allows modifying each value
    fn mut_iter(self) -> MutItems<'a, T>;

    /// Returns a mutable pointer to the last item in the vector.
    fn mut_last(self) -> Option<&'a mut T>;

    /// Returns a reversed iterator that allows modifying each value
    fn mut_rev_iter(self) -> RevMutItems<'a, T>;

    /// Returns an iterator over the mutable subslices of the vector
    /// which are separated by elements that match `pred`.  The
    /// matched element is not contained in the subslices.
    fn mut_split(self, pred: |&T|: 'a -> bool) -> MutSplits<'a, T>;

    /**
     * Returns an iterator over `size` elements of the vector at a time.
     * The chunks are mutable and do not overlap. If `size` does not divide the
     * length of the vector, then the last chunk will not have length
     * `size`.
     *
     * # Failure
     *
     * Fails if `size` is 0.
     */
    fn mut_chunks(self, chunk_size: uint) -> MutChunks<'a, T>;

    /**
     * Returns a mutable reference to the first element in this slice
     * and adjusts the slice in place so that it no longer contains
     * that element. O(1).
     *
     * Equivalent to:
     *
     * ```ignore
     *     if self.len() == 0 { return None; }
     *     let head = &mut self[0];
     *     *self = self.mut_slice_from(1);
     *     Some(head)
     * ```
     *
     * Returns `None` if slice is empty
     */
    fn mut_shift_ref(&mut self) -> Option<&'a mut T>;

    /**
     * Returns a mutable reference to the last element in this slice
     * and adjusts the slice in place so that it no longer contains
     * that element. O(1).
     *
     * Equivalent to:
     *
     * ```ignore
     *     if self.len() == 0 { return None; }
     *     let tail = &mut self[self.len() - 1];
     *     *self = self.mut_slice_to(self.len() - 1);
     *     Some(tail)
     * ```
     *
     * Returns `None` if slice is empty.
     */
    fn mut_pop_ref(&mut self) -> Option<&'a mut T>;

    /// Swaps two elements in a vector.
    ///
    /// Fails if `a` or `b` are out of bounds.
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
    fn swap(self, a: uint, b: uint);


    /// Divides one `&mut` into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// Fails if `mid > len`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = [1, 2, 3, 4, 5, 6];
    ///
    /// // scoped to restrict the lifetime of the borrows
    /// {
    ///    let (left, right) = v.mut_split_at(0);
    ///    assert!(left == &mut []);
    ///    assert!(right == &mut [1, 2, 3, 4, 5, 6]);
    /// }
    ///
    /// {
    ///     let (left, right) = v.mut_split_at(2);
    ///     assert!(left == &mut [1, 2]);
    ///     assert!(right == &mut [3, 4, 5, 6]);
    /// }
    ///
    /// {
    ///     let (left, right) = v.mut_split_at(6);
    ///     assert!(left == &mut [1, 2, 3, 4, 5, 6]);
    ///     assert!(right == &mut []);
    /// }
    /// ```
    fn mut_split_at(self, mid: uint) -> (&'a mut [T], &'a mut [T]);

    /// Reverse the order of elements in a vector, in place.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = [1, 2, 3];
    /// v.reverse();
    /// assert!(v == [3, 2, 1]);
    /// ```
    fn reverse(self);

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
     * * end - The index into `str` to stop copying from
     */
    fn move_from(self, src: ~[T], start: uint, end: uint) -> uint;

    /// Returns an unsafe mutable pointer to the element in index
    unsafe fn unsafe_mut_ref(self, index: uint) -> &'a mut T;

    /// Return an unsafe mutable pointer to the vector's buffer.
    ///
    /// The caller must ensure that the vector outlives the pointer this
    /// function returns, or else it will end up pointing to garbage.
    ///
    /// Modifying the vector may cause its buffer to be reallocated, which
    /// would also make any pointers to it invalid.
    #[inline]
    fn as_mut_ptr(self) -> *mut T;

    /// Unsafely sets the element in index to the value.
    ///
    /// This performs no bounds checks, and it is undefined behaviour
    /// if `index` is larger than the length of `self`. However, it
    /// does run the destructor at `index`. It is equivalent to
    /// `self[index] = val`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = ~["foo".to_owned(), "bar".to_owned(), "baz".to_owned()];
    ///
    /// unsafe {
    ///     // `"baz".to_owned()` is deallocated.
    ///     v.unsafe_set(2, "qux".to_owned());
    ///
    ///     // Out of bounds: could cause a crash, or overwriting
    ///     // other data, or something else.
    ///     // v.unsafe_set(10, "oops".to_owned());
    /// }
    /// ```
    unsafe fn unsafe_set(self, index: uint, val: T);

    /// Unchecked vector index assignment.  Does not drop the
    /// old value and hence is only suitable when the vector
    /// is newly allocated.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = ["foo".to_owned(), "bar".to_owned()];
    ///
    /// // memory leak! `"bar".to_owned()` is not deallocated.
    /// unsafe { v.init_elem(1, "baz".to_owned()); }
    /// ```
    unsafe fn init_elem(self, i: uint, val: T);

    /// Copies raw bytes from `src` to `self`.
    ///
    /// This does not run destructors on the overwritten elements, and
    /// ignores move semantics. `self` and `src` must not
    /// overlap. Fails if `self` is shorter than `src`.
    unsafe fn copy_memory(self, src: &[T]);
}

impl<'a,T> MutableVector<'a, T> for &'a mut [T] {
    #[inline]
    fn as_mut_slice(self) -> &'a mut [T] { self }

    fn mut_slice(self, start: uint, end: uint) -> &'a mut [T] {
        assert!(start <= end);
        assert!(end <= self.len());
        unsafe {
            transmute(Slice {
                    data: self.as_mut_ptr().offset(start as int) as *T,
                    len: (end - start)
                })
        }
    }

    #[inline]
    fn mut_slice_from(self, start: uint) -> &'a mut [T] {
        let len = self.len();
        self.mut_slice(start, len)
    }

    #[inline]
    fn mut_slice_to(self, end: uint) -> &'a mut [T] {
        self.mut_slice(0, end)
    }

    #[inline]
    fn mut_split_at(self, mid: uint) -> (&'a mut [T], &'a mut [T]) {
        unsafe {
            let len = self.len();
            let self2: &'a mut [T] = cast::transmute_copy(&self);
            (self.mut_slice(0, mid), self2.mut_slice(mid, len))
        }
    }

    #[inline]
    fn mut_iter(self) -> MutItems<'a, T> {
        unsafe {
            let p = self.as_mut_ptr();
            if mem::size_of::<T>() == 0 {
                MutItems{ptr: p,
                         end: (p as uint + self.len()) as *mut T,
                         marker: marker::ContravariantLifetime::<'a>,
                         marker2: marker::NoCopy}
            } else {
                MutItems{ptr: p,
                         end: p.offset(self.len() as int),
                         marker: marker::ContravariantLifetime::<'a>,
                         marker2: marker::NoCopy}
            }
        }
    }

    #[inline]
    fn mut_last(self) -> Option<&'a mut T> {
        let len = self.len();
        if len == 0 { return None; }
        Some(&mut self[len - 1])
    }

    #[inline]
    fn mut_rev_iter(self) -> RevMutItems<'a, T> {
        self.mut_iter().rev()
    }

    #[inline]
    fn mut_split(self, pred: |&T|: 'a -> bool) -> MutSplits<'a, T> {
        MutSplits { v: self, pred: pred, finished: false }
    }

    #[inline]
    fn mut_chunks(self, chunk_size: uint) -> MutChunks<'a, T> {
        assert!(chunk_size > 0);
        MutChunks { v: self, chunk_size: chunk_size }
    }

    fn mut_shift_ref(&mut self) -> Option<&'a mut T> {
        if self.len() == 0 { return None; }
        unsafe {
            let s: &mut Slice<T> = transmute(self);
            Some(cast::transmute_mut(&*raw::shift_ptr(s)))
        }
    }

    fn mut_pop_ref(&mut self) -> Option<&'a mut T> {
        if self.len() == 0 { return None; }
        unsafe {
            let s: &mut Slice<T> = transmute(self);
            Some(cast::transmute_mut(&*raw::pop_ptr(s)))
        }
    }

    fn swap(self, a: uint, b: uint) {
        unsafe {
            // Can't take two mutable loans from one vector, so instead just cast
            // them to their raw pointers to do the swap
            let pa: *mut T = &mut self[a];
            let pb: *mut T = &mut self[b];
            ptr::swap(pa, pb);
        }
    }

    fn reverse(self) {
        let mut i: uint = 0;
        let ln = self.len();
        while i < ln / 2 {
            self.swap(i, ln - i - 1);
            i += 1;
        }
    }

    #[inline]
    fn sort_by(self, compare: |&T, &T| -> Ordering) {
        merge_sort(self, compare)
    }

    #[inline]
    fn move_from(self, mut src: ~[T], start: uint, end: uint) -> uint {
        for (a, b) in self.mut_iter().zip(src.mut_slice(start, end).mut_iter()) {
            mem::swap(a, b);
        }
        cmp::min(self.len(), end-start)
    }

    #[inline]
    unsafe fn unsafe_mut_ref(self, index: uint) -> &'a mut T {
        transmute((self.repr().data as *mut T).offset(index as int))
    }

    #[inline]
    fn as_mut_ptr(self) -> *mut T {
        self.repr().data as *mut T
    }

    #[inline]
    unsafe fn unsafe_set(self, index: uint, val: T) {
        *self.unsafe_mut_ref(index) = val;
    }

    #[inline]
    unsafe fn init_elem(self, i: uint, val: T) {
        mem::move_val_init(&mut (*self.as_mut_ptr().offset(i as int)), val);
    }

    #[inline]
    unsafe fn copy_memory(self, src: &[T]) {
        let len_src = src.len();
        assert!(self.len() >= len_src);
        ptr::copy_nonoverlapping_memory(self.as_mut_ptr(), src.as_ptr(), len_src)
    }
}

/// Trait for &[T] where T is Cloneable
pub trait MutableCloneableVector<T> {
    /// Copies as many elements from `src` as it can into `self` (the
    /// shorter of `self.len()` and `src.len()`). Returns the number
    /// of elements copied.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::slice::MutableCloneableVector;
    ///
    /// let mut dst = [0, 0, 0];
    /// let src = [1, 2];
    ///
    /// assert!(dst.copy_from(src) == 2);
    /// assert!(dst == [1, 2, 0]);
    ///
    /// let src2 = [3, 4, 5, 6];
    /// assert!(dst.copy_from(src2) == 3);
    /// assert!(dst == [3, 4, 5]);
    /// ```
    fn copy_from(self, &[T]) -> uint;
}

impl<'a, T:Clone> MutableCloneableVector<T> for &'a mut [T] {
    #[inline]
    fn copy_from(self, src: &[T]) -> uint {
        for (a, b) in self.mut_iter().zip(src.iter()) {
            a.clone_from(b);
        }
        cmp::min(self.len(), src.len())
    }
}

/// Methods for mutable vectors with orderable elements, such as
/// in-place sorting.
pub trait MutableTotalOrdVector<T> {
    /// Sort the vector, in place.
    ///
    /// This is equivalent to `self.sort_by(|a, b| a.cmp(b))`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = [-5, 4, 1, -3, 2];
    ///
    /// v.sort();
    /// assert!(v == [-5, -3, 1, 2, 4]);
    /// ```
    fn sort(self);
}
impl<'a, T: TotalOrd> MutableTotalOrdVector<T> for &'a mut [T] {
    #[inline]
    fn sort(self) {
        self.sort_by(|a,b| a.cmp(b))
    }
}

/**
* Constructs a vector from an unsafe pointer to a buffer
*
* # Arguments
*
* * ptr - An unsafe pointer to a buffer of `T`
* * elts - The number of elements in the buffer
*/
// Wrapper for fn in raw: needs to be called by net_tcp::on_tcp_read_cb
pub unsafe fn from_buf<T>(ptr: *T, elts: uint) -> ~[T] {
    raw::from_buf_raw(ptr, elts)
}

/// Unsafe operations
pub mod raw {
    use cast::transmute;
    use iter::Iterator;
    use ptr::RawPtr;
    use ptr;
    use raw::Slice;
    use slice::{MutableVector, OwnedVector};
    use vec::Vec;

    /**
     * Form a slice from a pointer and length (as a number of units,
     * not bytes).
     */
    #[inline]
    pub unsafe fn buf_as_slice<T,U>(p: *T, len: uint, f: |v: &[T]| -> U)
                               -> U {
        f(transmute(Slice {
            data: p,
            len: len
        }))
    }

    /**
     * Form a slice from a pointer and length (as a number of units,
     * not bytes).
     */
    #[inline]
    pub unsafe fn mut_buf_as_slice<T,
                                   U>(
                                   p: *mut T,
                                   len: uint,
                                   f: |v: &mut [T]| -> U)
                                   -> U {
        f(transmute(Slice {
            data: p as *T,
            len: len
        }))
    }

    /**
    * Constructs a vector from an unsafe pointer to a buffer
    *
    * # Arguments
    *
    * * ptr - An unsafe pointer to a buffer of `T`
    * * elts - The number of elements in the buffer
    */
    // Was in raw, but needs to be called by net_tcp::on_tcp_read_cb
    #[inline]
    pub unsafe fn from_buf_raw<T>(ptr: *T, elts: uint) -> ~[T] {
        let mut dst = Vec::with_capacity(elts);
        dst.set_len(elts);
        ptr::copy_memory(dst.as_mut_ptr(), ptr, elts);
        dst.move_iter().collect()
    }

    /**
     * Returns a pointer to first element in slice and adjusts
     * slice so it no longer contains that element. Fails if
     * slice is empty. O(1).
     */
    pub unsafe fn shift_ptr<T>(slice: &mut Slice<T>) -> *T {
        if slice.len == 0 { fail!("shift on empty slice"); }
        let head: *T = slice.data;
        slice.data = slice.data.offset(1);
        slice.len -= 1;
        head
    }

    /**
     * Returns a pointer to last element in slice and adjusts
     * slice so it no longer contains that element. Fails if
     * slice is empty. O(1).
     */
    pub unsafe fn pop_ptr<T>(slice: &mut Slice<T>) -> *T {
        if slice.len == 0 { fail!("pop on empty slice"); }
        let tail: *T = slice.data.offset((slice.len - 1) as int);
        slice.len -= 1;
        tail
    }
}

/// Operations on `[u8]`.
pub mod bytes {
    use container::Container;
    use slice::MutableVector;
    use ptr;

    /// A trait for operations on mutable `[u8]`s.
    pub trait MutableByteVector {
        /// Sets all bytes of the receiver to the given value.
        fn set_memory(self, value: u8);
    }

    impl<'a> MutableByteVector for &'a mut [u8] {
        #[inline]
        fn set_memory(self, value: u8) {
            unsafe { ptr::set_memory(self.as_mut_ptr(), value, self.len()) };
        }
    }

    /// Copies data from `src` to `dst`
    ///
    /// `src` and `dst` must not overlap. Fails if the length of `dst`
    /// is less than the length of `src`.
    #[inline]
    pub fn copy_memory(dst: &mut [u8], src: &[u8]) {
        // Bound checks are done at .copy_memory.
        unsafe { dst.copy_memory(src) }
    }
}

impl<A: Clone> Clone for ~[A] {
    #[inline]
    fn clone(&self) -> ~[A] {
        // Use the fast to_owned on &[A] for cloning
        self.as_slice().to_owned()
    }
}

impl<'a, T: fmt::Show> fmt::Show for &'a [T] {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if f.flags & (1 << (fmt::parse::FlagAlternate as uint)) == 0 {
            try!(write!(f.buf, "["));
        }
        let mut is_first = true;
        for x in self.iter() {
            if is_first {
                is_first = false;
            } else {
                try!(write!(f.buf, ", "));
            }
            try!(write!(f.buf, "{}", *x))
        }
        if f.flags & (1 << (fmt::parse::FlagAlternate as uint)) == 0 {
            try!(write!(f.buf, "]"));
        }
        Ok(())
    }
}

impl<'a, T: fmt::Show> fmt::Show for &'a mut [T] {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl<T: fmt::Show> fmt::Show for ~[T] {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

// This works because every lifetime is a sub-lifetime of 'static
impl<'a, A> Default for &'a [A] {
    fn default() -> &'a [A] { &'a [] }
}

impl<A> Default for ~[A] {
    fn default() -> ~[A] { ~[] }
}

/// Immutable slice iterator
pub struct Items<'a, T> {
    ptr: *T,
    end: *T,
    marker: marker::ContravariantLifetime<'a>
}

/// Mutable slice iterator
pub struct MutItems<'a, T> {
    ptr: *mut T,
    end: *mut T,
    marker: marker::ContravariantLifetime<'a>,
    marker2: marker::NoCopy
}

macro_rules! iterator {
    (struct $name:ident -> $ptr:ty, $elem:ty) => {
        impl<'a, T> Iterator<$elem> for $name<'a, T> {
            #[inline]
            fn next(&mut self) -> Option<$elem> {
                // could be implemented with slices, but this avoids bounds checks
                unsafe {
                    if self.ptr == self.end {
                        None
                    } else {
                        let old = self.ptr;
                        self.ptr = if mem::size_of::<T>() == 0 {
                            // purposefully don't use 'ptr.offset' because for
                            // vectors with 0-size elements this would return the
                            // same pointer.
                            transmute(self.ptr as uint + 1)
                        } else {
                            self.ptr.offset(1)
                        };

                        Some(transmute(old))
                    }
                }
            }

            #[inline]
            fn size_hint(&self) -> (uint, Option<uint>) {
                let diff = (self.end as uint) - (self.ptr as uint);
                let exact = diff / mem::nonzero_size_of::<T>();
                (exact, Some(exact))
            }
        }

        impl<'a, T> DoubleEndedIterator<$elem> for $name<'a, T> {
            #[inline]
            fn next_back(&mut self) -> Option<$elem> {
                // could be implemented with slices, but this avoids bounds checks
                unsafe {
                    if self.end == self.ptr {
                        None
                    } else {
                        self.end = if mem::size_of::<T>() == 0 {
                            // See above for why 'ptr.offset' isn't used
                            transmute(self.end as uint - 1)
                        } else {
                            self.end.offset(-1)
                        };
                        Some(transmute(self.end))
                    }
                }
            }
        }
    }
}

impl<'a, T> RandomAccessIterator<&'a T> for Items<'a, T> {
    #[inline]
    fn indexable(&self) -> uint {
        let (exact, _) = self.size_hint();
        exact
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<&'a T> {
        unsafe {
            if index < self.indexable() {
                transmute(self.ptr.offset(index as int))
            } else {
                None
            }
        }
    }
}

iterator!{struct Items -> *T, &'a T}
pub type RevItems<'a, T> = Rev<Items<'a, T>>;

impl<'a, T> ExactSize<&'a T> for Items<'a, T> {}
impl<'a, T> ExactSize<&'a mut T> for MutItems<'a, T> {}

impl<'a, T> Clone for Items<'a, T> {
    fn clone(&self) -> Items<'a, T> { *self }
}

iterator!{struct MutItems -> *mut T, &'a mut T}
pub type RevMutItems<'a, T> = Rev<MutItems<'a, T>>;

/// An iterator over the subslices of the vector which are separated
/// by elements that match `pred`.
pub struct MutSplits<'a, T> {
    v: &'a mut [T],
    pred: |t: &T|: 'a -> bool,
    finished: bool
}

impl<'a, T> Iterator<&'a mut [T]> for MutSplits<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.finished { return None; }

        let pred = &mut self.pred;
        match self.v.iter().position(|x| (*pred)(x)) {
            None => {
                self.finished = true;
                let tmp = mem::replace(&mut self.v, &mut []);
                let len = tmp.len();
                let (head, tail) = tmp.mut_split_at(len);
                self.v = tail;
                Some(head)
            }
            Some(idx) => {
                let tmp = mem::replace(&mut self.v, &mut []);
                let (head, tail) = tmp.mut_split_at(idx);
                self.v = tail.mut_slice_from(1);
                Some(head)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        if self.finished {
            (0, Some(0))
        } else {
            // if the predicate doesn't match anything, we yield one slice
            // if it matches every element, we yield len+1 empty slices.
            (1, Some(self.v.len() + 1))
        }
    }
}

impl<'a, T> DoubleEndedIterator<&'a mut [T]> for MutSplits<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.finished { return None; }

        let pred = &mut self.pred;
        match self.v.iter().rposition(|x| (*pred)(x)) {
            None => {
                self.finished = true;
                let tmp = mem::replace(&mut self.v, &mut []);
                Some(tmp)
            }
            Some(idx) => {
                let tmp = mem::replace(&mut self.v, &mut []);
                let (head, tail) = tmp.mut_split_at(idx);
                self.v = head;
                Some(tail.mut_slice_from(1))
            }
        }
    }
}

/// An iterator over a vector in (non-overlapping) mutable chunks (`size`  elements at a time). When
/// the vector len is not evenly divided by the chunk size, the last slice of the iteration will be
/// the remainder.
pub struct MutChunks<'a, T> {
    v: &'a mut [T],
    chunk_size: uint
}

impl<'a, T> Iterator<&'a mut [T]> for MutChunks<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.v.len() == 0 {
            None
        } else {
            let sz = cmp::min(self.v.len(), self.chunk_size);
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.mut_split_at(sz);
            self.v = tail;
            Some(head)
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        if self.v.len() == 0 {
            (0, Some(0))
        } else {
            let (n, rem) = div_rem(self.v.len(), self.chunk_size);
            let n = if rem > 0 { n + 1 } else { n };
            (n, Some(n))
        }
    }
}

impl<'a, T> DoubleEndedIterator<&'a mut [T]> for MutChunks<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.v.len() == 0 {
            None
        } else {
            let remainder = self.v.len() % self.chunk_size;
            let sz = if remainder != 0 { remainder } else { self.chunk_size };
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (head, tail) = tmp.mut_split_at(tmp_len - sz);
            self.v = head;
            Some(tail)
        }
    }
}

/// An iterator that moves out of a vector.
pub struct MoveItems<T> {
    allocation: *mut u8, // the block of memory allocated for the vector
    iter: Items<'static, T>
}

impl<T> Iterator<T> for MoveItems<T> {
    #[inline]
    fn next(&mut self) -> Option<T> {
        unsafe {
            self.iter.next().map(|x| ptr::read(x))
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

impl<T> DoubleEndedIterator<T> for MoveItems<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        unsafe {
            self.iter.next_back().map(|x| ptr::read(x))
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for MoveItems<T> {
    fn drop(&mut self) {
        // destroy the remaining elements
        for _x in *self {}
        unsafe {
            exchange_free(self.allocation as *u8)
        }
    }
}

/// An iterator that moves out of a vector in reverse order.
pub type RevMoveItems<T> = Rev<MoveItems<T>>;

impl<A> FromIterator<A> for ~[A] {
    fn from_iter<T: Iterator<A>>(mut iterator: T) -> ~[A] {
        let mut xs: Vec<A> = iterator.collect();

        // Must shrink so the capacity is the same as the length. The length of
        // the ~[T] vector must exactly match the length of the allocation.
        xs.shrink_to_fit();

        let len = xs.len();
        assert!(len == xs.capacity());
        let data = xs.as_mut_ptr();

        let data_size = len.checked_mul(&mem::size_of::<A>());
        let data_size = data_size.expect("overflow in from_iter()");
        let size = mem::size_of::<RawVec<()>>().checked_add(&data_size);
        let size = size.expect("overflow in from_iter()");


        // This is some terribly awful code. Note that all of this will go away
        // with DST because creating ~[T] from Vec<T> will just be some pointer
        // swizzling.
        unsafe {
            let ret = malloc_raw(size) as *mut RawVec<()>;

            (*ret).fill = len * mem::nonzero_size_of::<A>();
            (*ret).alloc = len * mem::nonzero_size_of::<A>();
            ptr::copy_nonoverlapping_memory(&mut (*ret).data as *mut _ as *mut u8,
                                            data as *u8,
                                            data_size);
            xs.set_len(0); // ownership has been transferred
            cast::transmute(ret)
        }
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use mem;
    use slice::*;
    use cmp::*;
    use rand::{Rng, task_rng};

    fn square(n: uint) -> uint { n * n }

    fn is_odd(n: &uint) -> bool { *n % 2u == 1u }

    #[test]
    fn test_unsafe_ptrs() {
        unsafe {
            // Test on-stack copy-from-buf.
            let a = ~[1, 2, 3];
            let mut ptr = a.as_ptr();
            let b = from_buf(ptr, 3u);
            assert_eq!(b.len(), 3u);
            assert_eq!(b[0], 1);
            assert_eq!(b[1], 2);
            assert_eq!(b[2], 3);

            // Test on-heap copy-from-buf.
            let c = ~[1, 2, 3, 4, 5];
            ptr = c.as_ptr();
            let d = from_buf(ptr, 5u);
            assert_eq!(d.len(), 5u);
            assert_eq!(d[0], 1);
            assert_eq!(d[1], 2);
            assert_eq!(d[2], 3);
            assert_eq!(d[3], 4);
            assert_eq!(d[4], 5);
        }
    }

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
        assert!(![0].is_empty());
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
        let mut a = ~[11];
        assert_eq!(a.get(1), None);
        a = ~[11, 12];
        assert_eq!(a.get(1).unwrap(), &12);
        a = ~[11, 12, 13];
        assert_eq!(a.get(1).unwrap(), &12);
    }

    #[test]
    fn test_head() {
        let mut a = ~[];
        assert_eq!(a.head(), None);
        a = ~[11];
        assert_eq!(a.head().unwrap(), &11);
        a = ~[11, 12];
        assert_eq!(a.head().unwrap(), &11);
    }

    #[test]
    fn test_tail() {
        let mut a = ~[11];
        assert_eq!(a.tail(), &[]);
        a = ~[11, 12];
        assert_eq!(a.tail(), &[12]);
    }

    #[test]
    #[should_fail]
    fn test_tail_empty() {
        let a: ~[int] = ~[];
        a.tail();
    }

    #[test]
    fn test_tailn() {
        let mut a = ~[11, 12, 13];
        assert_eq!(a.tailn(0), &[11, 12, 13]);
        a = ~[11, 12, 13];
        assert_eq!(a.tailn(2), &[13]);
    }

    #[test]
    #[should_fail]
    fn test_tailn_empty() {
        let a: ~[int] = ~[];
        a.tailn(2);
    }

    #[test]
    fn test_init() {
        let mut a = ~[11];
        assert_eq!(a.init(), &[]);
        a = ~[11, 12];
        assert_eq!(a.init(), &[11]);
    }

    #[test]
    #[should_fail]
    fn test_init_empty() {
        let a: ~[int] = ~[];
        a.init();
    }

    #[test]
    fn test_initn() {
        let mut a = ~[11, 12, 13];
        assert_eq!(a.initn(0), &[11, 12, 13]);
        a = ~[11, 12, 13];
        assert_eq!(a.initn(2), &[11]);
    }

    #[test]
    #[should_fail]
    fn test_initn_empty() {
        let a: ~[int] = ~[];
        a.initn(2);
    }

    #[test]
    fn test_last() {
        let mut a = ~[];
        assert_eq!(a.last(), None);
        a = ~[11];
        assert_eq!(a.last().unwrap(), &11);
        a = ~[11, 12];
        assert_eq!(a.last().unwrap(), &12);
    }

    #[test]
    fn test_slice() {
        // Test fixed length vector.
        let vec_fixed = [1, 2, 3, 4];
        let v_a = vec_fixed.slice(1u, vec_fixed.len()).to_owned();
        assert_eq!(v_a.len(), 3u);
        assert_eq!(v_a[0], 2);
        assert_eq!(v_a[1], 3);
        assert_eq!(v_a[2], 4);

        // Test on stack.
        let vec_stack = &[1, 2, 3];
        let v_b = vec_stack.slice(1u, 3u).to_owned();
        assert_eq!(v_b.len(), 2u);
        assert_eq!(v_b[0], 2);
        assert_eq!(v_b[1], 3);

        // Test on exchange heap.
        let vec_unique = ~[1, 2, 3, 4, 5, 6];
        let v_d = vec_unique.slice(1u, 6u).to_owned();
        assert_eq!(v_d.len(), 5u);
        assert_eq!(v_d[0], 2);
        assert_eq!(v_d[1], 3);
        assert_eq!(v_d[2], 4);
        assert_eq!(v_d[3], 5);
        assert_eq!(v_d[4], 6);
    }

    #[test]
    fn test_slice_from() {
        let vec = &[1, 2, 3, 4];
        assert_eq!(vec.slice_from(0), vec);
        assert_eq!(vec.slice_from(2), &[3, 4]);
        assert_eq!(vec.slice_from(4), &[]);
    }

    #[test]
    fn test_slice_to() {
        let vec = &[1, 2, 3, 4];
        assert_eq!(vec.slice_to(4), vec);
        assert_eq!(vec.slice_to(2), &[1, 2]);
        assert_eq!(vec.slice_to(0), &[]);
    }


    #[test]
    fn test_pop() {
        let mut v = vec![5];
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
        let mut v = vec![1, 2, 3, 4, 5];
        let mut e = v.swap_remove(0);
        assert_eq!(e, Some(1));
        assert_eq!(v, vec![5, 2, 3, 4]);
        e = v.swap_remove(3);
        assert_eq!(e, Some(4));
        assert_eq!(v, vec![5, 2, 3]);

        e = v.swap_remove(3);
        assert_eq!(e, None);
        assert_eq!(v, vec![5, 2, 3]);
    }

    #[test]
    fn test_swap_remove_noncopyable() {
        // Tests that we don't accidentally run destructors twice.
        let mut v = vec![::unstable::sync::Exclusive::new(()),
                         ::unstable::sync::Exclusive::new(()),
                         ::unstable::sync::Exclusive::new(())];
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
        v.push(1);
        assert_eq!(v.len(), 1u);
        assert_eq!(v.as_slice()[0], 1);

        // Test on-heap push().
        v.push(2);
        assert_eq!(v.len(), 2u);
        assert_eq!(v.as_slice()[0], 1);
        assert_eq!(v.as_slice()[1], 2);
    }

    #[test]
    fn test_grow() {
        // Test on-stack grow().
        let mut v = vec![];
        v.grow(2u, &1);
        {
            let v = v.as_slice();
            assert_eq!(v.len(), 2u);
            assert_eq!(v[0], 1);
            assert_eq!(v[1], 1);
        }

        // Test on-heap grow().
        v.grow(3u, &2);
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
        let mut v = vec![1, 2, 3];
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
        let mut v = vec![~6,~5,~4];
        v.truncate(1);
        let v = v.as_slice();
        assert_eq!(v.len(), 1);
        assert_eq!(*(v[0]), 6);
        // If the unsafe block didn't drop things properly, we blow up here.
    }

    #[test]
    fn test_clear() {
        let mut v = vec![~6,~5,~4];
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
        case(vec![1], vec![1]);
        case(vec![1,1], vec![1]);
        case(vec![1,2,3], vec![1,2,3]);
        case(vec![1,1,2,3], vec![1,2,3]);
        case(vec![1,2,2,3], vec![1,2,3]);
        case(vec![1,2,3,3], vec![1,2,3]);
        case(vec![1,1,2,2,2,3,3], vec![1,2,3]);
    }

    #[test]
    fn test_dedup_unique() {
        let mut v0 = vec![~1, ~1, ~2, ~3];
        v0.dedup();
        let mut v1 = vec![~1, ~2, ~2, ~3];
        v1.dedup();
        let mut v2 = vec![~1, ~2, ~3, ~3];
        v2.dedup();
        /*
         * If the ~pointers were leaked or otherwise misused, valgrind and/or
         * rustrt should raise errors.
         */
    }

    #[test]
    fn test_dedup_shared() {
        let mut v0 = vec![~1, ~1, ~2, ~3];
        v0.dedup();
        let mut v1 = vec![~1, ~2, ~2, ~3];
        v1.dedup();
        let mut v2 = vec![~1, ~2, ~3, ~3];
        v2.dedup();
        /*
         * If the pointers were leaked or otherwise misused, valgrind and/or
         * rustrt should raise errors.
         */
    }

    #[test]
    fn test_retain() {
        let mut v = vec![1, 2, 3, 4, 5];
        v.retain(is_odd);
        assert_eq!(v, vec![1, 3, 5]);
    }

    #[test]
    fn test_zip_unzip() {
        let z1 = vec![(1, 4), (2, 5), (3, 6)];

        let (left, right) = unzip(z1.iter().map(|&x| x));

        assert_eq!((1, 4), (left[0], right[0]));
        assert_eq!((2, 5), (left[1], right[1]));
        assert_eq!((3, 6), (left[2], right[2]));
    }

    #[test]
    fn test_element_swaps() {
        let mut v = [1, 2, 3];
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
            assert_eq!(it.next(), Some(v.as_slice().to_owned()));
            assert_eq!(it.next(), None);
        }
        {
            let v = ["Hello".to_owned()];
            let mut it = v.permutations();
            let (min_size, max_opt) = it.size_hint();
            assert_eq!(min_size, 1);
            assert_eq!(max_opt.unwrap(), 1);
            assert_eq!(it.next(), Some(v.as_slice().to_owned()));
            assert_eq!(it.next(), None);
        }
        {
            let v = [1, 2, 3];
            let mut it = v.permutations();
            let (min_size, max_opt) = it.size_hint();
            assert_eq!(min_size, 3*2);
            assert_eq!(max_opt.unwrap(), 3*2);
            assert_eq!(it.next(), Some(~[1,2,3]));
            assert_eq!(it.next(), Some(~[1,3,2]));
            assert_eq!(it.next(), Some(~[3,1,2]));
            let (min_size, max_opt) = it.size_hint();
            assert_eq!(min_size, 3);
            assert_eq!(max_opt.unwrap(), 3);
            assert_eq!(it.next(), Some(~[3,2,1]));
            assert_eq!(it.next(), Some(~[2,3,1]));
            assert_eq!(it.next(), Some(~[2,1,3]));
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
    fn test_position_elem() {
        assert!([].position_elem(&1).is_none());

        let v1 = ~[1, 2, 3, 3, 2, 5];
        assert_eq!(v1.position_elem(&1), Some(0u));
        assert_eq!(v1.position_elem(&2), Some(1u));
        assert_eq!(v1.position_elem(&5), Some(5u));
        assert!(v1.position_elem(&4).is_none());
    }

    #[test]
    fn test_bsearch_elem() {
        assert_eq!([1,2,3,4,5].bsearch_elem(&5), Some(4));
        assert_eq!([1,2,3,4,5].bsearch_elem(&4), Some(3));
        assert_eq!([1,2,3,4,5].bsearch_elem(&3), Some(2));
        assert_eq!([1,2,3,4,5].bsearch_elem(&2), Some(1));
        assert_eq!([1,2,3,4,5].bsearch_elem(&1), Some(0));

        assert_eq!([2,4,6,8,10].bsearch_elem(&1), None);
        assert_eq!([2,4,6,8,10].bsearch_elem(&5), None);
        assert_eq!([2,4,6,8,10].bsearch_elem(&4), Some(1));
        assert_eq!([2,4,6,8,10].bsearch_elem(&10), Some(4));

        assert_eq!([2,4,6,8].bsearch_elem(&1), None);
        assert_eq!([2,4,6,8].bsearch_elem(&5), None);
        assert_eq!([2,4,6,8].bsearch_elem(&4), Some(1));
        assert_eq!([2,4,6,8].bsearch_elem(&8), Some(3));

        assert_eq!([2,4,6].bsearch_elem(&1), None);
        assert_eq!([2,4,6].bsearch_elem(&5), None);
        assert_eq!([2,4,6].bsearch_elem(&4), Some(1));
        assert_eq!([2,4,6].bsearch_elem(&6), Some(2));

        assert_eq!([2,4].bsearch_elem(&1), None);
        assert_eq!([2,4].bsearch_elem(&5), None);
        assert_eq!([2,4].bsearch_elem(&2), Some(0));
        assert_eq!([2,4].bsearch_elem(&4), Some(1));

        assert_eq!([2].bsearch_elem(&1), None);
        assert_eq!([2].bsearch_elem(&5), None);
        assert_eq!([2].bsearch_elem(&2), Some(0));

        assert_eq!([].bsearch_elem(&1), None);
        assert_eq!([].bsearch_elem(&5), None);

        assert!([1,1,1,1,1].bsearch_elem(&1) != None);
        assert!([1,1,1,1,2].bsearch_elem(&1) != None);
        assert!([1,1,1,2,2].bsearch_elem(&1) != None);
        assert!([1,1,2,2,2].bsearch_elem(&1) != None);
        assert_eq!([1,2,2,2,2].bsearch_elem(&1), Some(0));

        assert_eq!([1,2,3,4,5].bsearch_elem(&6), None);
        assert_eq!([1,2,3,4,5].bsearch_elem(&0), None);
    }

    #[test]
    fn test_reverse() {
        let mut v: ~[int] = ~[10, 20];
        assert_eq!(v[0], 10);
        assert_eq!(v[1], 20);
        v.reverse();
        assert_eq!(v[0], 20);
        assert_eq!(v[1], 10);

        let mut v3: ~[int] = ~[];
        v3.reverse();
        assert!(v3.is_empty());
    }

    #[test]
    fn test_sort() {
        use realstd::slice::Vector;
        use realstd::clone::Clone;
        for len in range(4u, 25) {
            for _ in range(0, 100) {
                let mut v = task_rng().gen_vec::<uint>(len);
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
        for len in range(4, 25) {
            for _ in range(0 , 10) {
                let mut counts = [0, .. 10];

                // create a vector like [(6, 1), (5, 1), (6, 2), ...],
                // where the first item of each tuple is random, but
                // the second item represents which occurrence of that
                // number this element is, i.e. the second elements
                // will occur in sorted order.
                let mut v = range(0, len).map(|_| {
                        let n = task_rng().gen::<uint>() % 10;
                        counts[n] += 1;
                        (n, counts[n])
                    }).collect::<~[(uint, int)]>();

                // only sort on the first element, so an unstable sort
                // may mix up the counts.
                v.sort_by(|&(a,_), &(b,_)| a.cmp(&b));

                // this comparison includes the count (the second item
                // of the tuple), so elements with equal first items
                // will need to be ordered with increasing
                // counts... i.e. exactly asserting that this sort is
                // stable.
                assert!(v.windows(2).all(|w| w[0] <= w[1]));
            }
        }
    }

    #[test]
    fn test_partition() {
        assert_eq!((~[]).partition(|x: &int| *x < 3), (~[], ~[]));
        assert_eq!((~[1, 2, 3]).partition(|x: &int| *x < 4), (~[1, 2, 3], ~[]));
        assert_eq!((~[1, 2, 3]).partition(|x: &int| *x < 2), (~[1], ~[2, 3]));
        assert_eq!((~[1, 2, 3]).partition(|x: &int| *x < 0), (~[], ~[1, 2, 3]));
    }

    #[test]
    fn test_partitioned() {
        assert_eq!(([]).partitioned(|x: &int| *x < 3), (~[], ~[]))
        assert_eq!(([1, 2, 3]).partitioned(|x: &int| *x < 4), (~[1, 2, 3], ~[]));
        assert_eq!(([1, 2, 3]).partitioned(|x: &int| *x < 2), (~[1], ~[2, 3]));
        assert_eq!(([1, 2, 3]).partitioned(|x: &int| *x < 0), (~[], ~[1, 2, 3]));
    }

    #[test]
    fn test_concat() {
        let v: [~[int], ..0] = [];
        assert_eq!(v.concat_vec(), ~[]);
        assert_eq!([~[1], ~[2,3]].concat_vec(), ~[1, 2, 3]);

        assert_eq!([&[1], &[2,3]].concat_vec(), ~[1, 2, 3]);
    }

    #[test]
    fn test_connect() {
        let v: [~[int], ..0] = [];
        assert_eq!(v.connect_vec(&0), ~[]);
        assert_eq!([~[1], ~[2, 3]].connect_vec(&0), ~[1, 0, 2, 3]);
        assert_eq!([~[1], ~[2], ~[3]].connect_vec(&0), ~[1, 0, 2, 0, 3]);

        assert_eq!(v.connect_vec(&0), ~[]);
        assert_eq!([&[1], &[2, 3]].connect_vec(&0), ~[1, 0, 2, 3]);
        assert_eq!([&[1], &[2], &[3]].connect_vec(&0), ~[1, 0, 2, 0, 3]);
    }

    #[test]
    fn test_shift() {
        let mut x = vec![1, 2, 3];
        assert_eq!(x.shift(), Some(1));
        assert_eq!(&x, &vec![2, 3]);
        assert_eq!(x.shift(), Some(2));
        assert_eq!(x.shift(), Some(3));
        assert_eq!(x.shift(), None);
        assert_eq!(x.len(), 0);
    }

    #[test]
    fn test_unshift() {
        let mut x = vec![1, 2, 3];
        x.unshift(0);
        assert_eq!(x, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_insert() {
        let mut a = vec![1, 2, 4];
        a.insert(2, 3);
        assert_eq!(a, vec![1, 2, 3, 4]);

        let mut a = vec![1, 2, 3];
        a.insert(0, 0);
        assert_eq!(a, vec![0, 1, 2, 3]);

        let mut a = vec![1, 2, 3];
        a.insert(3, 4);
        assert_eq!(a, vec![1, 2, 3, 4]);

        let mut a = vec![];
        a.insert(0, 1);
        assert_eq!(a, vec![1]);
    }

    #[test]
    #[should_fail]
    fn test_insert_oob() {
        let mut a = vec![1, 2, 3];
        a.insert(4, 5);
    }

    #[test]
    fn test_remove() {
        let mut a = vec![1,2,3,4];

        assert_eq!(a.remove(2), Some(3));
        assert_eq!(a, vec![1,2,4]);

        assert_eq!(a.remove(2), Some(4));
        assert_eq!(a, vec![1,2]);

        assert_eq!(a.remove(2), None);
        assert_eq!(a, vec![1,2]);

        assert_eq!(a.remove(0), Some(1));
        assert_eq!(a, vec![2]);

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
        let v = vec![1, 2, 3, 4, 5];
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
            ~0
        });
    }

    #[test]
    #[should_fail]
    fn test_from_elem_fail() {
        use cast;
        use rc::Rc;

        struct S {
            f: int,
            boxes: (~int, Rc<int>)
        }

        impl Clone for S {
            fn clone(&self) -> S {
                let s = unsafe { cast::transmute_mut(self) };
                s.f += 1;
                if s.f == 10 { fail!() }
                S { f: s.f, boxes: s.boxes.clone() }
            }
        }

        let s = S { f: 0, boxes: (~0, Rc::new(0)) };
        let _ = Vec::from_elem(100, s);
    }

    #[test]
    #[should_fail]
    fn test_grow_fn_fail() {
        use rc::Rc;
        let mut v = vec![];
        v.grow_fn(100, |i| {
            if i == 50 {
                fail!()
            }
            (~0, Rc::new(0))
        })
    }

    #[test]
    #[should_fail]
    fn test_permute_fail() {
        use rc::Rc;
        let v = [(~0, Rc::new(0)), (~0, Rc::new(0)), (~0, Rc::new(0)), (~0, Rc::new(0))];
        let mut i = 0;
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
            let mut a = [1, 2, 3, 4];
            let b = [1, 2, 3, 4, 5];
            a.copy_memory(b);
        }
    }

    #[test]
    fn test_total_ord() {
        [1, 2, 3, 4].cmp(& &[1, 2, 3]) == Greater;
        [1, 2, 3].cmp(& &[1, 2, 3, 4]) == Less;
        [1, 2, 3, 4].cmp(& &[1, 2, 3, 4]) == Equal;
        [1, 2, 3, 4, 5, 5, 5, 5].cmp(& &[1, 2, 3, 4, 5, 6]) == Less;
        [2, 2].cmp(& &[1, 2, 3, 4]) == Greater;
    }

    #[test]
    fn test_iterator() {
        use iter::*;
        let xs = [1, 2, 5, 10, 11];
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
        use iter::*;
        let xs = [1, 2, 5, 10, 11];
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
        use iter::*;
        let mut xs = [1, 2, 5, 10, 11];
        assert_eq!(xs.iter().size_hint(), (5, Some(5)));
        assert_eq!(xs.rev_iter().size_hint(), (5, Some(5)));
        assert_eq!(xs.mut_iter().size_hint(), (5, Some(5)));
        assert_eq!(xs.mut_rev_iter().size_hint(), (5, Some(5)));
    }

    #[test]
    fn test_iter_clone() {
        let xs = [1, 2, 5];
        let mut it = xs.iter();
        it.next();
        let mut jt = it.clone();
        assert_eq!(it.next(), jt.next());
        assert_eq!(it.next(), jt.next());
        assert_eq!(it.next(), jt.next());
    }

    #[test]
    fn test_mut_iterator() {
        use iter::*;
        let mut xs = [1, 2, 3, 4, 5];
        for x in xs.mut_iter() {
            *x += 1;
        }
        assert!(xs == [2, 3, 4, 5, 6])
    }

    #[test]
    fn test_rev_iterator() {
        use iter::*;

        let xs = [1, 2, 5, 10, 11];
        let ys = [11, 10, 5, 2, 1];
        let mut i = 0;
        for &x in xs.rev_iter() {
            assert_eq!(x, ys[i]);
            i += 1;
        }
        assert_eq!(i, 5);
    }

    #[test]
    fn test_mut_rev_iterator() {
        use iter::*;
        let mut xs = [1u, 2, 3, 4, 5];
        for (i,x) in xs.mut_rev_iter().enumerate() {
            *x += i;
        }
        assert!(xs == [5, 5, 5, 5, 5])
    }

    #[test]
    fn test_move_iterator() {
        use iter::*;
        let xs = ~[1u,2,3,4,5];
        assert_eq!(xs.move_iter().fold(0, |a: uint, b: uint| 10*a + b), 12345);
    }

    #[test]
    fn test_move_rev_iterator() {
        use iter::*;
        let xs = ~[1u,2,3,4,5];
        assert_eq!(xs.move_rev_iter().fold(0, |a: uint, b: uint| 10*a + b), 54321);
    }

    #[test]
    fn test_splitator() {
        let xs = &[1i,2,3,4,5];

        assert_eq!(xs.split(|x| *x % 2 == 0).collect::<~[&[int]]>(),
                   ~[&[1], &[3], &[5]]);
        assert_eq!(xs.split(|x| *x == 1).collect::<~[&[int]]>(),
                   ~[&[], &[2,3,4,5]]);
        assert_eq!(xs.split(|x| *x == 5).collect::<~[&[int]]>(),
                   ~[&[1,2,3,4], &[]]);
        assert_eq!(xs.split(|x| *x == 10).collect::<~[&[int]]>(),
                   ~[&[1,2,3,4,5]]);
        assert_eq!(xs.split(|_| true).collect::<~[&[int]]>(),
                   ~[&[], &[], &[], &[], &[], &[]]);

        let xs: &[int] = &[];
        assert_eq!(xs.split(|x| *x == 5).collect::<~[&[int]]>(), ~[&[]]);
    }

    #[test]
    fn test_splitnator() {
        let xs = &[1i,2,3,4,5];

        assert_eq!(xs.splitn(0, |x| *x % 2 == 0).collect::<~[&[int]]>(),
                   ~[&[1,2,3,4,5]]);
        assert_eq!(xs.splitn(1, |x| *x % 2 == 0).collect::<~[&[int]]>(),
                   ~[&[1], &[3,4,5]]);
        assert_eq!(xs.splitn(3, |_| true).collect::<~[&[int]]>(),
                   ~[&[], &[], &[], &[4,5]]);

        let xs: &[int] = &[];
        assert_eq!(xs.splitn(1, |x| *x == 5).collect::<~[&[int]]>(), ~[&[]]);
    }

    #[test]
    fn test_rsplitator() {
        let xs = &[1i,2,3,4,5];

        assert_eq!(xs.rsplit(|x| *x % 2 == 0).collect::<~[&[int]]>(),
                   ~[&[5], &[3], &[1]]);
        assert_eq!(xs.rsplit(|x| *x == 1).collect::<~[&[int]]>(),
                   ~[&[2,3,4,5], &[]]);
        assert_eq!(xs.rsplit(|x| *x == 5).collect::<~[&[int]]>(),
                   ~[&[], &[1,2,3,4]]);
        assert_eq!(xs.rsplit(|x| *x == 10).collect::<~[&[int]]>(),
                   ~[&[1,2,3,4,5]]);

        let xs: &[int] = &[];
        assert_eq!(xs.rsplit(|x| *x == 5).collect::<~[&[int]]>(), ~[&[]]);
    }

    #[test]
    fn test_rsplitnator() {
        let xs = &[1,2,3,4,5];

        assert_eq!(xs.rsplitn(0, |x| *x % 2 == 0).collect::<~[&[int]]>(),
                   ~[&[1,2,3,4,5]]);
        assert_eq!(xs.rsplitn(1, |x| *x % 2 == 0).collect::<~[&[int]]>(),
                   ~[&[5], &[1,2,3]]);
        assert_eq!(xs.rsplitn(3, |_| true).collect::<~[&[int]]>(),
                   ~[&[], &[], &[], &[1,2]]);

        let xs: &[int] = &[];
        assert_eq!(xs.rsplitn(1, |x| *x == 5).collect::<~[&[int]]>(), ~[&[]]);
    }

    #[test]
    fn test_windowsator() {
        let v = &[1i,2,3,4];

        assert_eq!(v.windows(2).collect::<~[&[int]]>(), ~[&[1,2], &[2,3], &[3,4]]);
        assert_eq!(v.windows(3).collect::<~[&[int]]>(), ~[&[1i,2,3], &[2,3,4]]);
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

        assert_eq!(v.chunks(2).collect::<~[&[int]]>(), ~[&[1i,2], &[3,4], &[5]]);
        assert_eq!(v.chunks(3).collect::<~[&[int]]>(), ~[&[1i,2,3], &[4,5]]);
        assert_eq!(v.chunks(6).collect::<~[&[int]]>(), ~[&[1i,2,3,4,5]]);

        assert_eq!(v.chunks(2).rev().collect::<~[&[int]]>(), ~[&[5i], &[3,4], &[1,2]]);
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
        let mut a = [1,2,3,4,5];
        let b = ~[6,7,8];
        assert_eq!(a.move_from(b, 0, 3), 3);
        assert!(a == [6,7,8,4,5]);
        let mut a = [7,2,8,1];
        let b = ~[3,1,4,1,5,9];
        assert_eq!(a.move_from(b, 0, 6), 4);
        assert!(a == [3,1,4,1]);
        let mut a = [1,2,3,4];
        let b = ~[5,6,7,8,9,0];
        assert_eq!(a.move_from(b, 2, 3), 1);
        assert!(a == [7,2,3,4]);
        let mut a = [1,2,3,4,5];
        let b = ~[5,6,7,8,9,0];
        assert_eq!(a.mut_slice(2,4).move_from(b,1,6), 2);
        assert!(a == [1,2,6,7,5]);
    }

    #[test]
    fn test_copy_from() {
        let mut a = [1,2,3,4,5];
        let b = [6,7,8];
        assert_eq!(a.copy_from(b), 3);
        assert!(a == [6,7,8,4,5]);
        let mut c = [7,2,8,1];
        let d = [3,1,4,1,5,9];
        assert_eq!(c.copy_from(d), 4);
        assert!(c == [3,1,4,1]);
    }

    #[test]
    fn test_reverse_part() {
        let mut values = [1,2,3,4,5];
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
        let empty: ~[int] = ~[];
        test_show_vec!(empty, "[]".to_owned());
        test_show_vec!(~[1], "[1]".to_owned());
        test_show_vec!(~[1, 2, 3], "[1, 2, 3]".to_owned());
        test_show_vec!(~[~[], ~[1u], ~[1u, 1u]], "[[], [1], [1, 1]]".to_owned());

        let empty_mut: &mut [int] = &mut[];
        test_show_vec!(empty_mut, "[]".to_owned());
        test_show_vec!(&mut[1], "[1]".to_owned());
        test_show_vec!(&mut[1, 2, 3], "[1, 2, 3]".to_owned());
        test_show_vec!(&mut[&mut[], &mut[1u], &mut[1u, 1u]], "[[], [1], [1, 1]]".to_owned());
    }

    #[test]
    fn test_vec_default() {
        use default::Default;
        macro_rules! t (
            ($ty:ty) => {{
                let v: $ty = Default::default();
                assert!(v.is_empty());
            }}
        );

        t!(&[int]);
        t!(~[int]);
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
        v.push(1);
        v.push(2);
    }

    #[test]
    #[should_fail]
    fn test_overflow_does_not_cause_segfault_managed() {
        use rc::Rc;
        let mut v = vec![Rc::new(1)];
        v.reserve_exact(-1);
        v.push(Rc::new(2));
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

    #[deriving(Clone, Eq)]
    struct Foo;

    #[test]
    fn test_iter_zero_sized() {
        let mut v = vec![Foo, Foo, Foo];
        assert_eq!(v.len(), 3);
        let mut cnt = 0;

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

        let xs = vec![Foo, Foo, Foo];
        assert_eq!(format!("{:?}", xs.slice(0, 2).to_owned()),
                   "~[slice::tests::Foo, slice::tests::Foo]".to_owned());

        let xs: [Foo, ..3] = [Foo, Foo, Foo];
        assert_eq!(format!("{:?}", xs.slice(0, 2).to_owned()),
                   "~[slice::tests::Foo, slice::tests::Foo]".to_owned());
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
        for i in range(4, 100) {
            xs.push(i)
        }
        assert_eq!(xs.capacity(), 128);
        xs.shrink_to_fit();
        assert_eq!(xs.capacity(), 100);
        assert_eq!(xs, range(0, 100).collect::<Vec<_>>());
    }

    #[test]
    fn test_starts_with() {
        assert!(bytes!("foobar").starts_with(bytes!("foo")));
        assert!(!bytes!("foobar").starts_with(bytes!("oob")));
        assert!(!bytes!("foobar").starts_with(bytes!("bar")));
        assert!(!bytes!("foo").starts_with(bytes!("foobar")));
        assert!(!bytes!("bar").starts_with(bytes!("foobar")));
        assert!(bytes!("foobar").starts_with(bytes!("foobar")));
        let empty: &[u8] = [];
        assert!(empty.starts_with(empty));
        assert!(!empty.starts_with(bytes!("foo")));
        assert!(bytes!("foobar").starts_with(empty));
    }

    #[test]
    fn test_ends_with() {
        assert!(bytes!("foobar").ends_with(bytes!("bar")));
        assert!(!bytes!("foobar").ends_with(bytes!("oba")));
        assert!(!bytes!("foobar").ends_with(bytes!("foo")));
        assert!(!bytes!("foo").ends_with(bytes!("foobar")));
        assert!(!bytes!("bar").ends_with(bytes!("foobar")));
        assert!(bytes!("foobar").ends_with(bytes!("foobar")));
        let empty: &[u8] = [];
        assert!(empty.ends_with(empty));
        assert!(!empty.ends_with(bytes!("foo")));
        assert!(bytes!("foobar").ends_with(empty));
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
        let mut xs = [0,1,0,2,3,0,0,4,5,0];
        assert_eq!(xs.mut_split(|x| *x == 0).len(), 6);
        for slice in xs.mut_split(|x| *x == 0) {
            slice.reverse();
        }
        assert!(xs == [0,1,0,3,2,0,0,5,4,0]);

        let mut xs = [0,1,0,2,3,0,0,4,5,0,6,7];
        for slice in xs.mut_split(|x| *x == 0).take(5) {
            slice.reverse();
        }
        assert!(xs == [0,1,0,3,2,0,0,5,4,0,6,7]);
    }

    #[test]
    fn test_mut_splitator_rev() {
        let mut xs = [1,2,0,3,4,0,0,5,6,0];
        for slice in xs.mut_split(|x| *x == 0).rev().take(4) {
            slice.reverse();
        }
        assert!(xs == [1,2,0,4,3,0,0,6,5,0]);
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
        let mut v = [1, 2, 3, 4];
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
        let mut x = [1, 2, 3, 4, 5];
        let h = x.mut_last();
        assert_eq!(*h.unwrap(), 5);

        let y: &mut [int] = [];
        assert!(y.mut_last().is_none());
    }
}

#[cfg(test)]
mod bench {
    extern crate test;
    use self::test::Bencher;
    use mem;
    use prelude::*;
    use ptr;
    use rand::{weak_rng, Rng};

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
        let mut v = Vec::from_elem(100, 0);

        b.iter(|| {
            let mut i = 0;
            for x in v.mut_iter() {
                *x = i;
                i += 1;
            }
        })
    }

    #[bench]
    fn add(b: &mut Bencher) {
        let xs: &[int] = [5, ..10];
        let ys: &[int] = [5, ..10];
        b.iter(|| {
            xs + ys;
        });
    }

    #[bench]
    fn concat(b: &mut Bencher) {
        let xss: Vec<Vec<uint>> = Vec::from_fn(100, |i| range(0, i).collect());
        b.iter(|| {
            xss.as_slice().concat_vec()
        });
    }

    #[bench]
    fn connect(b: &mut Bencher) {
        let xss: Vec<Vec<uint>> = Vec::from_fn(100, |i| range(0, i).collect());
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
    fn zero_1kb_fixed_repeat(b: &mut Bencher) {
        b.iter(|| {
            ~[0u8, ..1024]
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
                *x = 0;
            }
            v
        });
    }

    #[bench]
    fn random_inserts(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
                let mut v = Vec::from_elem(30, (0u, 0u));
                for _ in range(0, 100) {
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
                for _ in range(0, 100) {
                    let l = v.len();
                    v.remove(rng.gen::<uint>() % l);
                }
            })
    }

    #[bench]
    fn sort_random_small(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v = rng.gen_vec::<u64>(5);
            v.as_mut_slice().sort();
        });
        b.bytes = 5 * mem::size_of::<u64>() as u64;
    }

    #[bench]
    fn sort_random_medium(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v = rng.gen_vec::<u64>(100);
            v.as_mut_slice().sort();
        });
        b.bytes = 100 * mem::size_of::<u64>() as u64;
    }

    #[bench]
    fn sort_random_large(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v = rng.gen_vec::<u64>(10000);
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
            let mut v = rng.gen_vec::<BigSortable>(5);
            v.sort();
        });
        b.bytes = 5 * mem::size_of::<BigSortable>() as u64;
    }

    #[bench]
    fn sort_big_random_medium(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v = rng.gen_vec::<BigSortable>(100);
            v.sort();
        });
        b.bytes = 100 * mem::size_of::<BigSortable>() as u64;
    }

    #[bench]
    fn sort_big_random_large(b: &mut Bencher) {
        let mut rng = weak_rng();
        b.iter(|| {
            let mut v = rng.gen_vec::<BigSortable>(10000);
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
