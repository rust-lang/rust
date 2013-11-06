// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Vector manipulation

The `vec` module contains useful code to help work with vector values.
Vectors are Rust's list type. Vectors contain zero or more values of
homogeneous types:

```rust
let int_vector = [1,2,3];
let str_vector = ["one", "two", "three"];
 ```

This is a big module, but for a high-level overview:

## Structs

Several structs that are useful for vectors, such as `VecIterator`, which
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
let mut numbers = ~[0, 1, 2];
numbers.push(7);
// numbers is now ~[0, 1, 2, 7];
 ```

## Implementations of other traits

Vectors are a very useful type, and so there's several implementations of
traits from other modules. Some notable examples:

* `Clone`
* `Eq`, `Ord`, `TotalEq`, `TotalOrd` -- vectors can be compared,
  if the element type defines the corresponding trait.

## Iteration

The method `iter()` returns an iteration value for a vector or a vector slice.
The iterator yields borrowed pointers to the vector's elements, so if the element
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

#[warn(non_camel_case_types)];

use cast;
use clone::{Clone, DeepClone};
use container::{Container, Mutable};
use cmp::{Eq, TotalOrd, Ordering, Less, Equal, Greater};
use cmp;
use default::Default;
use iter::*;
use libc::c_void;
use num::{Integer, CheckedAdd, Saturating};
use option::{None, Option, Some};
use ptr::to_unsafe_ptr;
use ptr;
use ptr::RawPtr;
use rt::global_heap::malloc_raw;
use rt::global_heap::realloc_raw;
use mem;
use mem::size_of;
use uint;
use unstable::finally::Finally;
use unstable::intrinsics;
use unstable::intrinsics::{get_tydesc, owns_managed};
use unstable::raw::{Box, Repr, Slice, Vec};
use vec;
use util;

/**
 * Creates and initializes an owned vector.
 *
 * Creates an owned vector of size `n_elts` and initializes the elements
 * to the value returned by the function `op`.
 */
pub fn from_fn<T>(n_elts: uint, op: |uint| -> T) -> ~[T] {
    unsafe {
        let mut v = with_capacity(n_elts);
        let p = raw::to_mut_ptr(v);
        let mut i: uint = 0u;
        do (|| {
            while i < n_elts {
                intrinsics::move_val_init(&mut(*ptr::mut_offset(p, i as int)), op(i));
                i += 1u;
            }
        }).finally {
            raw::set_len(&mut v, i);
        }
        v
    }
}

/**
 * Creates and initializes an owned vector.
 *
 * Creates an owned vector of size `n_elts` and initializes the elements
 * to the value `t`.
 */
pub fn from_elem<T:Clone>(n_elts: uint, t: T) -> ~[T] {
    // FIXME (#7136): manually inline from_fn for 2x plus speedup (sadly very
    // important, from_elem is a bottleneck in borrowck!). Unfortunately it
    // still is substantially slower than using the unsafe
    // vec::with_capacity/ptr::set_memory for primitive types.
    unsafe {
        let mut v = with_capacity(n_elts);
        let p = raw::to_mut_ptr(v);
        let mut i = 0u;
        do (|| {
            while i < n_elts {
                intrinsics::move_val_init(&mut(*ptr::mut_offset(p, i as int)), t.clone());
                i += 1u;
            }
        }).finally {
            raw::set_len(&mut v, i);
        }
        v
    }
}

/// Creates a new vector with a capacity of `capacity`
#[inline]
pub fn with_capacity<T>(capacity: uint) -> ~[T] {
    unsafe {
        if owns_managed::<T>() {
            let mut vec = ~[];
            vec.reserve(capacity);
            vec
        } else {
            let alloc = capacity * mem::nonzero_size_of::<T>();
            let size = alloc + mem::size_of::<Vec<()>>();
            if alloc / mem::nonzero_size_of::<T>() != capacity || size < alloc {
                fail!("vector size is too large: {}", capacity);
            }
            let ptr = malloc_raw(size) as *mut Vec<()>;
            (*ptr).alloc = alloc;
            (*ptr).fill = 0;
            cast::transmute(ptr)
        }
    }
}

/**
 * Builds a vector by calling a provided function with an argument
 * function that pushes an element to the back of a vector.
 * The initial capacity for the vector may optionally be specified.
 *
 * # Arguments
 *
 * * size - An option, maybe containing initial size of the vector to reserve
 * * builder - A function that will construct the vector. It receives
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline]
pub fn build<A>(size: Option<uint>, builder: |push: |v: A||) -> ~[A] {
    let mut vec = with_capacity(size.unwrap_or(4));
    builder(|x| vec.push(x));
    vec
}

/// An iterator over the slices of a vector separated by elements that
/// match a predicate function.
pub struct SplitIterator<'self, T> {
    priv v: &'self [T],
    priv n: uint,
    priv pred: &'self fn(t: &T) -> bool,
    priv finished: bool
}

impl<'self, T> Iterator<&'self [T]> for SplitIterator<'self, T> {
    #[inline]
    fn next(&mut self) -> Option<&'self [T]> {
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
pub struct RSplitIterator<'self, T> {
    priv v: &'self [T],
    priv n: uint,
    priv pred: &'self fn(t: &T) -> bool,
    priv finished: bool
}

impl<'self, T> Iterator<&'self [T]> for RSplitIterator<'self, T> {
    #[inline]
    fn next(&mut self) -> Option<&'self [T]> {
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

// Appending

/// Iterates over the `rhs` vector, copying each element and appending it to the
/// `lhs`. Afterwards, the `lhs` is then returned for use again.
#[inline]
pub fn append<T:Clone>(lhs: ~[T], rhs: &[T]) -> ~[T] {
    let mut v = lhs;
    v.push_all(rhs);
    v
}

/// Appends one element to the vector provided. The vector itself is then
/// returned for use again.
#[inline]
pub fn append_one<T>(lhs: ~[T], x: T) -> ~[T] {
    let mut v = lhs;
    v.push(x);
    v
}

// Functional utilities

/**
 * Apply a function to each element of a vector and return a concatenation
 * of each result vector
 */
pub fn flat_map<T, U>(v: &[T], f: |t: &T| -> ~[U]) -> ~[U] {
    let mut result = ~[];
    for elem in v.iter() { result.push_all_move(f(elem)); }
    result
}

#[allow(missing_doc)]
pub trait VectorVector<T> {
    // FIXME #5898: calling these .concat and .connect conflicts with
    // StrVector::con{cat,nect}, since they have generic contents.
    /// Flattens a vector of vectors of T into a single vector of T.
    fn concat_vec(&self) -> ~[T];

    /// Concatenate a vector of vectors, placing a given separator between each.
    fn connect_vec(&self, sep: &T) -> ~[T];
}

impl<'self, T: Clone, V: Vector<T>> VectorVector<T> for &'self [V] {
    fn concat_vec(&self) -> ~[T] {
        let size = self.iter().fold(0u, |acc, v| acc + v.as_slice().len());
        let mut result = with_capacity(size);
        for v in self.iter() {
            result.push_all(v.as_slice())
        }
        result
    }

    fn connect_vec(&self, sep: &T) -> ~[T] {
        let size = self.iter().fold(0u, |acc, v| acc + v.as_slice().len());
        let mut result = with_capacity(size + self.len());
        let mut first = true;
        for v in self.iter() {
            if first { first = false } else { result.push(sep.clone()) }
            result.push_all(v.as_slice())
        }
        result
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
    let mut ts = with_capacity(lo);
    let mut us = with_capacity(lo);
    for (t, u) in iter {
        ts.push(t);
        us.push(u);
    }
    (ts, us)
}

/// An Iterator that yields the element swaps needed to produce
/// a sequence of all possible permutations for an indexed sequence of
/// elements. Each permutation is only a single swap apart.
///
/// The Steinhaus–Johnson–Trotter algorithm is used.
///
/// Generates even and odd permutations alternatingly.
///
/// The last generated swap is always (0, 1), and it returns the
/// sequence to its initial order.
pub struct ElementSwaps {
    priv sdir: ~[SizeDirection],
    /// If true, emit the last swap that returns the sequence to initial state
    priv emit_reset: bool,
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
                    .to_owned_vec()
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
                Some((i, j))
            },
            None => if self.emit_reset && self.sdir.len() > 1 {
                self.emit_reset = false;
                Some((0, 1))
            } else {
                None
            }
        }
    }
}

/// An Iterator that uses `ElementSwaps` to iterate through
/// all possible permutations of a vector.
///
/// The first iteration yields a clone of the vector as it is,
/// then each successive element is the vector with one
/// swap applied.
///
/// Generates even and odd permutations alternatingly.
pub struct Permutations<T> {
    priv swaps: ElementSwaps,
    priv v: ~[T],
}

impl<T: Clone> Iterator<~[T]> for Permutations<T> {
    #[inline]
    fn next(&mut self) -> Option<~[T]> {
        match self.swaps.next() {
            None => None,
            Some((a, b)) => {
                let elt = self.v.clone();
                self.v.swap(a, b);
                Some(elt)
            }
        }
    }
}

/// An iterator over the (overlapping) slices of length `size` within
/// a vector.
#[deriving(Clone)]
pub struct WindowIter<'self, T> {
    priv v: &'self [T],
    priv size: uint
}

impl<'self, T> Iterator<&'self [T]> for WindowIter<'self, T> {
    #[inline]
    fn next(&mut self) -> Option<&'self [T]> {
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
pub struct ChunkIter<'self, T> {
    priv v: &'self [T],
    priv size: uint
}

impl<'self, T> Iterator<&'self [T]> for ChunkIter<'self, T> {
    #[inline]
    fn next(&mut self) -> Option<&'self [T]> {
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
            let (n, rem) = self.v.len().div_rem(&self.size);
            let n = if rem > 0 { n+1 } else { n };
            (n, Some(n))
        }
    }
}

impl<'self, T> DoubleEndedIterator<&'self [T]> for ChunkIter<'self, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'self [T]> {
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

impl<'self, T> RandomAccessIterator<&'self [T]> for ChunkIter<'self, T> {
    #[inline]
    fn indexable(&self) -> uint {
        self.v.len()/self.size + if self.v.len() % self.size != 0 { 1 } else { 0 }
    }

    #[inline]
    fn idx(&self, index: uint) -> Option<&'self [T]> {
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

    use clone::Clone;
    use cmp::{Eq, Ord, TotalEq, TotalOrd, Ordering, Equiv};
    use iter::order;
    use ops::Add;

    impl<'self,T:Eq> Eq for &'self [T] {
        fn eq(&self, other: & &'self [T]) -> bool {
            self.len() == other.len() &&
                order::eq(self.iter(), other.iter())
        }
        fn ne(&self, other: & &'self [T]) -> bool {
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

    impl<T:Eq> Eq for @[T] {
        #[inline]
        fn eq(&self, other: &@[T]) -> bool { self.as_slice() == *other }
        #[inline]
        fn ne(&self, other: &@[T]) -> bool { !self.eq(other) }
    }

    impl<'self,T:TotalEq> TotalEq for &'self [T] {
        fn equals(&self, other: & &'self [T]) -> bool {
            self.len() == other.len() &&
                order::equals(self.iter(), other.iter())
        }
    }

    impl<T:TotalEq> TotalEq for ~[T] {
        #[inline]
        fn equals(&self, other: &~[T]) -> bool { self.as_slice().equals(&other.as_slice()) }
    }

    impl<T:TotalEq> TotalEq for @[T] {
        #[inline]
        fn equals(&self, other: &@[T]) -> bool { self.as_slice().equals(&other.as_slice()) }
    }

    impl<'self,T:Eq, V: Vector<T>> Equiv<V> for &'self [T] {
        #[inline]
        fn equiv(&self, other: &V) -> bool { self.as_slice() == other.as_slice() }
    }

    impl<'self,T:Eq, V: Vector<T>> Equiv<V> for ~[T] {
        #[inline]
        fn equiv(&self, other: &V) -> bool { self.as_slice() == other.as_slice() }
    }

    impl<'self,T:Eq, V: Vector<T>> Equiv<V> for @[T] {
        #[inline]
        fn equiv(&self, other: &V) -> bool { self.as_slice() == other.as_slice() }
    }

    impl<'self,T:TotalOrd> TotalOrd for &'self [T] {
        fn cmp(&self, other: & &'self [T]) -> Ordering {
            order::cmp(self.iter(), other.iter())
        }
    }

    impl<T: TotalOrd> TotalOrd for ~[T] {
        #[inline]
        fn cmp(&self, other: &~[T]) -> Ordering { self.as_slice().cmp(&other.as_slice()) }
    }

    impl<T: TotalOrd> TotalOrd for @[T] {
        #[inline]
        fn cmp(&self, other: &@[T]) -> Ordering { self.as_slice().cmp(&other.as_slice()) }
    }

    impl<'self, T: Eq + Ord> Ord for &'self [T] {
        fn lt(&self, other: & &'self [T]) -> bool {
            order::lt(self.iter(), other.iter())
        }
        #[inline]
        fn le(&self, other: & &'self [T]) -> bool {
            order::le(self.iter(), other.iter())
        }
        #[inline]
        fn ge(&self, other: & &'self [T]) -> bool {
            order::ge(self.iter(), other.iter())
        }
        #[inline]
        fn gt(&self, other: & &'self [T]) -> bool {
            order::gt(self.iter(), other.iter())
        }
    }

    impl<T: Eq + Ord> Ord for ~[T] {
        #[inline]
        fn lt(&self, other: &~[T]) -> bool { self.as_slice() < other.as_slice() }
        #[inline]
        fn le(&self, other: &~[T]) -> bool { self.as_slice() <= other.as_slice() }
        #[inline]
        fn ge(&self, other: &~[T]) -> bool { self.as_slice() >= other.as_slice() }
        #[inline]
        fn gt(&self, other: &~[T]) -> bool { self.as_slice() > other.as_slice() }
    }

    impl<T: Eq + Ord> Ord for @[T] {
        #[inline]
        fn lt(&self, other: &@[T]) -> bool { self.as_slice() < other.as_slice() }
        #[inline]
        fn le(&self, other: &@[T]) -> bool { self.as_slice() <= other.as_slice() }
        #[inline]
        fn ge(&self, other: &@[T]) -> bool { self.as_slice() >= other.as_slice() }
        #[inline]
        fn gt(&self, other: &@[T]) -> bool { self.as_slice() > other.as_slice() }
    }

    impl<'self,T:Clone, V: Vector<T>> Add<V, ~[T]> for &'self [T] {
        #[inline]
        fn add(&self, rhs: &V) -> ~[T] {
            let mut res = with_capacity(self.len() + rhs.as_slice().len());
            res.push_all(*self);
            res.push_all(rhs.as_slice());
            res
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

impl<'self,T> Vector<T> for &'self [T] {
    #[inline(always)]
    fn as_slice<'a>(&'a self) -> &'a [T] { *self }
}

impl<T> Vector<T> for ~[T] {
    #[inline(always)]
    fn as_slice<'a>(&'a self) -> &'a [T] { let v: &'a [T] = *self; v }
}

impl<T> Vector<T> for @[T] {
    #[inline(always)]
    fn as_slice<'a>(&'a self) -> &'a [T] { let v: &'a [T] = *self; v }
}

impl<'self, T> Container for &'self [T] {
    /// Returns the length of a vector
    #[inline]
    fn len(&self) -> uint {
        self.as_imm_buf(|_p, len| len)
    }
}

impl<T> Container for ~[T] {
    /// Returns the length of a vector
    #[inline]
    fn len(&self) -> uint {
        self.as_imm_buf(|_p, len| len)
    }
}

/// Extension methods for vector slices with copyable elements
pub trait CopyableVector<T> {
    /// Copy `self` into a new owned vector
    fn to_owned(&self) -> ~[T];

    /// Convert `self` into a owned vector, not making a copy if possible.
    fn into_owned(self) -> ~[T];
}

/// Extension methods for vector slices
impl<'self, T: Clone> CopyableVector<T> for &'self [T] {
    /// Returns a copy of `v`.
    #[inline]
    fn to_owned(&self) -> ~[T] {
        let mut result = with_capacity(self.len());
        for e in self.iter() {
            result.push((*e).clone());
        }
        result
    }

    #[inline(always)]
    fn into_owned(self) -> ~[T] { self.to_owned() }
}

/// Extension methods for owned vectors
impl<T: Clone> CopyableVector<T> for ~[T] {
    #[inline]
    fn to_owned(&self) -> ~[T] { self.clone() }

    #[inline(always)]
    fn into_owned(self) -> ~[T] { self }
}

/// Extension methods for managed vectors
impl<T: Clone> CopyableVector<T> for @[T] {
    #[inline]
    fn to_owned(&self) -> ~[T] { self.as_slice().to_owned() }

    #[inline(always)]
    fn into_owned(self) -> ~[T] { self.to_owned() }
}

/// Extension methods for vectors
pub trait ImmutableVector<'self, T> {
    /**
     * Returns a slice of self between `start` and `end`.
     *
     * Fails when `start` or `end` point outside the bounds of self,
     * or when `start` > `end`.
     */
    fn slice(&self, start: uint, end: uint) -> &'self [T];

    /**
     * Returns a slice of self from `start` to the end of the vec.
     *
     * Fails when `start` points outside the bounds of self.
     */
    fn slice_from(&self, start: uint) -> &'self [T];

    /**
     * Returns a slice of self from the start of the vec to `end`.
     *
     * Fails when `end` points outside the bounds of self.
     */
    fn slice_to(&self, end: uint) -> &'self [T];
    /// Returns an iterator over the vector
    fn iter(self) -> VecIterator<'self, T>;
    /// Returns a reversed iterator over a vector
    fn rev_iter(self) -> RevIterator<'self, T>;
    /// Returns an iterator over the subslices of the vector which are
    /// separated by elements that match `pred`.
    fn split_iter(self, pred: &'self fn(&T) -> bool) -> SplitIterator<'self, T>;
    /// Returns an iterator over the subslices of the vector which are
    /// separated by elements that match `pred`, limited to splitting
    /// at most `n` times.
    fn splitn_iter(self, n: uint, pred: &'self fn(&T) -> bool) -> SplitIterator<'self, T>;
    /// Returns an iterator over the subslices of the vector which are
    /// separated by elements that match `pred`. This starts at the
    /// end of the vector and works backwards.
    fn rsplit_iter(self, pred: &'self fn(&T) -> bool) -> RSplitIterator<'self, T>;
    /// Returns an iterator over the subslices of the vector which are
    /// separated by elements that match `pred` limited to splitting
    /// at most `n` times. This starts at the end of the vector and
    /// works backwards.
    fn rsplitn_iter(self,  n: uint, pred: &'self fn(&T) -> bool) -> RSplitIterator<'self, T>;

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
     * for win in v.window_iter() {
     *     println!("{:?}", win);
     * }
     * ```
     *
     */
    fn window_iter(self, size: uint) -> WindowIter<'self, T>;
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
     * for win in v.chunk_iter() {
     *     println!("{:?}", win);
     * }
     * ```
     *
     */
    fn chunk_iter(self, size: uint) -> ChunkIter<'self, T>;

    /// Returns the element of a vector at the given index, or `None` if the
    /// index is out of bounds
    fn get_opt(&self, index: uint) -> Option<&'self T>;
    /// Returns the first element of a vector, failing if the vector is empty.
    fn head(&self) -> &'self T;
    /// Returns the first element of a vector, or `None` if it is empty
    fn head_opt(&self) -> Option<&'self T>;
    /// Returns all but the first element of a vector
    fn tail(&self) -> &'self [T];
    /// Returns all but the first `n' elements of a vector
    fn tailn(&self, n: uint) -> &'self [T];
    /// Returns all but the last element of a vector
    fn init(&self) -> &'self [T];
    /// Returns all but the last `n' elemnts of a vector
    fn initn(&self, n: uint) -> &'self [T];
    /// Returns the last element of a vector, failing if the vector is empty.
    fn last(&self) -> &'self T;
    /// Returns the last element of a vector, or `None` if it is empty.
    fn last_opt(&self) -> Option<&'self T>;
    /**
     * Apply a function to each element of a vector and return a concatenation
     * of each result vector
     */
    fn flat_map<U>(&self, f: |t: &T| -> ~[U]) -> ~[U];
    /// Returns a pointer to the element at the given index, without doing
    /// bounds checking.
    unsafe fn unsafe_ref(&self, index: uint) -> *T;

    /**
     * Binary search a sorted vector with a comparator function.
     *
     * The comparator should implement an order consistent with the sort
     * order of the underlying vector, returning an order code that indicates
     * whether its argument is `Less`, `Equal` or `Greater` the desired target.
     *
     * Returns the index where the comparator returned `Equal`, or `None` if
     * not found.
     */
    fn bsearch(&self, f: |&T| -> Ordering) -> Option<uint>;

    /// Deprecated, use iterators where possible
    /// (`self.iter().map(f)`). Apply a function to each element
    /// of a vector and return the results.
    fn map<U>(&self, |t: &T| -> U) -> ~[U];

    /**
     * Work with the buffer of a vector.
     *
     * Allows for unsafe manipulation of vector contents, which is useful for
     * foreign interop.
     */
    fn as_imm_buf<U>(&self, f: |*T, uint| -> U) -> U;
}

impl<'self,T> ImmutableVector<'self, T> for &'self [T] {
    #[inline]
    fn slice(&self, start: uint, end: uint) -> &'self [T] {
        assert!(start <= end);
        assert!(end <= self.len());
        do self.as_imm_buf |p, _len| {
            unsafe {
                cast::transmute(Slice {
                    data: ptr::offset(p, start as int),
                    len: (end - start)
                })
            }
        }
    }

    #[inline]
    fn slice_from(&self, start: uint) -> &'self [T] {
        self.slice(start, self.len())
    }

    #[inline]
    fn slice_to(&self, end: uint) -> &'self [T] {
        self.slice(0, end)
    }

    #[inline]
    fn iter(self) -> VecIterator<'self, T> {
        unsafe {
            let p = vec::raw::to_ptr(self);
            if mem::size_of::<T>() == 0 {
                VecIterator{ptr: p,
                            end: (p as uint + self.len()) as *T,
                            lifetime: None}
            } else {
                VecIterator{ptr: p,
                            end: p.offset(self.len() as int),
                            lifetime: None}
            }
        }
    }

    #[inline]
    fn rev_iter(self) -> RevIterator<'self, T> {
        self.iter().invert()
    }

    #[inline]
    fn split_iter(self, pred: &'self fn(&T) -> bool) -> SplitIterator<'self, T> {
        self.splitn_iter(uint::max_value, pred)
    }
    #[inline]
    fn splitn_iter(self, n: uint, pred: &'self fn(&T) -> bool) -> SplitIterator<'self, T> {
        SplitIterator {
            v: self,
            n: n,
            pred: pred,
            finished: false
        }
    }
    #[inline]
    fn rsplit_iter(self, pred: &'self fn(&T) -> bool) -> RSplitIterator<'self, T> {
        self.rsplitn_iter(uint::max_value, pred)
    }
    #[inline]
    fn rsplitn_iter(self, n: uint, pred: &'self fn(&T) -> bool) -> RSplitIterator<'self, T> {
        RSplitIterator {
            v: self,
            n: n,
            pred: pred,
            finished: false
        }
    }

    fn window_iter(self, size: uint) -> WindowIter<'self, T> {
        assert!(size != 0);
        WindowIter { v: self, size: size }
    }

    fn chunk_iter(self, size: uint) -> ChunkIter<'self, T> {
        assert!(size != 0);
        ChunkIter { v: self, size: size }
    }

    #[inline]
    fn get_opt(&self, index: uint) -> Option<&'self T> {
        if index < self.len() { Some(&self[index]) } else { None }
    }

    #[inline]
    fn head(&self) -> &'self T {
        if self.len() == 0 { fail!("head: empty vector") }
        &self[0]
    }

    #[inline]
    fn head_opt(&self) -> Option<&'self T> {
        if self.len() == 0 { None } else { Some(&self[0]) }
    }

    #[inline]
    fn tail(&self) -> &'self [T] { self.slice(1, self.len()) }

    #[inline]
    fn tailn(&self, n: uint) -> &'self [T] { self.slice(n, self.len()) }

    #[inline]
    fn init(&self) -> &'self [T] {
        self.slice(0, self.len() - 1)
    }

    #[inline]
    fn initn(&self, n: uint) -> &'self [T] {
        self.slice(0, self.len() - n)
    }

    #[inline]
    fn last(&self) -> &'self T {
        if self.len() == 0 { fail!("last: empty vector") }
        &self[self.len() - 1]
    }

    #[inline]
    fn last_opt(&self) -> Option<&'self T> {
            if self.len() == 0 { None } else { Some(&self[self.len() - 1]) }
    }

    #[inline]
    fn flat_map<U>(&self, f: |t: &T| -> ~[U]) -> ~[U] {
        flat_map(*self, f)
    }

    #[inline]
    unsafe fn unsafe_ref(&self, index: uint) -> *T {
        self.repr().data.offset(index as int)
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

    fn map<U>(&self, f: |t: &T| -> U) -> ~[U] {
        self.iter().map(f).collect()
    }

    #[inline]
    fn as_imm_buf<U>(&self, f: |*T, uint| -> U) -> U {
        let s = self.repr();
        f(s.data, s.len)
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

impl<'self,T:Eq> ImmutableEqVector<T> for &'self [T] {
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

impl<'self, T: TotalOrd> ImmutableTotalOrdVector<T> for &'self [T] {
    fn bsearch_elem(&self, x: &T) -> Option<uint> {
        self.bsearch(|p| p.cmp(x))
    }
}

/// Extension methods for vectors containing `Clone` elements.
pub trait ImmutableCopyableVector<T> {
    /**
     * Partitions the vector into those that satisfies the predicate, and
     * those that do not.
     */
    fn partitioned(&self, f: |&T| -> bool) -> (~[T], ~[T]);
    /// Returns the element at the given index, without doing bounds checking.
    unsafe fn unsafe_get(&self, elem: uint) -> T;

    /// Create an iterator that yields every possible permutation of the
    /// vector in succession.
    fn permutations_iter(self) -> Permutations<T>;
}

impl<'self,T:Clone> ImmutableCopyableVector<T> for &'self [T] {
    #[inline]
    fn partitioned(&self, f: |&T| -> bool) -> (~[T], ~[T]) {
        let mut lefts  = ~[];
        let mut rights = ~[];

        for elt in self.iter() {
            if f(elt) {
                lefts.push((*elt).clone());
            } else {
                rights.push((*elt).clone());
            }
        }

        (lefts, rights)
    }

    #[inline]
    unsafe fn unsafe_get(&self, index: uint) -> T {
        (*self.unsafe_ref(index)).clone()
    }

    fn permutations_iter(self) -> Permutations<T> {
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
    /// Note that this performs O(n) swaps, and so `move_rev_iter`
    /// (which just calls `pop` repeatedly) is more efficient.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let v = ~[~"a", ~"b"];
    /// for s in v.move_iter() {
    ///   // s has type ~str, not &~str
    ///   println(s);
    /// }
    /// ```
    fn move_iter(self) -> MoveIterator<T>;
    /// Creates a consuming iterator that moves out of the vector in
    /// reverse order. Also see `move_iter`, however note that this
    /// is more efficient.
    fn move_rev_iter(self) -> MoveRevIterator<T>;

    /**
     * Reserves capacity for exactly `n` elements in the given vector.
     *
     * If the capacity for `self` is already equal to or greater than the requested
     * capacity, then no action is taken.
     *
     * # Arguments
     *
     * * n - The number of elements to reserve space for
     *
     * # Failure
     *
     * This method always succeeds in reserving space for `n` elements, or it does
     * not return.
     */
    fn reserve(&mut self, n: uint);
    /**
     * Reserves capacity for at least `n` elements in the given vector.
     *
     * This function will over-allocate in order to amortize the allocation costs
     * in scenarios where the caller may need to repeatedly reserve additional
     * space.
     *
     * If the capacity for `self` is already equal to or greater than the requested
     * capacity, then no action is taken.
     *
     * # Arguments
     *
     * * n - The number of elements to reserve space for
     */
    fn reserve_at_least(&mut self, n: uint);
    /**
     * Reserves capacity for at least `n` additional elements in the given vector.
     *
     * # Failure
     *
     * Fails if the new required capacity overflows uint.
     *
     * May also fail if `reserve` fails.
     */
    fn reserve_additional(&mut self, n: uint);
    /// Returns the number of elements the vector can hold without reallocating.
    fn capacity(&self) -> uint;
    /// Shrink the capacity of the vector to match the length
    fn shrink_to_fit(&mut self);

    /// Append an element to a vector
    fn push(&mut self, t: T);
    /// Takes ownership of the vector `rhs`, moving all elements into
    /// the current vector. This does not copy any elements, and it is
    /// illegal to use the `rhs` vector after calling this method
    /// (because it is moved here).
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut a = ~[~1];
    /// a.push_all_move(~[~2, ~3, ~4]);
    /// assert!(a == ~[~1, ~2, ~3, ~4]);
    /// ```
    fn push_all_move(&mut self, rhs: ~[T]);
    /// Remove the last element from a vector and return it, failing if it is empty
    fn pop(&mut self) -> T;
    /// Remove the last element from a vector and return it, or `None` if it is empty
    fn pop_opt(&mut self) -> Option<T>;
    /// Removes the first element from a vector and return it
    fn shift(&mut self) -> T;
    /// Removes the first element from a vector and return it, or `None` if it is empty
    fn shift_opt(&mut self) -> Option<T>;
    /// Prepend an element to the vector
    fn unshift(&mut self, x: T);

    /// Insert an element at position i within v, shifting all
    /// elements after position i one position to the right.
    fn insert(&mut self, i: uint, x:T);

    /// Remove and return the element at position i within v, shifting
    /// all elements after position i one position to the left.
    fn remove(&mut self, i: uint) -> T;

    /**
     * Remove an element from anywhere in the vector and return it, replacing it
     * with the last element. This does not preserve ordering, but is O(1).
     *
     * Fails if index >= length.
     */
    fn swap_remove(&mut self, index: uint) -> T;

    /// Shorten a vector, dropping excess elements.
    fn truncate(&mut self, newlen: uint);

    /**
     * Like `filter()`, but in place.  Preserves order of `v`.  Linear time.
     */
    fn retain(&mut self, f: |t: &T| -> bool);
    /**
     * Partitions the vector into those that satisfies the predicate, and
     * those that do not.
     */
    fn partition(self, f: |&T| -> bool) -> (~[T], ~[T]);

    /**
     * Expands a vector in place, initializing the new elements to the result of
     * a function
     *
     * Function `init_op` is called `n` times with the values [0..`n`)
     *
     * # Arguments
     *
     * * n - The number of elements to add
     * * init_op - A function to call to retreive each appended element's
     *             value
     */
    fn grow_fn(&mut self, n: uint, op: |uint| -> T);
}

impl<T> OwnedVector<T> for ~[T] {
    fn move_iter(self) -> MoveIterator<T> {
        MoveIterator { v: self, idx: 0 }
    }
    fn move_rev_iter(self) -> MoveRevIterator<T> {
        MoveRevIterator { v: self }
    }

    fn reserve(&mut self, n: uint) {
        // Only make the (slow) call into the runtime if we have to
        if self.capacity() < n {
            unsafe {
                let td = get_tydesc::<T>();
                if owns_managed::<T>() {
                    let ptr: *mut *mut Box<Vec<()>> = cast::transmute(self);
                    ::at_vec::raw::reserve_raw(td, ptr, n);
                } else {
                    let ptr: *mut *mut Vec<()> = cast::transmute(self);
                    let alloc = n * mem::nonzero_size_of::<T>();
                    let size = alloc + mem::size_of::<Vec<()>>();
                    if alloc / mem::nonzero_size_of::<T>() != n || size < alloc {
                        fail!("vector size is too large: {}", n);
                    }
                    *ptr = realloc_raw(*ptr as *mut c_void, size)
                           as *mut Vec<()>;
                    (**ptr).alloc = alloc;
                }
            }
        }
    }

    #[inline]
    fn reserve_at_least(&mut self, n: uint) {
        self.reserve(uint::next_power_of_two_opt(n).unwrap_or(n));
    }

    #[inline]
    fn reserve_additional(&mut self, n: uint) {
        if self.capacity() - self.len() < n {
            match self.len().checked_add(&n) {
                None => fail!("vec::reserve_additional: `uint` overflow"),
                Some(new_cap) => self.reserve_at_least(new_cap)
            }
        }
    }

    #[inline]
    fn capacity(&self) -> uint {
        unsafe {
            if owns_managed::<T>() {
                let repr: **Box<Vec<()>> = cast::transmute(self);
                (**repr).data.alloc / mem::nonzero_size_of::<T>()
            } else {
                let repr: **Vec<()> = cast::transmute(self);
                (**repr).alloc / mem::nonzero_size_of::<T>()
            }
        }
    }

    fn shrink_to_fit(&mut self) {
        unsafe {
            let ptr: *mut *mut Vec<()> = cast::transmute(self);
            let alloc = (**ptr).fill;
            let size = alloc + mem::size_of::<Vec<()>>();
            *ptr = realloc_raw(*ptr as *mut c_void, size) as *mut Vec<()>;
            (**ptr).alloc = alloc;
        }
    }

    #[inline]
    fn push(&mut self, t: T) {
        unsafe {
            if owns_managed::<T>() {
                let repr: **Box<Vec<()>> = cast::transmute(&mut *self);
                let fill = (**repr).data.fill;
                if (**repr).data.alloc <= fill {
                    self.reserve_additional(1);
                }

                push_fast(self, t);
            } else {
                let repr: **Vec<()> = cast::transmute(&mut *self);
                let fill = (**repr).fill;
                if (**repr).alloc <= fill {
                    self.reserve_additional(1);
                }

                push_fast(self, t);
            }
        }

        // This doesn't bother to make sure we have space.
        #[inline] // really pretty please
        unsafe fn push_fast<T>(this: &mut ~[T], t: T) {
            if owns_managed::<T>() {
                let repr: **mut Box<Vec<u8>> = cast::transmute(this);
                let fill = (**repr).data.fill;
                (**repr).data.fill += mem::nonzero_size_of::<T>();
                let p = to_unsafe_ptr(&((**repr).data.data));
                let p = ptr::offset(p, fill as int) as *mut T;
                intrinsics::move_val_init(&mut(*p), t);
            } else {
                let repr: **mut Vec<u8> = cast::transmute(this);
                let fill = (**repr).fill;
                (**repr).fill += mem::nonzero_size_of::<T>();
                let p = to_unsafe_ptr(&((**repr).data));
                let p = ptr::offset(p, fill as int) as *mut T;
                intrinsics::move_val_init(&mut(*p), t);
            }
        }

    }

    #[inline]
    fn push_all_move(&mut self, mut rhs: ~[T]) {
        let self_len = self.len();
        let rhs_len = rhs.len();
        let new_len = self_len + rhs_len;
        self.reserve_additional(rhs.len());
        unsafe { // Note: infallible.
            let self_p = vec::raw::to_mut_ptr(*self);
            let rhs_p = vec::raw::to_ptr(rhs);
            ptr::copy_memory(ptr::mut_offset(self_p, self_len as int), rhs_p, rhs_len);
            raw::set_len(self, new_len);
            raw::set_len(&mut rhs, 0);
        }
    }

    fn pop_opt(&mut self) -> Option<T> {
        match self.len() {
            0  => None,
            ln => {
                let valptr = ptr::to_mut_unsafe_ptr(&mut self[ln - 1u]);
                unsafe {
                    raw::set_len(self, ln - 1u);
                    Some(ptr::read_ptr(&*valptr))
                }
            }
        }
    }


    #[inline]
    fn pop(&mut self) -> T {
        self.pop_opt().expect("pop: empty vector")
    }

    #[inline]
    fn shift(&mut self) -> T {
        self.shift_opt().expect("shift: empty vector")
    }

    fn shift_opt(&mut self) -> Option<T> {
        unsafe {
            let ln = match self.len() {
                0 => return None,
                1 => return self.pop_opt(),
                2 =>  {
                    let last = self.pop();
                    let first = self.pop_opt();
                    self.push(last);
                    return first;
                }
                x => x
            };

            let next_ln = self.len() - 1;

            // Save the last element. We're going to overwrite its position
            let work_elt = self.pop();
            // We still should have room to work where what last element was
            assert!(self.capacity() >= ln);
            // Pretend like we have the original length so we can use
            // the vector copy_memory to overwrite the hole we just made
            raw::set_len(self, ln);

            // Memcopy the head element (the one we want) to the location we just
            // popped. For the moment it unsafely exists at both the head and last
            // positions
            {
                let first_slice = self.slice(0, 1);
                let last_slice = self.slice(next_ln, ln);
                raw::copy_memory(cast::transmute(last_slice), first_slice, 1);
            }

            // Memcopy everything to the left one element
            {
                let init_slice = self.slice(0, next_ln);
                let tail_slice = self.slice(1, ln);
                raw::copy_memory(cast::transmute(init_slice),
                                 tail_slice,
                                 next_ln);
            }

            // Set the new length. Now the vector is back to normal
            raw::set_len(self, next_ln);

            // Swap out the element we want from the end
            let vp = raw::to_mut_ptr(*self);
            let vp = ptr::mut_offset(vp, (next_ln - 1) as int);

            Some(ptr::replace_ptr(vp, work_elt))
        }
    }

    fn unshift(&mut self, x: T) {
        let v = util::replace(self, ~[x]);
        self.push_all_move(v);
    }
    fn insert(&mut self, i: uint, x:T) {
        let len = self.len();
        assert!(i <= len);

        self.push(x);
        let mut j = len;
        while j > i {
            self.swap(j, j - 1);
            j -= 1;
        }
    }
    fn remove(&mut self, i: uint) -> T {
        let len = self.len();
        assert!(i < len);

        let mut j = i;
        while j < len - 1 {
            self.swap(j, j + 1);
            j += 1;
        }
        self.pop()
    }
    fn swap_remove(&mut self, index: uint) -> T {
        let ln = self.len();
        if index >= ln {
            fail!("vec::swap_remove - index {} >= length {}", index, ln);
        }
        if index < ln - 1 {
            self.swap(index, ln - 1);
        }
        self.pop()
    }
    fn truncate(&mut self, newlen: uint) {
        do self.as_mut_buf |p, oldlen| {
            assert!(newlen <= oldlen);
            unsafe {
                // This loop is optimized out for non-drop types.
                for i in range(newlen, oldlen) {
                    ptr::read_and_zero_ptr(ptr::mut_offset(p, i as int));
                }
            }
        }
        unsafe { raw::set_len(self, newlen); }
    }

    fn retain(&mut self, f: |t: &T| -> bool) {
        let len = self.len();
        let mut deleted: uint = 0;

        for i in range(0u, len) {
            if !f(&self[i]) {
                deleted += 1;
            } else if deleted > 0 {
                self.swap(i - deleted, i);
            }
        }

        if deleted > 0 {
            self.truncate(len - deleted);
        }
    }

    #[inline]
    fn partition(self, f: |&T| -> bool) -> (~[T], ~[T]) {
        let mut lefts  = ~[];
        let mut rights = ~[];

        for elt in self.move_iter() {
            if f(&elt) {
                lefts.push(elt);
            } else {
                rights.push(elt);
            }
        }

        (lefts, rights)
    }
    fn grow_fn(&mut self, n: uint, op: |uint| -> T) {
        let new_len = self.len() + n;
        self.reserve_at_least(new_len);
        let mut i: uint = 0u;
        while i < n {
            self.push(op(i));
            i += 1u;
        }
    }
}

impl<T> Mutable for ~[T] {
    /// Clear the vector, removing all values.
    fn clear(&mut self) { self.truncate(0) }
}

/// Extension methods for owned vectors containing `Clone` elements.
pub trait OwnedCopyableVector<T:Clone> {
    /// Iterates over the slice `rhs`, copies each element, and then appends it to
    /// the vector provided `v`. The `rhs` vector is traversed in-order.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut a = ~[1];
    /// a.push_all([2, 3, 4]);
    /// assert!(a == ~[1, 2, 3, 4]);
    /// ```
    fn push_all(&mut self, rhs: &[T]);

    /**
     * Expands a vector in place, initializing the new elements to a given value
     *
     * # Arguments
     *
     * * n - The number of elements to add
     * * initval - The value for the new elements
     */
    fn grow(&mut self, n: uint, initval: &T);

    /**
     * Sets the value of a vector element at a given index, growing the vector as
     * needed
     *
     * Sets the element at position `index` to `val`. If `index` is past the end
     * of the vector, expands the vector by replicating `initval` to fill the
     * intervening space.
     */
    fn grow_set(&mut self, index: uint, initval: &T, val: T);
}

impl<T:Clone> OwnedCopyableVector<T> for ~[T] {
    #[inline]
    fn push_all(&mut self, rhs: &[T]) {
        let new_len = self.len() + rhs.len();
        self.reserve(new_len);

        for elt in rhs.iter() {
            self.push((*elt).clone())
        }
    }
    fn grow(&mut self, n: uint, initval: &T) {
        let new_len = self.len() + n;
        self.reserve_at_least(new_len);
        let mut i: uint = 0u;

        while i < n {
            self.push((*initval).clone());
            i += 1u;
        }
    }
    fn grow_set(&mut self, index: uint, initval: &T, val: T) {
        let l = self.len();
        if index >= l { self.grow(index - l + 1u, initval); }
        self[index] = val;
    }
}

/// Extension methods for owned vectors containing `Eq` elements.
pub trait OwnedEqVector<T:Eq> {
    /**
    * Remove consecutive repeated elements from a vector; if the vector is
    * sorted, this removes all duplicates.
    */
    fn dedup(&mut self);
}

impl<T:Eq> OwnedEqVector<T> for ~[T] {
    fn dedup(&mut self) {
        unsafe {
            // Although we have a mutable reference to `self`, we cannot make
            // *arbitrary* changes. There exists the possibility that this
            // vector is contained with an `@mut` box and hence is still
            // readable by the outside world during the `Eq` comparisons.
            // Moreover, those comparisons could fail, so we must ensure
            // that the vector is in a valid state at all time.
            //
            // The way that we handle this is by using swaps; we iterate
            // over all the elements, swapping as we go so that at the end
            // the elements we wish to keep are in the front, and those we
            // wish to reject are at the back. We can then truncate the
            // vector. This operation is still O(n).
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
            // Comparing self[r] against self[w-1], tis is not a duplicate, so
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
            // Duplicate, advance r. End of vec. Truncate to w.

            let ln = self.len();
            if ln < 1 { return; }

            // Avoid bounds checks by using unsafe pointers.
            let p = vec::raw::to_mut_ptr(*self);
            let mut r = 1;
            let mut w = 1;

            while r < ln {
                let p_r = ptr::mut_offset(p, r as int);
                let p_wm1 = ptr::mut_offset(p, (w - 1) as int);
                if *p_r != *p_wm1 {
                    if r != w {
                        let p_w = ptr::mut_offset(p_wm1, 1);
                        util::swap(&mut *p_r, &mut *p_w);
                    }
                    w += 1;
                }
                r += 1;
            }

            self.truncate(w);
        }
    }
}

/// Extension methods for vectors such that their elements are
/// mutable.
pub trait MutableVector<'self, T> {
    /// Return a slice that points into another slice.
    fn mut_slice(self, start: uint, end: uint) -> &'self mut [T];
    /**
     * Returns a slice of self from `start` to the end of the vec.
     *
     * Fails when `start` points outside the bounds of self.
     */
    fn mut_slice_from(self, start: uint) -> &'self mut [T];
    /**
     * Returns a slice of self from the start of the vec to `end`.
     *
     * Fails when `end` points outside the bounds of self.
     */
    fn mut_slice_to(self, end: uint) -> &'self mut [T];
    /// Returns an iterator that allows modifying each value
    fn mut_iter(self) -> VecMutIterator<'self, T>;
    /// Returns a reversed iterator that allows modifying each value
    fn mut_rev_iter(self) -> MutRevIterator<'self, T>;

    /**
     * Swaps two elements in a vector
     *
     * # Arguments
     *
     * * a - The index of the first element
     * * b - The index of the second element
     */
    fn swap(self, a: uint, b: uint);

    /**
     * Divides one `&mut` into two. The first will
     * contain all indices from `0..mid` (excluding the index `mid`
     * itself) and the second will contain all indices from
     * `mid..len` (excluding the index `len` itself).
     */
    fn mut_split(self, mid: uint) -> (&'self mut [T],
                                      &'self mut [T]);

    /// Reverse the order of elements in a vector, in place
    fn reverse(self);

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
    unsafe fn unsafe_mut_ref(self, index: uint) -> *mut T;
    /// Unsafely sets the element in index to the value
    unsafe fn unsafe_set(self, index: uint, val: T);

    /// Similar to `as_imm_buf` but passing a `*mut T`
    fn as_mut_buf<U>(self, f: |*mut T, uint| -> U) -> U;
}

impl<'self,T> MutableVector<'self, T> for &'self mut [T] {
    #[inline]
    fn mut_slice(self, start: uint, end: uint) -> &'self mut [T] {
        assert!(start <= end);
        assert!(end <= self.len());
        do self.as_mut_buf |p, _len| {
            unsafe {
                cast::transmute(Slice {
                    data: ptr::mut_offset(p, start as int) as *T,
                    len: (end - start)
                })
            }
        }
    }

    #[inline]
    fn mut_slice_from(self, start: uint) -> &'self mut [T] {
        let len = self.len();
        self.mut_slice(start, len)
    }

    #[inline]
    fn mut_slice_to(self, end: uint) -> &'self mut [T] {
        self.mut_slice(0, end)
    }

    #[inline]
    fn mut_split(self, mid: uint) -> (&'self mut [T], &'self mut [T]) {
        unsafe {
            let len = self.len();
            let self2: &'self mut [T] = cast::transmute_copy(&self);
            (self.mut_slice(0, mid), self2.mut_slice(mid, len))
        }
    }

    #[inline]
    fn mut_iter(self) -> VecMutIterator<'self, T> {
        unsafe {
            let p = vec::raw::to_mut_ptr(self);
            if mem::size_of::<T>() == 0 {
                VecMutIterator{ptr: p,
                               end: (p as uint + self.len()) as *mut T,
                               lifetime: None}
            } else {
                VecMutIterator{ptr: p,
                               end: p.offset(self.len() as int),
                               lifetime: None}
            }
        }
    }

    #[inline]
    fn mut_rev_iter(self) -> MutRevIterator<'self, T> {
        self.mut_iter().invert()
    }

    fn swap(self, a: uint, b: uint) {
        unsafe {
            // Can't take two mutable loans from one vector, so instead just cast
            // them to their raw pointers to do the swap
            let pa: *mut T = &mut self[a];
            let pb: *mut T = &mut self[b];
            ptr::swap_ptr(pa, pb);
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
    fn move_from(self, mut src: ~[T], start: uint, end: uint) -> uint {
        for (a, b) in self.mut_iter().zip(src.mut_slice(start, end).mut_iter()) {
            util::swap(a, b);
        }
        cmp::min(self.len(), end-start)
    }

    #[inline]
    unsafe fn unsafe_mut_ref(self, index: uint) -> *mut T {
        ptr::mut_offset(self.repr().data as *mut T, index as int)
    }

    #[inline]
    unsafe fn unsafe_set(self, index: uint, val: T) {
        *self.unsafe_mut_ref(index) = val;
    }

    #[inline]
    fn as_mut_buf<U>(self, f: |*mut T, uint| -> U) -> U {
        let Slice{ data, len } = self.repr();
        f(data as *mut T, len)
    }
}

/// Trait for &[T] where T is Cloneable
pub trait MutableCloneableVector<T> {
    /// Copies as many elements from `src` as it can into `self`
    /// (the shorter of self.len() and src.len()). Returns the number of elements copied.
    fn copy_from(self, &[T]) -> uint;
}

impl<'self, T:Clone> MutableCloneableVector<T> for &'self mut [T] {
    #[inline]
    fn copy_from(self, src: &[T]) -> uint {
        for (a, b) in self.mut_iter().zip(src.iter()) {
            a.clone_from(b);
        }
        cmp::min(self.len(), src.len())
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
    use cast;
    use clone::Clone;
    use option::Some;
    use ptr;
    use mem;
    use unstable::intrinsics;
    use vec::{with_capacity, ImmutableVector, MutableVector};
    use unstable::raw::{Box, Vec, Slice};
    use unstable::intrinsics::owns_managed;

    /**
     * Sets the length of a vector
     *
     * This will explicitly set the size of the vector, without actually
     * modifying its buffers, so it is up to the caller to ensure that
     * the vector is actually the specified size.
     */
    #[inline]
    pub unsafe fn set_len<T>(v: &mut ~[T], new_len: uint) {
        if owns_managed::<T>() {
            let repr: **mut Box<Vec<()>> = cast::transmute(v);
            (**repr).data.fill = new_len * mem::nonzero_size_of::<T>();
        } else {
            let repr: **mut Vec<()> = cast::transmute(v);
            (**repr).fill = new_len * mem::nonzero_size_of::<T>();
        }
    }

    /**
     * Returns an unsafe pointer to the vector's buffer
     *
     * The caller must ensure that the vector outlives the pointer this
     * function returns, or else it will end up pointing to garbage.
     *
     * Modifying the vector may cause its buffer to be reallocated, which
     * would also make any pointers to it invalid.
     */
    #[inline]
    pub fn to_ptr<T>(v: &[T]) -> *T {
        v.repr().data
    }

    /** see `to_ptr()` */
    #[inline]
    pub fn to_mut_ptr<T>(v: &mut [T]) -> *mut T {
        v.repr().data as *mut T
    }

    /**
     * Form a slice from a pointer and length (as a number of units,
     * not bytes).
     */
    #[inline]
    pub unsafe fn buf_as_slice<T,U>(p: *T, len: uint, f: |v: &[T]| -> U)
                               -> U {
        f(cast::transmute(Slice {
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
        f(cast::transmute(Slice {
            data: p as *T,
            len: len
        }))
    }

    /**
     * Unchecked vector indexing.
     */
    #[inline]
    pub unsafe fn get<T:Clone>(v: &[T], i: uint) -> T {
        v.as_imm_buf(|p, _len| (*ptr::offset(p, i as int)).clone())
    }

    /**
     * Unchecked vector index assignment.  Does not drop the
     * old value and hence is only suitable when the vector
     * is newly allocated.
     */
    #[inline]
    pub unsafe fn init_elem<T>(v: &mut [T], i: uint, val: T) {
        let mut box = Some(val);
        do v.as_mut_buf |p, _len| {
            intrinsics::move_val_init(&mut(*ptr::mut_offset(p, i as int)),
                                      box.take_unwrap());
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
    // Was in raw, but needs to be called by net_tcp::on_tcp_read_cb
    #[inline]
    pub unsafe fn from_buf_raw<T>(ptr: *T, elts: uint) -> ~[T] {
        let mut dst = with_capacity(elts);
        set_len(&mut dst, elts);
        dst.as_mut_buf(|p_dst, _len_dst| ptr::copy_memory(p_dst, ptr, elts));
        dst
    }

    /**
      * Copies data from one vector to another.
      *
      * Copies `count` bytes from `src` to `dst`. The source and destination
      * may overlap.
      */
    #[inline]
    pub unsafe fn copy_memory<T>(dst: &mut [T], src: &[T],
                                 count: uint) {
        assert!(dst.len() >= count);
        assert!(src.len() >= count);

        do dst.as_mut_buf |p_dst, _len_dst| {
            do src.as_imm_buf |p_src, _len_src| {
                ptr::copy_memory(p_dst, p_src, count)
            }
        }
    }
}

/// Operations on `[u8]`
pub mod bytes {
    use libc;
    use num;
    use vec::raw;
    use vec;
    use ptr;

    /// A trait for operations on mutable operations on `[u8]`
    pub trait MutableByteVector {
        /// Sets all bytes of the receiver to the given value.
        fn set_memory(self, value: u8);
    }

    impl<'self> MutableByteVector for &'self mut [u8] {
        #[inline]
        fn set_memory(self, value: u8) {
            do self.as_mut_buf |p, len| {
                unsafe { ptr::set_memory(p, value, len) };
            }
        }
    }

    /// Bytewise string comparison
    pub fn memcmp(a: &~[u8], b: &~[u8]) -> int {
        let a_len = a.len();
        let b_len = b.len();
        let n = num::min(a_len, b_len) as libc::size_t;
        let r = unsafe {
            libc::memcmp(raw::to_ptr(*a) as *libc::c_void,
                         raw::to_ptr(*b) as *libc::c_void, n) as int
        };

        if r != 0 { r } else {
            if a_len == b_len {
                0
            } else if a_len < b_len {
                -1
            } else {
                1
            }
        }
    }

    /// Bytewise less than or equal
    pub fn lt(a: &~[u8], b: &~[u8]) -> bool { memcmp(a, b) < 0 }

    /// Bytewise less than or equal
    pub fn le(a: &~[u8], b: &~[u8]) -> bool { memcmp(a, b) <= 0 }

    /// Bytewise equality
    pub fn eq(a: &~[u8], b: &~[u8]) -> bool { memcmp(a, b) == 0 }

    /// Bytewise inequality
    pub fn ne(a: &~[u8], b: &~[u8]) -> bool { memcmp(a, b) != 0 }

    /// Bytewise greater than or equal
    pub fn ge(a: &~[u8], b: &~[u8]) -> bool { memcmp(a, b) >= 0 }

    /// Bytewise greater than
    pub fn gt(a: &~[u8], b: &~[u8]) -> bool { memcmp(a, b) > 0 }

    /**
      * Copies data from one vector to another.
      *
      * Copies `count` bytes from `src` to `dst`. The source and destination
      * may overlap.
      */
    #[inline]
    pub fn copy_memory(dst: &mut [u8], src: &[u8], count: uint) {
        // Bound checks are done at vec::raw::copy_memory.
        unsafe { vec::raw::copy_memory(dst, src, count) }
    }

    /**
     * Allocate space in `dst` and append the data in `src`.
     */
    #[inline]
    pub fn push_bytes(dst: &mut ~[u8], src: &[u8]) {
        let old_len = dst.len();
        dst.reserve_additional(src.len());
        unsafe {
            do dst.as_mut_buf |p_dst, len_dst| {
                do src.as_imm_buf |p_src, len_src| {
                    ptr::copy_memory(p_dst.offset(len_dst as int), p_src, len_src)
                }
            }
            vec::raw::set_len(dst, old_len + src.len());
        }
    }
}

impl<A: Clone> Clone for ~[A] {
    #[inline]
    fn clone(&self) -> ~[A] {
        self.iter().map(|item| item.clone()).collect()
    }

    fn clone_from(&mut self, source: &~[A]) {
        if self.len() < source.len() {
            *self = source.clone()
        } else {
            self.truncate(source.len());
            for (x, y) in self.mut_iter().zip(source.iter()) {
                x.clone_from(y);
            }
        }
    }
}

impl<A: DeepClone> DeepClone for ~[A] {
    #[inline]
    fn deep_clone(&self) -> ~[A] {
        self.iter().map(|item| item.deep_clone()).collect()
    }

    fn deep_clone_from(&mut self, source: &~[A]) {
        if self.len() < source.len() {
            *self = source.deep_clone()
        } else {
            self.truncate(source.len());
            for (x, y) in self.mut_iter().zip(source.iter()) {
                x.deep_clone_from(y);
            }
        }
    }
}

// This works because every lifetime is a sub-lifetime of 'static
impl<'self, A> Default for &'self [A] {
    fn default() -> &'self [A] { &'self [] }
}

impl<A> Default for ~[A] {
    fn default() -> ~[A] { ~[] }
}

impl<A> Default for @[A] {
    fn default() -> @[A] { @[] }
}

macro_rules! iterator {
    /* FIXME: #4375 Cannot attach documentation/attributes to a macro generated struct.
    (struct $name:ident -> $ptr:ty, $elem:ty) => {
        pub struct $name<'self, T> {
            priv ptr: $ptr,
            priv end: $ptr,
            priv lifetime: $elem // FIXME: #5922
        }
    };*/
    (impl $name:ident -> $elem:ty) => {
        impl<'self, T> Iterator<$elem> for $name<'self, T> {
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
                            cast::transmute(self.ptr as uint + 1)
                        } else {
                            self.ptr.offset(1)
                        };

                        Some(cast::transmute(old))
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
    }
}

macro_rules! double_ended_iterator {
    (impl $name:ident -> $elem:ty) => {
        impl<'self, T> DoubleEndedIterator<$elem> for $name<'self, T> {
            #[inline]
            fn next_back(&mut self) -> Option<$elem> {
                // could be implemented with slices, but this avoids bounds checks
                unsafe {
                    if self.end == self.ptr {
                        None
                    } else {
                        self.end = if mem::size_of::<T>() == 0 {
                            // See above for why 'ptr.offset' isn't used
                            cast::transmute(self.end as uint - 1)
                        } else {
                            self.end.offset(-1)
                        };
                        Some(cast::transmute(self.end))
                    }
                }
            }
        }
    }
}

impl<'self, T> RandomAccessIterator<&'self T> for VecIterator<'self, T> {
    #[inline]
    fn indexable(&self) -> uint {
        let (exact, _) = self.size_hint();
        exact
    }

    #[inline]
    fn idx(&self, index: uint) -> Option<&'self T> {
        unsafe {
            if index < self.indexable() {
                cast::transmute(self.ptr.offset(index as int))
            } else {
                None
            }
        }
    }
}

//iterator!{struct VecIterator -> *T, &'self T}
/// An iterator for iterating over a vector.
pub struct VecIterator<'self, T> {
    priv ptr: *T,
    priv end: *T,
    priv lifetime: Option<&'self ()> // FIXME: #5922
}
iterator!{impl VecIterator -> &'self T}
double_ended_iterator!{impl VecIterator -> &'self T}
pub type RevIterator<'self, T> = Invert<VecIterator<'self, T>>;

impl<'self, T> ExactSize<&'self T> for VecIterator<'self, T> {}
impl<'self, T> ExactSize<&'self mut T> for VecMutIterator<'self, T> {}

impl<'self, T> Clone for VecIterator<'self, T> {
    fn clone(&self) -> VecIterator<'self, T> { *self }
}

//iterator!{struct VecMutIterator -> *mut T, &'self mut T}
/// An iterator for mutating the elements of a vector.
pub struct VecMutIterator<'self, T> {
    priv ptr: *mut T,
    priv end: *mut T,
    priv lifetime: Option<&'self mut ()> // FIXME: #5922
}
iterator!{impl VecMutIterator -> &'self mut T}
double_ended_iterator!{impl VecMutIterator -> &'self mut T}
pub type MutRevIterator<'self, T> = Invert<VecMutIterator<'self, T>>;

/// An iterator that moves out of a vector.
#[deriving(Clone)]
pub struct MoveIterator<T> {
    priv v: ~[T],
    priv idx: uint,
}

impl<T> Iterator<T> for MoveIterator<T> {
    #[inline]
    fn next(&mut self) -> Option<T> {
        // this is peculiar, but is required for safety with respect
        // to dtors. It traverses the first half of the vec, and
        // removes them by swapping them with the last element (and
        // popping), which results in the second half in reverse
        // order, and so these can just be pop'd off. That is,
        //
        // [1,2,3,4,5] => 1, [5,2,3,4] => 2, [5,4,3] => 3, [5,4] => 4,
        // [5] -> 5, []
        let l = self.v.len();
        if self.idx < l {
            self.v.swap(self.idx, l - 1);
            self.idx += 1;
        }

        self.v.pop_opt()
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let l = self.v.len();
        (l, Some(l))
    }
}

/// An iterator that moves out of a vector in reverse order.
#[deriving(Clone)]
pub struct MoveRevIterator<T> {
    priv v: ~[T]
}

impl<T> Iterator<T> for MoveRevIterator<T> {
    #[inline]
    fn next(&mut self) -> Option<T> {
        self.v.pop_opt()
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let l = self.v.len();
        (l, Some(l))
    }
}

impl<A> FromIterator<A> for ~[A] {
    fn from_iterator<T: Iterator<A>>(iterator: &mut T) -> ~[A] {
        let (lower, _) = iterator.size_hint();
        let mut xs = with_capacity(lower);
        for x in *iterator {
            xs.push(x);
        }
        xs
    }
}

impl<A> Extendable<A> for ~[A] {
    fn extend<T: Iterator<A>>(&mut self, iterator: &mut T) {
        let (lower, _) = iterator.size_hint();
        let len = self.len();
        self.reserve(len + lower);
        for x in *iterator {
            self.push(x);
        }
    }
}

#[cfg(test)]
mod tests {
    use option::{None, Option, Some};
    use mem;
    use vec::*;
    use cmp::*;
    use prelude::*;

    fn square(n: uint) -> uint { n * n }

    fn square_ref(n: &uint) -> uint { square(*n) }

    fn is_three(n: &uint) -> bool { *n == 3u }

    fn is_odd(n: &uint) -> bool { *n % 2u == 1u }

    fn is_equal(x: &uint, y:&uint) -> bool { *x == *y }

    fn square_if_odd_r(n: &uint) -> Option<uint> {
        if *n % 2u == 1u { Some(*n * *n) } else { None }
    }

    fn square_if_odd_v(n: uint) -> Option<uint> {
        if n % 2u == 1u { Some(n * n) } else { None }
    }

    fn add(x: uint, y: &uint) -> uint { x + *y }

    #[test]
    fn test_unsafe_ptrs() {
        unsafe {
            // Test on-stack copy-from-buf.
            let a = ~[1, 2, 3];
            let mut ptr = raw::to_ptr(a);
            let b = from_buf(ptr, 3u);
            assert_eq!(b.len(), 3u);
            assert_eq!(b[0], 1);
            assert_eq!(b[1], 2);
            assert_eq!(b[2], 3);

            // Test on-heap copy-from-buf.
            let c = ~[1, 2, 3, 4, 5];
            ptr = raw::to_ptr(c);
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
        let mut v = from_fn(3u, square);
        assert_eq!(v.len(), 3u);
        assert_eq!(v[0], 0u);
        assert_eq!(v[1], 1u);
        assert_eq!(v[2], 4u);

        // Test on-heap from_fn.
        v = from_fn(5u, square);
        assert_eq!(v.len(), 5u);
        assert_eq!(v[0], 0u);
        assert_eq!(v[1], 1u);
        assert_eq!(v[2], 4u);
        assert_eq!(v[3], 9u);
        assert_eq!(v[4], 16u);
    }

    #[test]
    fn test_from_elem() {
        // Test on-stack from_elem.
        let mut v = from_elem(2u, 10u);
        assert_eq!(v.len(), 2u);
        assert_eq!(v[0], 10u);
        assert_eq!(v[1], 10u);

        // Test on-heap from_elem.
        v = from_elem(6u, 20u);
        assert_eq!(v[0], 20u);
        assert_eq!(v[1], 20u);
        assert_eq!(v[2], 20u);
        assert_eq!(v[3], 20u);
        assert_eq!(v[4], 20u);
        assert_eq!(v[5], 20u);
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
    fn test_get_opt() {
        let mut a = ~[11];
        assert_eq!(a.get_opt(1), None);
        a = ~[11, 12];
        assert_eq!(a.get_opt(1).unwrap(), &12);
        a = ~[11, 12, 13];
        assert_eq!(a.get_opt(1).unwrap(), &12);
    }

    #[test]
    fn test_head() {
        let mut a = ~[11];
        assert_eq!(a.head(), &11);
        a = ~[11, 12];
        assert_eq!(a.head(), &11);
    }

    #[test]
    #[should_fail]
    fn test_head_empty() {
        let a: ~[int] = ~[];
        a.head();
    }

    #[test]
    fn test_head_opt() {
        let mut a = ~[];
        assert_eq!(a.head_opt(), None);
        a = ~[11];
        assert_eq!(a.head_opt().unwrap(), &11);
        a = ~[11, 12];
        assert_eq!(a.head_opt().unwrap(), &11);
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
        let mut a = ~[11];
        assert_eq!(a.last(), &11);
        a = ~[11, 12];
        assert_eq!(a.last(), &12);
    }

    #[test]
    #[should_fail]
    fn test_last_empty() {
        let a: ~[int] = ~[];
        a.last();
    }

    #[test]
    fn test_last_opt() {
        let mut a = ~[];
        assert_eq!(a.last_opt(), None);
        a = ~[11];
        assert_eq!(a.last_opt().unwrap(), &11);
        a = ~[11, 12];
        assert_eq!(a.last_opt().unwrap(), &12);
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

        // Test on managed heap.
        let vec_managed = @[1, 2, 3, 4, 5];
        let v_c = vec_managed.slice(0u, 3u).to_owned();
        assert_eq!(v_c.len(), 3u);
        assert_eq!(v_c[0], 1);
        assert_eq!(v_c[1], 2);
        assert_eq!(v_c[2], 3);

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
        // Test on-heap pop.
        let mut v = ~[1, 2, 3, 4, 5];
        let e = v.pop();
        assert_eq!(v.len(), 4u);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);
        assert_eq!(v[3], 4);
        assert_eq!(e, 5);
    }

    #[test]
    fn test_pop_opt() {
        let mut v = ~[5];
        let e = v.pop_opt();
        assert_eq!(v.len(), 0);
        assert_eq!(e, Some(5));
        let f = v.pop_opt();
        assert_eq!(f, None);
        let g = v.pop_opt();
        assert_eq!(g, None);
    }

    fn test_swap_remove() {
        let mut v = ~[1, 2, 3, 4, 5];
        let mut e = v.swap_remove(0);
        assert_eq!(v.len(), 4);
        assert_eq!(e, 1);
        assert_eq!(v[0], 5);
        e = v.swap_remove(3);
        assert_eq!(v.len(), 3);
        assert_eq!(e, 4);
        assert_eq!(v[0], 5);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);
    }

    #[test]
    fn test_swap_remove_noncopyable() {
        // Tests that we don't accidentally run destructors twice.
        let mut v = ~[::unstable::sync::Exclusive::new(()),
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
        let mut v = ~[];
        v.push(1);
        assert_eq!(v.len(), 1u);
        assert_eq!(v[0], 1);

        // Test on-heap push().
        v.push(2);
        assert_eq!(v.len(), 2u);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
    }

    #[test]
    fn test_grow() {
        // Test on-stack grow().
        let mut v = ~[];
        v.grow(2u, &1);
        assert_eq!(v.len(), 2u);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 1);

        // Test on-heap grow().
        v.grow(3u, &2);
        assert_eq!(v.len(), 5u);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 1);
        assert_eq!(v[2], 2);
        assert_eq!(v[3], 2);
        assert_eq!(v[4], 2);
    }

    #[test]
    fn test_grow_fn() {
        let mut v = ~[];
        v.grow_fn(3u, square);
        assert_eq!(v.len(), 3u);
        assert_eq!(v[0], 0u);
        assert_eq!(v[1], 1u);
        assert_eq!(v[2], 4u);
    }

    #[test]
    fn test_grow_set() {
        let mut v = ~[1, 2, 3];
        v.grow_set(4u, &4, 5);
        assert_eq!(v.len(), 5u);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);
        assert_eq!(v[3], 4);
        assert_eq!(v[4], 5);
    }

    #[test]
    fn test_truncate() {
        let mut v = ~[@6,@5,@4];
        v.truncate(1);
        assert_eq!(v.len(), 1);
        assert_eq!(*(v[0]), 6);
        // If the unsafe block didn't drop things properly, we blow up here.
    }

    #[test]
    fn test_clear() {
        let mut v = ~[@6,@5,@4];
        v.clear();
        assert_eq!(v.len(), 0);
        // If the unsafe block didn't drop things properly, we blow up here.
    }

    #[test]
    fn test_dedup() {
        fn case(a: ~[uint], b: ~[uint]) {
            let mut v = a;
            v.dedup();
            assert_eq!(v, b);
        }
        case(~[], ~[]);
        case(~[1], ~[1]);
        case(~[1,1], ~[1]);
        case(~[1,2,3], ~[1,2,3]);
        case(~[1,1,2,3], ~[1,2,3]);
        case(~[1,2,2,3], ~[1,2,3]);
        case(~[1,2,3,3], ~[1,2,3]);
        case(~[1,1,2,2,2,3,3], ~[1,2,3]);
    }

    #[test]
    fn test_dedup_unique() {
        let mut v0 = ~[~1, ~1, ~2, ~3];
        v0.dedup();
        let mut v1 = ~[~1, ~2, ~2, ~3];
        v1.dedup();
        let mut v2 = ~[~1, ~2, ~3, ~3];
        v2.dedup();
        /*
         * If the ~pointers were leaked or otherwise misused, valgrind and/or
         * rustrt should raise errors.
         */
    }

    #[test]
    fn test_dedup_shared() {
        let mut v0 = ~[@1, @1, @2, @3];
        v0.dedup();
        let mut v1 = ~[@1, @2, @2, @3];
        v1.dedup();
        let mut v2 = ~[@1, @2, @3, @3];
        v2.dedup();
        /*
         * If the @pointers were leaked or otherwise misused, valgrind and/or
         * rustrt should raise errors.
         */
    }

    #[test]
    fn test_map() {
        // Test on-stack map.
        let v = &[1u, 2u, 3u];
        let mut w = v.map(square_ref);
        assert_eq!(w.len(), 3u);
        assert_eq!(w[0], 1u);
        assert_eq!(w[1], 4u);
        assert_eq!(w[2], 9u);

        // Test on-heap map.
        let v = ~[1u, 2u, 3u, 4u, 5u];
        w = v.map(square_ref);
        assert_eq!(w.len(), 5u);
        assert_eq!(w[0], 1u);
        assert_eq!(w[1], 4u);
        assert_eq!(w[2], 9u);
        assert_eq!(w[3], 16u);
        assert_eq!(w[4], 25u);
    }

    #[test]
    fn test_retain() {
        let mut v = ~[1, 2, 3, 4, 5];
        v.retain(is_odd);
        assert_eq!(v, ~[1, 3, 5]);
    }

    #[test]
    fn test_zip_unzip() {
        let z1 = ~[(1, 4), (2, 5), (3, 6)];

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
                0 => assert_eq!(v, [1, 3, 2]),
                1 => assert_eq!(v, [3, 1, 2]),
                2 => assert_eq!(v, [3, 2, 1]),
                3 => assert_eq!(v, [2, 3, 1]),
                4 => assert_eq!(v, [2, 1, 3]),
                5 => assert_eq!(v, [1, 2, 3]),
                _ => fail!(),
            }
        }
    }

    #[test]
    fn test_permutations() {
        use hashmap;
        {
            let v: [int, ..0] = [];
            let mut it = v.permutations_iter();
            assert_eq!(it.next(), None);
        }
        {
            let v = [~"Hello"];
            let mut it = v.permutations_iter();
            assert_eq!(it.next(), None);
        }
        {
            let v = [1, 2, 3];
            let mut it = v.permutations_iter();
            assert_eq!(it.next(), Some(~[1,2,3]));
            assert_eq!(it.next(), Some(~[1,3,2]));
            assert_eq!(it.next(), Some(~[3,1,2]));
            assert_eq!(it.next(), Some(~[3,2,1]));
            assert_eq!(it.next(), Some(~[2,3,1]));
            assert_eq!(it.next(), Some(~[2,1,3]));
            assert_eq!(it.next(), None);
        }
        {
            // check that we have N! unique permutations
            let mut set = hashmap::HashSet::new();
            let v = ['A', 'B', 'C', 'D', 'E', 'F'];
            for perm in v.permutations_iter() {
                set.insert(perm);
            }
            assert_eq!(set.len(), 2 * 3 * 4 * 5 * 6);
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
        let mut x = ~[1, 2, 3];
        assert_eq!(x.shift(), 1);
        assert_eq!(&x, &~[2, 3]);
        assert_eq!(x.shift(), 2);
        assert_eq!(x.shift(), 3);
        assert_eq!(x.len(), 0);
    }

    #[test]
    fn test_shift_opt() {
        let mut x = ~[1, 2, 3];
        assert_eq!(x.shift_opt(), Some(1));
        assert_eq!(&x, &~[2, 3]);
        assert_eq!(x.shift_opt(), Some(2));
        assert_eq!(x.shift_opt(), Some(3));
        assert_eq!(x.shift_opt(), None);
        assert_eq!(x.len(), 0);
    }

    #[test]
    fn test_unshift() {
        let mut x = ~[1, 2, 3];
        x.unshift(0);
        assert_eq!(x, ~[0, 1, 2, 3]);
    }

    #[test]
    fn test_insert() {
        let mut a = ~[1, 2, 4];
        a.insert(2, 3);
        assert_eq!(a, ~[1, 2, 3, 4]);

        let mut a = ~[1, 2, 3];
        a.insert(0, 0);
        assert_eq!(a, ~[0, 1, 2, 3]);

        let mut a = ~[1, 2, 3];
        a.insert(3, 4);
        assert_eq!(a, ~[1, 2, 3, 4]);

        let mut a = ~[];
        a.insert(0, 1);
        assert_eq!(a, ~[1]);
    }

    #[test]
    #[should_fail]
    fn test_insert_oob() {
        let mut a = ~[1, 2, 3];
        a.insert(4, 5);
    }

    #[test]
    fn test_remove() {
        let mut a = ~[1, 2, 3, 4];
        a.remove(2);
        assert_eq!(a, ~[1, 2, 4]);

        let mut a = ~[1, 2, 3];
        a.remove(0);
        assert_eq!(a, ~[2, 3]);

        let mut a = ~[1];
        a.remove(0);
        assert_eq!(a, ~[]);
    }

    #[test]
    #[should_fail]
    fn test_remove_oob() {
        let mut a = ~[1, 2, 3];
        a.remove(3);
    }

    #[test]
    fn test_capacity() {
        let mut v = ~[0u64];
        v.reserve(10u);
        assert_eq!(v.capacity(), 10u);
        let mut v = ~[0u32];
        v.reserve(10u);
        assert_eq!(v.capacity(), 10u);
    }

    #[test]
    fn test_slice_2() {
        let v = ~[1, 2, 3, 4, 5];
        let v = v.slice(1u, 3u);
        assert_eq!(v.len(), 2u);
        assert_eq!(v[0], 2);
        assert_eq!(v[1], 3);
    }


    #[test]
    #[should_fail]
    fn test_from_fn_fail() {
        do from_fn(100) |v| {
            if v == 50 { fail!() }
            (~0, @0)
        };
    }

    #[test]
    #[should_fail]
    fn test_from_elem_fail() {
        use cast;

        struct S {
            f: int,
            boxes: (~int, @int)
        }

        impl Clone for S {
            fn clone(&self) -> S {
                let s = unsafe { cast::transmute_mut(self) };
                s.f += 1;
                if s.f == 10 { fail!() }
                S { f: s.f, boxes: s.boxes.clone() }
            }
        }

        let s = S { f: 0, boxes: (~0, @0) };
        let _ = from_elem(100, s);
    }

    #[test]
    #[should_fail]
    fn test_build_fail() {
        do build(None) |push| {
            push((~0, @0));
            push((~0, @0));
            push((~0, @0));
            push((~0, @0));
            fail!();
        };
    }

    #[test]
    #[should_fail]
    fn test_grow_fn_fail() {
        let mut v = ~[];
        do v.grow_fn(100) |i| {
            if i == 50 {
                fail!()
            }
            (~0, @0)
        }
    }

    #[test]
    #[should_fail]
    fn test_map_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do v.map |_elt| {
            if i == 2 {
                fail!()
            }
            i += 1;
            ~[(~0, @0)]
        };
    }

    #[test]
    #[should_fail]
    fn test_flat_map_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do flat_map(v) |_elt| {
            if i == 2 {
                fail!()
            }
            i += 1;
            ~[(~0, @0)]
        };
    }

    #[test]
    #[should_fail]
    fn test_permute_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        for _ in v.permutations_iter() {
            if i == 2 {
                fail!()
            }
            i += 1;
        }
    }

    #[test]
    #[should_fail]
    fn test_as_imm_buf_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        do v.as_imm_buf |_buf, _i| {
            fail!()
        }
    }

    #[test]
    #[should_fail]
    fn test_as_mut_buf_fail() {
        let mut v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        do v.as_mut_buf |_buf, _i| {
            fail!()
        }
    }

    #[test]
    #[should_fail]
    fn test_copy_memory_oob() {
        unsafe {
            let mut a = [1, 2, 3, 4];
            let b = [1, 2, 3, 4, 5];
            raw::copy_memory(a, b, 5);
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
        assert_eq!(xs, [2, 3, 4, 5, 6])
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
        assert_eq!(xs, [5, 5, 5, 5, 5])
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
    fn test_split_iterator() {
        let xs = &[1i,2,3,4,5];

        assert_eq!(xs.split_iter(|x| *x % 2 == 0).collect::<~[&[int]]>(),
                   ~[&[1], &[3], &[5]]);
        assert_eq!(xs.split_iter(|x| *x == 1).collect::<~[&[int]]>(),
                   ~[&[], &[2,3,4,5]]);
        assert_eq!(xs.split_iter(|x| *x == 5).collect::<~[&[int]]>(),
                   ~[&[1,2,3,4], &[]]);
        assert_eq!(xs.split_iter(|x| *x == 10).collect::<~[&[int]]>(),
                   ~[&[1,2,3,4,5]]);
        assert_eq!(xs.split_iter(|_| true).collect::<~[&[int]]>(),
                   ~[&[], &[], &[], &[], &[], &[]]);

        let xs: &[int] = &[];
        assert_eq!(xs.split_iter(|x| *x == 5).collect::<~[&[int]]>(), ~[&[]]);
    }

    #[test]
    fn test_splitn_iterator() {
        let xs = &[1i,2,3,4,5];

        assert_eq!(xs.splitn_iter(0, |x| *x % 2 == 0).collect::<~[&[int]]>(),
                   ~[&[1,2,3,4,5]]);
        assert_eq!(xs.splitn_iter(1, |x| *x % 2 == 0).collect::<~[&[int]]>(),
                   ~[&[1], &[3,4,5]]);
        assert_eq!(xs.splitn_iter(3, |_| true).collect::<~[&[int]]>(),
                   ~[&[], &[], &[], &[4,5]]);

        let xs: &[int] = &[];
        assert_eq!(xs.splitn_iter(1, |x| *x == 5).collect::<~[&[int]]>(), ~[&[]]);
    }

    #[test]
    fn test_rsplit_iterator() {
        let xs = &[1i,2,3,4,5];

        assert_eq!(xs.rsplit_iter(|x| *x % 2 == 0).collect::<~[&[int]]>(),
                   ~[&[5], &[3], &[1]]);
        assert_eq!(xs.rsplit_iter(|x| *x == 1).collect::<~[&[int]]>(),
                   ~[&[2,3,4,5], &[]]);
        assert_eq!(xs.rsplit_iter(|x| *x == 5).collect::<~[&[int]]>(),
                   ~[&[], &[1,2,3,4]]);
        assert_eq!(xs.rsplit_iter(|x| *x == 10).collect::<~[&[int]]>(),
                   ~[&[1,2,3,4,5]]);

        let xs: &[int] = &[];
        assert_eq!(xs.rsplit_iter(|x| *x == 5).collect::<~[&[int]]>(), ~[&[]]);
    }

    #[test]
    fn test_rsplitn_iterator() {
        let xs = &[1,2,3,4,5];

        assert_eq!(xs.rsplitn_iter(0, |x| *x % 2 == 0).collect::<~[&[int]]>(),
                   ~[&[1,2,3,4,5]]);
        assert_eq!(xs.rsplitn_iter(1, |x| *x % 2 == 0).collect::<~[&[int]]>(),
                   ~[&[5], &[1,2,3]]);
        assert_eq!(xs.rsplitn_iter(3, |_| true).collect::<~[&[int]]>(),
                   ~[&[], &[], &[], &[1,2]]);

        let xs: &[int] = &[];
        assert_eq!(xs.rsplitn_iter(1, |x| *x == 5).collect::<~[&[int]]>(), ~[&[]]);
    }

    #[test]
    fn test_window_iterator() {
        let v = &[1i,2,3,4];

        assert_eq!(v.window_iter(2).collect::<~[&[int]]>(), ~[&[1,2], &[2,3], &[3,4]]);
        assert_eq!(v.window_iter(3).collect::<~[&[int]]>(), ~[&[1i,2,3], &[2,3,4]]);
        assert!(v.window_iter(6).next().is_none());
    }

    #[test]
    #[should_fail]
    fn test_window_iterator_0() {
        let v = &[1i,2,3,4];
        let _it = v.window_iter(0);
    }

    #[test]
    fn test_chunk_iterator() {
        let v = &[1i,2,3,4,5];

        assert_eq!(v.chunk_iter(2).collect::<~[&[int]]>(), ~[&[1i,2], &[3,4], &[5]]);
        assert_eq!(v.chunk_iter(3).collect::<~[&[int]]>(), ~[&[1i,2,3], &[4,5]]);
        assert_eq!(v.chunk_iter(6).collect::<~[&[int]]>(), ~[&[1i,2,3,4,5]]);

        assert_eq!(v.chunk_iter(2).invert().collect::<~[&[int]]>(), ~[&[5i], &[3,4], &[1,2]]);
        let it = v.chunk_iter(2);
        assert_eq!(it.indexable(), 3);
        assert_eq!(it.idx(0).unwrap(), &[1,2]);
        assert_eq!(it.idx(1).unwrap(), &[3,4]);
        assert_eq!(it.idx(2).unwrap(), &[5]);
        assert_eq!(it.idx(3), None);
    }

    #[test]
    #[should_fail]
    fn test_chunk_iterator_0() {
        let v = &[1i,2,3,4];
        let _it = v.chunk_iter(0);
    }

    #[test]
    fn test_move_from() {
        let mut a = [1,2,3,4,5];
        let b = ~[6,7,8];
        assert_eq!(a.move_from(b, 0, 3), 3);
        assert_eq!(a, [6,7,8,4,5]);
        let mut a = [7,2,8,1];
        let b = ~[3,1,4,1,5,9];
        assert_eq!(a.move_from(b, 0, 6), 4);
        assert_eq!(a, [3,1,4,1]);
        let mut a = [1,2,3,4];
        let b = ~[5,6,7,8,9,0];
        assert_eq!(a.move_from(b, 2, 3), 1);
        assert_eq!(a, [7,2,3,4]);
        let mut a = [1,2,3,4,5];
        let b = ~[5,6,7,8,9,0];
        assert_eq!(a.mut_slice(2,4).move_from(b,1,6), 2);
        assert_eq!(a, [1,2,6,7,5]);
    }

    #[test]
    fn test_copy_from() {
        let mut a = [1,2,3,4,5];
        let b = [6,7,8];
        assert_eq!(a.copy_from(b), 3);
        assert_eq!(a, [6,7,8,4,5]);
        let mut c = [7,2,8,1];
        let d = [3,1,4,1,5,9];
        assert_eq!(c.copy_from(d), 4);
        assert_eq!(c, [3,1,4,1]);
    }

    #[test]
    fn test_reverse_part() {
        let mut values = [1,2,3,4,5];
        values.mut_slice(1, 4).reverse();
        assert_eq!(values, [1,4,3,2,5]);
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
        t!(@[int]);
        t!(~[int]);
    }

    #[test]
    fn test_bytes_set_memory() {
        use vec::bytes::MutableByteVector;
        let mut values = [1u8,2,3,4,5];
        values.mut_slice(0,5).set_memory(0xAB);
        assert_eq!(values, [0xAB, 0xAB, 0xAB, 0xAB, 0xAB]);
        values.mut_slice(2,4).set_memory(0xFF);
        assert_eq!(values, [0xAB, 0xAB, 0xFF, 0xFF, 0xAB]);
    }

    #[test]
    #[should_fail]
    fn test_overflow_does_not_cause_segfault() {
        let mut v = ~[];
        v.reserve(-1);
        v.push(1);
        v.push(2);
    }

    #[test]
    #[should_fail]
    fn test_overflow_does_not_cause_segfault_managed() {
        let mut v = ~[@1];
        v.reserve(-1);
        v.push(@2);
    }

    #[test]
    fn test_mut_split() {
        let mut values = [1u8,2,3,4,5];
        {
            let (left, right) = values.mut_split(2);
            assert_eq!(left.slice(0, left.len()), [1, 2]);
            for p in left.mut_iter() {
                *p += 1;
            }

            assert_eq!(right.slice(0, right.len()), [3, 4, 5]);
            for p in right.mut_iter() {
                *p += 2;
            }
        }

        assert_eq!(values, [2, 3, 5, 6, 7]);
    }

    #[deriving(Clone, Eq)]
    struct Foo;

    #[test]
    fn test_iter_zero_sized() {
        let mut v = ~[Foo, Foo, Foo];
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

        let xs = ~[Foo, Foo, Foo];
        assert_eq!(format!("{:?}", xs.slice(0, 2).to_owned()),
                   ~"~[vec::tests::Foo, vec::tests::Foo]");

        let xs: [Foo, ..3] = [Foo, Foo, Foo];
        assert_eq!(format!("{:?}", xs.slice(0, 2).to_owned()),
                   ~"~[vec::tests::Foo, vec::tests::Foo]");
        cnt = 0;
        for f in xs.iter() {
            assert!(*f == Foo);
            cnt += 1;
        }
        assert!(cnt == 3);
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut xs = ~[0, 1, 2, 3];
        for i in range(4, 100) {
            xs.push(i)
        }
        assert_eq!(xs.capacity(), 128);
        xs.shrink_to_fit();
        assert_eq!(xs.capacity(), 100);
        assert_eq!(xs, range(0, 100).to_owned_vec());
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
}

#[cfg(test)]
mod bench {
    use extra::test::BenchHarness;
    use iter::range;
    use vec;
    use vec::VectorVector;
    use option::*;

    #[bench]
    fn iterator(bh: &mut BenchHarness) {
        // peculiar numbers to stop LLVM from optimising the summation
        // out.
        let v = vec::from_fn(100, |i| i ^ (i << 1) ^ (i >> 1));

        do bh.iter {
            let mut sum = 0;
            for x in v.iter() {
                sum += *x;
            }
            // sum == 11806, to stop dead code elimination.
            if sum == 0 {fail!()}
        }
    }

    #[bench]
    fn mut_iterator(bh: &mut BenchHarness) {
        let mut v = vec::from_elem(100, 0);

        do bh.iter {
            let mut i = 0;
            for x in v.mut_iter() {
                *x = i;
                i += 1;
            }
        }
    }

    #[bench]
    fn add(bh: &mut BenchHarness) {
        let xs: &[int] = [5, ..10];
        let ys: &[int] = [5, ..10];
        do bh.iter() {
            xs + ys;
        }
    }

    #[bench]
    fn concat(bh: &mut BenchHarness) {
        let xss: &[~[uint]] = vec::from_fn(100, |i| range(0, i).collect());
        do bh.iter {
            xss.concat_vec();
        }
    }

    #[bench]
    fn connect(bh: &mut BenchHarness) {
        let xss: &[~[uint]] = vec::from_fn(100, |i| range(0, i).collect());
        do bh.iter {
            xss.connect_vec(&0);
        }
    }

    #[bench]
    fn push(bh: &mut BenchHarness) {
        let mut vec: ~[uint] = ~[0u];
        do bh.iter() {
            vec.push(0);
        }
    }

    #[bench]
    fn starts_with_same_vector(bh: &mut BenchHarness) {
        let vec: ~[uint] = vec::from_fn(100, |i| i);
        do bh.iter() {
            vec.starts_with(vec);
        }
    }

    #[bench]
    fn starts_with_single_element(bh: &mut BenchHarness) {
        let vec: ~[uint] = ~[0u];
        do bh.iter() {
            vec.starts_with(vec);
        }
    }

    #[bench]
    fn starts_with_diff_one_element_at_end(bh: &mut BenchHarness) {
        let vec: ~[uint] = vec::from_fn(100, |i| i);
        let mut match_vec: ~[uint] = vec::from_fn(99, |i| i);
        match_vec.push(0);
        do bh.iter() {
            vec.starts_with(match_vec);
        }
    }

    #[bench]
    fn ends_with_same_vector(bh: &mut BenchHarness) {
        let vec: ~[uint] = vec::from_fn(100, |i| i);
        do bh.iter() {
            vec.ends_with(vec);
        }
    }

    #[bench]
    fn ends_with_single_element(bh: &mut BenchHarness) {
        let vec: ~[uint] = ~[0u];
        do bh.iter() {
            vec.ends_with(vec);
        }
    }

    #[bench]
    fn ends_with_diff_one_element_at_beginning(bh: &mut BenchHarness) {
        let vec: ~[uint] = vec::from_fn(100, |i| i);
        let mut match_vec: ~[uint] = vec::from_fn(100, |i| i);
        match_vec[0] = 200;
        do bh.iter() {
            vec.starts_with(match_vec);
        }
    }

    #[bench]
    fn contains_last_element(bh: &mut BenchHarness) {
        let vec: ~[uint] = vec::from_fn(100, |i| i);
        do bh.iter() {
                vec.contains(&99u);
        }
    }
}
