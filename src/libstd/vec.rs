// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Vectors

#[warn(non_camel_case_types)];

use cast::transmute;
use cast;
use container::{Container, Mutable};
use cmp;
use cmp::{Eq, TotalEq, TotalOrd, Ordering, Less, Equal, Greater};
use clone::Clone;
use iterator::{FromIterator, Iterator, IteratorUtil};
use kinds::Copy;
use libc;
use libc::c_void;
use num::Zero;
use option::{None, Option, Some};
use ptr::to_unsafe_ptr;
use ptr;
use ptr::RawPtr;
use rt::global_heap::realloc_raw;
use sys;
use sys::size_of;
use uint;
use unstable::intrinsics;
#[cfg(stage0)]
use intrinsic::{get_tydesc, TyDesc};
#[cfg(not(stage0))]
use unstable::intrinsics::{get_tydesc, contains_managed, TyDesc};
use vec;
use util;

extern {
    #[fast_ffi]
    unsafe fn vec_reserve_shared_actual(t: *TyDesc, v: **raw::VecRepr, n: libc::size_t);
}

/// Returns true if two vectors have the same length
pub fn same_length<T, U>(xs: &[T], ys: &[U]) -> bool {
    xs.len() == ys.len()
}

/**
 * Creates and initializes an owned vector.
 *
 * Creates an owned vector of size `n_elts` and initializes the elements
 * to the value returned by the function `op`.
 */
pub fn from_fn<T>(n_elts: uint, op: &fn(uint) -> T) -> ~[T] {
    unsafe {
        let mut v = with_capacity(n_elts);
        do v.as_mut_buf |p, _len| {
            let mut i: uint = 0u;
            while i < n_elts {
                intrinsics::move_val_init(&mut(*ptr::mut_offset(p, i)), op(i));
                i += 1u;
            }
        }
        raw::set_len(&mut v, n_elts);
        v
    }
}

/**
 * Creates and initializes an owned vector.
 *
 * Creates an owned vector of size `n_elts` and initializes the elements
 * to the value `t`.
 */
pub fn from_elem<T:Copy>(n_elts: uint, t: T) -> ~[T] {
    // FIXME (#7136): manually inline from_fn for 2x plus speedup (sadly very
    // important, from_elem is a bottleneck in borrowck!). Unfortunately it
    // still is substantially slower than using the unsafe
    // vec::with_capacity/ptr::set_memory for primitive types.
    unsafe {
        let mut v = with_capacity(n_elts);
        do v.as_mut_buf |p, _len| {
            let mut i = 0u;
            while i < n_elts {
                intrinsics::move_val_init(&mut(*ptr::mut_offset(p, i)), copy t);
                i += 1u;
            }
        }
        raw::set_len(&mut v, n_elts);
        v
    }
}

/// Creates a new unique vector with the same contents as the slice
pub fn to_owned<T:Copy>(t: &[T]) -> ~[T] {
    from_fn(t.len(), |i| copy t[i])
}

/// Creates a new vector with a capacity of `capacity`
pub fn with_capacity<T>(capacity: uint) -> ~[T] {
    let mut vec = ~[];
    vec.reserve(capacity);
    vec
}

/**
 * Builds a vector by calling a provided function with an argument
 * function that pushes an element to the back of a vector.
 * This version takes an initial capacity for the vector.
 *
 * # Arguments
 *
 * * size - An initial size of the vector to reserve
 * * builder - A function that will construct the vector. It receives
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline]
pub fn build_sized<A>(size: uint, builder: &fn(push: &fn(v: A))) -> ~[A] {
    let mut vec = with_capacity(size);
    builder(|x| vec.push(x));
    vec
}

/**
 * Builds a vector by calling a provided function with an argument
 * function that pushes an element to the back of a vector.
 *
 * # Arguments
 *
 * * builder - A function that will construct the vector. It receives
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline]
pub fn build<A>(builder: &fn(push: &fn(v: A))) -> ~[A] {
    build_sized(4, builder)
}

/**
 * Builds a vector by calling a provided function with an argument
 * function that pushes an element to the back of a vector.
 * This version takes an initial size for the vector.
 *
 * # Arguments
 *
 * * size - An option, maybe containing initial size of the vector to reserve
 * * builder - A function that will construct the vector. It receives
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline]
pub fn build_sized_opt<A>(size: Option<uint>,
                          builder: &fn(push: &fn(v: A)))
                       -> ~[A] {
    build_sized(size.get_or_default(4), builder)
}

/// An iterator over the slices of a vector separated by elements that
/// match a predicate function.
pub struct VecSplitIterator<'self, T> {
    priv v: &'self [T],
    priv n: uint,
    priv pred: &'self fn(t: &T) -> bool,
    priv finished: bool
}

impl<'self, T> Iterator<&'self [T]> for VecSplitIterator<'self, T> {
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
}

/// An iterator over the slices of a vector separated by elements that
/// match a predicate function, from back to front.
pub struct VecRSplitIterator<'self, T> {
    priv v: &'self [T],
    priv n: uint,
    priv pred: &'self fn(t: &T) -> bool,
    priv finished: bool
}

impl<'self, T> Iterator<&'self [T]> for VecRSplitIterator<'self, T> {
    fn next(&mut self) -> Option<&'self [T]> {
        if self.finished { return None; }

        if self.n == 0 {
            self.finished = true;
            return Some(self.v);
        }

        match self.v.rposition(|x| (self.pred)(x)) {
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
}

// Appending

/// Iterates over the `rhs` vector, copying each element and appending it to the
/// `lhs`. Afterwards, the `lhs` is then returned for use again.
#[inline]
pub fn append<T:Copy>(lhs: ~[T], rhs: &[T]) -> ~[T] {
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
pub fn flat_map<T, U>(v: &[T], f: &fn(t: &T) -> ~[U]) -> ~[U] {
    let mut result = ~[];
    for v.iter().advance |elem| { result.push_all_move(f(elem)); }
    result
}

/// Flattens a vector of vectors of T into a single vector of T.
pub fn concat<T:Copy>(v: &[~[T]]) -> ~[T] { v.concat_vec() }

/// Concatenate a vector of vectors, placing a given separator between each
pub fn connect<T:Copy>(v: &[~[T]], sep: &T) -> ~[T] { v.connect_vec(sep) }

/// Flattens a vector of vectors of T into a single vector of T.
pub fn concat_slices<T:Copy>(v: &[&[T]]) -> ~[T] { v.concat_vec() }

/// Concatenate a vector of vectors, placing a given separator between each
pub fn connect_slices<T:Copy>(v: &[&[T]], sep: &T) -> ~[T] { v.connect_vec(sep) }

#[allow(missing_doc)]
pub trait VectorVector<T> {
    // FIXME #5898: calling these .concat and .connect conflicts with
    // StrVector::con{cat,nect}, since they have generic contents.
    pub fn concat_vec(&self) -> ~[T];
    pub fn connect_vec(&self, sep: &T) -> ~[T];
}

impl<'self, T:Copy> VectorVector<T> for &'self [~[T]] {
    /// Flattens a vector of slices of T into a single vector of T.
    pub fn concat_vec(&self) -> ~[T] {
        self.flat_map(|inner| copy *inner)
    }

    /// Concatenate a vector of vectors, placing a given separator between each.
    pub fn connect_vec(&self, sep: &T) -> ~[T] {
        let mut r = ~[];
        let mut first = true;
        for self.iter().advance |inner| {
            if first { first = false; } else { r.push(copy *sep); }
            r.push_all(copy *inner);
        }
        r
    }
}

impl<'self, T:Copy> VectorVector<T> for &'self [&'self [T]] {
    /// Flattens a vector of slices of T into a single vector of T.
    pub fn concat_vec(&self) -> ~[T] {
        self.flat_map(|&inner| inner.to_owned())
    }

    /// Concatenate a vector of slices, placing a given separator between each.
    pub fn connect_vec(&self, sep: &T) -> ~[T] {
        let mut r = ~[];
        let mut first = true;
        for self.iter().advance |&inner| {
            if first { first = false; } else { r.push(copy *sep); }
            r.push_all(inner);
        }
        r
    }
}

// FIXME: if issue #586 gets implemented, could have a postcondition
// saying the two result lists have the same length -- or, could
// return a nominal record with a constraint saying that, instead of
// returning a tuple (contingent on issue #869)
/**
 * Convert a vector of pairs into a pair of vectors, by reference. As unzip().
 */
pub fn unzip_slice<T:Copy,U:Copy>(v: &[(T, U)]) -> (~[T], ~[U]) {
    let mut ts = ~[];
    let mut us = ~[];
    for v.iter().advance |p| {
        let (t, u) = copy *p;
        ts.push(t);
        us.push(u);
    }
    (ts, us)
}

/**
 * Convert a vector of pairs into a pair of vectors.
 *
 * Returns a tuple containing two vectors where the i-th element of the first
 * vector contains the first element of the i-th tuple of the input vector,
 * and the i-th element of the second vector contains the second element
 * of the i-th tuple of the input vector.
 */
pub fn unzip<T,U>(v: ~[(T, U)]) -> (~[T], ~[U]) {
    let mut ts = ~[];
    let mut us = ~[];
    for v.consume_iter().advance |p| {
        let (t, u) = p;
        ts.push(t);
        us.push(u);
    }
    (ts, us)
}

/**
 * Convert two vectors to a vector of pairs, by reference. As zip().
 */
pub fn zip_slice<T:Copy,U:Copy>(v: &[T], u: &[U])
        -> ~[(T, U)] {
    let mut zipped = ~[];
    let sz = v.len();
    let mut i = 0u;
    assert_eq!(sz, u.len());
    while i < sz {
        zipped.push((copy v[i], copy u[i]));
        i += 1u;
    }
    zipped
}

/**
 * Convert two vectors to a vector of pairs.
 *
 * Returns a vector of tuples, where the i-th tuple contains the
 * i-th elements from each of the input vectors.
 */
pub fn zip<T, U>(mut v: ~[T], mut u: ~[U]) -> ~[(T, U)] {
    let mut i = v.len();
    assert_eq!(i, u.len());
    let mut w = with_capacity(i);
    while i > 0 {
        w.push((v.pop(),u.pop()));
        i -= 1;
    }
    w.reverse();
    w
}

/**
 * Iterate over all permutations of vector `v`.
 *
 * Permutations are produced in lexicographic order with respect to the order
 * of elements in `v` (so if `v` is sorted then the permutations are
 * lexicographically sorted).
 *
 * The total number of permutations produced is `v.len()!`.  If `v` contains
 * repeated elements, then some permutations are repeated.
 *
 * See [Algorithms to generate
 * permutations](http://en.wikipedia.org/wiki/Permutation).
 *
 *  # Arguments
 *
 *  * `values` - A vector of values from which the permutations are
 *  chosen
 *
 *  * `fun` - The function to iterate over the combinations
 */
pub fn each_permutation<T:Copy>(values: &[T], fun: &fn(perm : &[T]) -> bool) -> bool {
    let length = values.len();
    let mut permutation = vec::from_fn(length, |i| copy values[i]);
    if length <= 1 {
        fun(permutation);
        return true;
    }
    let mut indices = vec::from_fn(length, |i| i);
    loop {
        if !fun(permutation) { return true; }
        // find largest k such that indices[k] < indices[k+1]
        // if no such k exists, all permutations have been generated
        let mut k = length - 2;
        while k > 0 && indices[k] >= indices[k+1] {
            k -= 1;
        }
        if k == 0 && indices[0] > indices[1] { return true; }
        // find largest l such that indices[k] < indices[l]
        // k+1 is guaranteed to be such
        let mut l = length - 1;
        while indices[k] >= indices[l] {
            l -= 1;
        }
        // swap indices[k] and indices[l]; sort indices[k+1..]
        // (they're just reversed)
        indices.swap(k, l);
        indices.mut_slice(k+1, length).reverse();
        // fixup permutation based on indices
        for uint::range(k, length) |i| {
            permutation[i] = copy values[indices[i]];
        }
    }
}

/// An iterator over the (overlapping) slices of length `size` within
/// a vector.
pub struct VecWindowIter<'self, T> {
    priv v: &'self [T],
    priv size: uint
}

impl<'self, T> Iterator<&'self [T]> for VecWindowIter<'self, T> {
    fn next(&mut self) -> Option<&'self [T]> {
        if self.size > self.v.len() {
            None
        } else {
            let ret = Some(self.v.slice(0, self.size));
            self.v = self.v.slice(1, self.v.len());
            ret
        }
    }
}

/// An iterator over a vector in (non-overlapping) chunks (`size`
/// elements at a time).
pub struct VecChunkIter<'self, T> {
    priv v: &'self [T],
    priv size: uint
}

impl<'self, T> Iterator<&'self [T]> for VecChunkIter<'self, T> {
    fn next(&mut self) -> Option<&'self [T]> {
        if self.size == 0 {
            None
        } else if self.size >= self.v.len() {
            // finished
            self.size = 0;
            Some(self.v)
        } else {
            let ret = Some(self.v.slice(0, self.size));
            self.v = self.v.slice(self.size, self.v.len());
            ret
        }
    }
}

// Equality

#[cfg(not(test))]
pub mod traits {
    use super::Vector;
    use kinds::Copy;
    use cmp::{Eq, Ord, TotalEq, TotalOrd, Ordering, Equal, Equiv};
    use ops::Add;

    impl<'self,T:Eq> Eq for &'self [T] {
        fn eq(&self, other: & &'self [T]) -> bool {
            self.len() == other.len() &&
                self.iter().zip(other.iter()).all(|(s,o)| *s == *o)
        }
        #[inline]
        fn ne(&self, other: & &'self [T]) -> bool { !self.eq(other) }
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
                self.iter().zip(other.iter()).all(|(s,o)| s.equals(o))
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
            for self.iter().zip(other.iter()).advance |(s,o)| {
                match s.cmp(o) {
                    Equal => {},
                    non_eq => { return non_eq; }
                }
            }
            self.len().cmp(&other.len())
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

    impl<'self,T:Ord> Ord for &'self [T] {
        fn lt(&self, other: & &'self [T]) -> bool {
            for self.iter().zip(other.iter()).advance |(s,o)| {
                if *s < *o { return true; }
                if *s > *o { return false; }
            }
            self.len() < other.len()
        }
        #[inline]
        fn le(&self, other: & &'self [T]) -> bool { !(*other < *self) }
        #[inline]
        fn ge(&self, other: & &'self [T]) -> bool { !(*self < *other) }
        #[inline]
        fn gt(&self, other: & &'self [T]) -> bool { *other < *self }
    }

    impl<T:Ord> Ord for ~[T] {
        #[inline]
        fn lt(&self, other: &~[T]) -> bool { self.as_slice() < other.as_slice() }
        #[inline]
        fn le(&self, other: &~[T]) -> bool { self.as_slice() <= other.as_slice() }
        #[inline]
        fn ge(&self, other: &~[T]) -> bool { self.as_slice() >= other.as_slice() }
        #[inline]
        fn gt(&self, other: &~[T]) -> bool { self.as_slice() > other.as_slice() }
    }

    impl<T:Ord> Ord for @[T] {
        #[inline]
        fn lt(&self, other: &@[T]) -> bool { self.as_slice() < other.as_slice() }
        #[inline]
        fn le(&self, other: &@[T]) -> bool { self.as_slice() <= other.as_slice() }
        #[inline]
        fn ge(&self, other: &@[T]) -> bool { self.as_slice() >= other.as_slice() }
        #[inline]
        fn gt(&self, other: &@[T]) -> bool { self.as_slice() > other.as_slice() }
    }

    impl<'self,T:Copy, V: Vector<T>> Add<V, ~[T]> for &'self [T] {
        #[inline]
        fn add(&self, rhs: &V) -> ~[T] {
            let mut res = self.to_owned();
            res.push_all(rhs.as_slice());
            res
        }
    }
    impl<T:Copy, V: Vector<T>> Add<V, ~[T]> for ~[T] {
        #[inline]
        fn add(&self, rhs: &V) -> ~[T] {
            let mut res = self.to_owned();
            res.push_all(rhs.as_slice());
            res
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
    /// Returns true if a vector contains no elements
    #[inline]
    fn is_empty(&self) -> bool {
        self.as_imm_buf(|_p, len| len == 0u)
    }

    /// Returns the length of a vector
    #[inline]
    fn len(&self) -> uint {
        self.as_imm_buf(|_p, len| len)
    }
}

impl<T> Container for ~[T] {
    /// Returns true if a vector contains no elements
    #[inline]
    fn is_empty(&self) -> bool {
        self.as_imm_buf(|_p, len| len == 0u)
    }

    /// Returns the length of a vector
    #[inline]
    fn len(&self) -> uint {
        self.as_imm_buf(|_p, len| len)
    }
}

#[allow(missing_doc)]
pub trait CopyableVector<T> {
    fn to_owned(&self) -> ~[T];
}

/// Extension methods for vectors
impl<'self,T:Copy> CopyableVector<T> for &'self [T] {
    /// Returns a copy of `v`.
    #[inline]
    fn to_owned(&self) -> ~[T] {
        let mut result = with_capacity(self.len());
        for self.iter().advance |e| {
            result.push(copy *e);
        }
        result
    }
}

#[allow(missing_doc)]
pub trait ImmutableVector<'self, T> {
    fn slice(&self, start: uint, end: uint) -> &'self [T];
    fn iter(self) -> VecIterator<'self, T>;
    fn rev_iter(self) -> VecRevIterator<'self, T>;
    fn split_iter(self, pred: &'self fn(&T) -> bool) -> VecSplitIterator<'self, T>;
    fn splitn_iter(self, n: uint, pred: &'self fn(&T) -> bool) -> VecSplitIterator<'self, T>;
    fn rsplit_iter(self, pred: &'self fn(&T) -> bool) -> VecRSplitIterator<'self, T>;
    fn rsplitn_iter(self,  n: uint, pred: &'self fn(&T) -> bool) -> VecRSplitIterator<'self, T>;

    fn window_iter(self, size: uint) -> VecWindowIter<'self, T>;
    fn chunk_iter(self, size: uint) -> VecChunkIter<'self, T>;

    fn head(&self) -> &'self T;
    fn head_opt(&self) -> Option<&'self T>;
    fn tail(&self) -> &'self [T];
    fn tailn(&self, n: uint) -> &'self [T];
    fn init(&self) -> &'self [T];
    fn initn(&self, n: uint) -> &'self [T];
    fn last(&self) -> &'self T;
    fn last_opt(&self) -> Option<&'self T>;
    fn rposition(&self, f: &fn(t: &T) -> bool) -> Option<uint>;
    fn flat_map<U>(&self, f: &fn(t: &T) -> ~[U]) -> ~[U];
    unsafe fn unsafe_ref(&self, index: uint) -> *T;

    fn bsearch(&self, f: &fn(&T) -> Ordering) -> Option<uint>;

    fn map<U>(&self, &fn(t: &T) -> U) -> ~[U];

    fn as_imm_buf<U>(&self, f: &fn(*T, uint) -> U) -> U;
}

/// Extension methods for vectors
impl<'self,T> ImmutableVector<'self, T> for &'self [T] {
    /// Return a slice that points into another slice.
    #[inline]
    fn slice(&self, start: uint, end: uint) -> &'self [T] {
    assert!(start <= end);
    assert!(end <= self.len());
        do self.as_imm_buf |p, _len| {
            unsafe {
                transmute((ptr::offset(p, start),
                           (end - start) * sys::nonzero_size_of::<T>()))
            }
        }
    }

    #[inline]
    fn iter(self) -> VecIterator<'self, T> {
        unsafe {
            let p = vec::raw::to_ptr(self);
            VecIterator{ptr: p, end: p.offset(self.len()),
                        lifetime: cast::transmute(p)}
        }
    }
    #[inline]
    fn rev_iter(self) -> VecRevIterator<'self, T> {
        unsafe {
            let p = vec::raw::to_ptr(self);
            VecRevIterator{ptr: p.offset(self.len() - 1),
                           end: p.offset(-1),
                           lifetime: cast::transmute(p)}
        }
    }

    /// Returns an iterator over the subslices of the vector which are
    /// separated by elements that match `pred`.
    #[inline]
    fn split_iter(self, pred: &'self fn(&T) -> bool) -> VecSplitIterator<'self, T> {
        self.splitn_iter(uint::max_value, pred)
    }
    /// Returns an iterator over the subslices of the vector which are
    /// separated by elements that match `pred`, limited to splitting
    /// at most `n` times.
    #[inline]
    fn splitn_iter(self, n: uint, pred: &'self fn(&T) -> bool) -> VecSplitIterator<'self, T> {
        VecSplitIterator {
            v: self,
            n: n,
            pred: pred,
            finished: false
        }
    }
    /// Returns an iterator over the subslices of the vector which are
    /// separated by elements that match `pred`. This starts at the
    /// end of the vector and works backwards.
    #[inline]
    fn rsplit_iter(self, pred: &'self fn(&T) -> bool) -> VecRSplitIterator<'self, T> {
        self.rsplitn_iter(uint::max_value, pred)
    }
    /// Returns an iterator over the subslices of the vector which are
    /// separated by elements that match `pred` limited to splitting
    /// at most `n` times. This starts at the end of the vector and
    /// works backwards.
    #[inline]
    fn rsplitn_iter(self, n: uint, pred: &'self fn(&T) -> bool) -> VecRSplitIterator<'self, T> {
        VecRSplitIterator {
            v: self,
            n: n,
            pred: pred,
            finished: false
        }
    }

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
     * ~~~ {.rust}
     * let v = &[1,2,3,4];
     * for v.window_iter().advance |win| {
     *     io::println(fmt!("%?", win));
     * }
     * ~~~
     *
     */
    fn window_iter(self, size: uint) -> VecWindowIter<'self, T> {
        assert!(size != 0);
        VecWindowIter { v: self, size: size }
    }

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
     * ~~~ {.rust}
     * let v = &[1,2,3,4,5];
     * for v.chunk_iter().advance |win| {
     *     io::println(fmt!("%?", win));
     * }
     * ~~~
     *
     */
    fn chunk_iter(self, size: uint) -> VecChunkIter<'self, T> {
        assert!(size != 0);
        VecChunkIter { v: self, size: size }
    }

    /// Returns the first element of a vector, failing if the vector is empty.
    #[inline]
    fn head(&self) -> &'self T {
        if self.len() == 0 { fail!("head: empty vector") }
        &self[0]
    }

    /// Returns the first element of a vector, or `None` if it is empty
    #[inline]
    fn head_opt(&self) -> Option<&'self T> {
        if self.len() == 0 { None } else { Some(&self[0]) }
    }

    /// Returns all but the first element of a vector
    #[inline]
    fn tail(&self) -> &'self [T] { self.slice(1, self.len()) }

    /// Returns all but the first `n' elements of a vector
    #[inline]
    fn tailn(&self, n: uint) -> &'self [T] { self.slice(n, self.len()) }

    /// Returns all but the last element of a vector
    #[inline]
    fn init(&self) -> &'self [T] {
        self.slice(0, self.len() - 1)
    }

    /// Returns all but the last `n' elemnts of a vector
    #[inline]
    fn initn(&self, n: uint) -> &'self [T] {
        self.slice(0, self.len() - n)
    }

    /// Returns the last element of a vector, failing if the vector is empty.
    #[inline]
    fn last(&self) -> &'self T {
        if self.len() == 0 { fail!("last: empty vector") }
        &self[self.len() - 1]
    }

    /// Returns the last element of a vector, or `None` if it is empty.
    #[inline]
    fn last_opt(&self) -> Option<&'self T> {
            if self.len() == 0 { None } else { Some(&self[self.len() - 1]) }
    }

    /**
     * Find the last index matching some predicate
     *
     * Apply function `f` to each element of `v` in reverse order.  When
     * function `f` returns true then an option containing the index is
     * returned. If `f` matches no elements then None is returned.
     */
    #[inline]
    fn rposition(&self, f: &fn(t: &T) -> bool) -> Option<uint> {
        for self.rev_iter().enumerate().advance |(i, t)| {
            if f(t) { return Some(self.len() - i - 1); }
        }
        None
    }

    /**
     * Apply a function to each element of a vector and return a concatenation
     * of each result vector
     */
    #[inline]
    fn flat_map<U>(&self, f: &fn(t: &T) -> ~[U]) -> ~[U] {
        flat_map(*self, f)
    }
    /// Returns a pointer to the element at the given index, without doing
    /// bounds checking.
    #[inline]
    unsafe fn unsafe_ref(&self, index: uint) -> *T {
        let (ptr, _): (*T, uint) = transmute(*self);
        ptr.offset(index)
    }

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
    fn bsearch(&self, f: &fn(&T) -> Ordering) -> Option<uint> {
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

    /// Deprecated, use iterators where possible
    /// (`self.iter().transform(f)`). Apply a function to each element
    /// of a vector and return the results.
    fn map<U>(&self, f: &fn(t: &T) -> U) -> ~[U] {
        self.iter().transform(f).collect()
    }

    /**
     * Work with the buffer of a vector.
     *
     * Allows for unsafe manipulation of vector contents, which is useful for
     * foreign interop.
     */
    #[inline]
    fn as_imm_buf<U>(&self,
                     /* NB---this CANNOT be const, see below */
                     f: &fn(*T, uint) -> U) -> U {
        // NB---Do not change the type of s to `&const [T]`.  This is
        // unsound.  The reason is that we are going to create immutable pointers
        // into `s` and pass them to `f()`, but in fact they are potentially
        // pointing at *mutable memory*.  Use `as_mut_buf` instead!

        unsafe {
            let v : *(*T,uint) = transmute(self);
            let (buf,len) = *v;
            f(buf, len / sys::nonzero_size_of::<T>())
        }
    }
}

#[allow(missing_doc)]
pub trait ImmutableEqVector<T:Eq> {
    fn position_elem(&self, t: &T) -> Option<uint>;
    fn rposition_elem(&self, t: &T) -> Option<uint>;
    fn contains(&self, x: &T) -> bool;
}

impl<'self,T:Eq> ImmutableEqVector<T> for &'self [T] {
    /// Find the first index containing a matching value
    #[inline]
    fn position_elem(&self, x: &T) -> Option<uint> {
        self.iter().position(|y| *x == *y)
    }

    /// Find the last index containing a matching value
    #[inline]
    fn rposition_elem(&self, t: &T) -> Option<uint> {
        self.rposition(|x| *x == *t)
    }

    /// Return true if a vector contains an element with the given value
    fn contains(&self, x: &T) -> bool {
        for self.iter().advance |elt| { if *x == *elt { return true; } }
        false
    }
}

#[allow(missing_doc)]
pub trait ImmutableTotalOrdVector<T: TotalOrd> {
    fn bsearch_elem(&self, x: &T) -> Option<uint>;
}

impl<'self, T: TotalOrd> ImmutableTotalOrdVector<T> for &'self [T] {
    /**
     * Binary search a sorted vector for a given element.
     *
     * Returns the index of the element or None if not found.
     */
    fn bsearch_elem(&self, x: &T) -> Option<uint> {
        self.bsearch(|p| p.cmp(x))
    }
}

#[allow(missing_doc)]
pub trait ImmutableCopyableVector<T> {
    fn partitioned(&self, f: &fn(&T) -> bool) -> (~[T], ~[T]);
    unsafe fn unsafe_get(&self, elem: uint) -> T;
}

/// Extension methods for vectors
impl<'self,T:Copy> ImmutableCopyableVector<T> for &'self [T] {
    /**
     * Partitions the vector into those that satisfies the predicate, and
     * those that do not.
     */
    #[inline]
    fn partitioned(&self, f: &fn(&T) -> bool) -> (~[T], ~[T]) {
        let mut lefts  = ~[];
        let mut rights = ~[];

        for self.iter().advance |elt| {
            if f(elt) {
                lefts.push(copy *elt);
            } else {
                rights.push(copy *elt);
            }
        }

        (lefts, rights)
    }

    /// Returns the element at the given index, without doing bounds checking.
    #[inline]
    unsafe fn unsafe_get(&self, index: uint) -> T {
        copy *self.unsafe_ref(index)
    }
}

#[allow(missing_doc)]
pub trait OwnedVector<T> {
    fn consume_iter(self) -> VecConsumeIterator<T>;
    fn consume_rev_iter(self) -> VecConsumeRevIterator<T>;

    fn reserve(&mut self, n: uint);
    fn reserve_at_least(&mut self, n: uint);
    fn capacity(&self) -> uint;

    fn push(&mut self, t: T);
    unsafe fn push_fast(&mut self, t: T);

    fn push_all_move(&mut self, rhs: ~[T]);
    fn pop(&mut self) -> T;
    fn pop_opt(&mut self) -> Option<T>;
    fn shift(&mut self) -> T;
    fn shift_opt(&mut self) -> Option<T>;
    fn unshift(&mut self, x: T);
    fn insert(&mut self, i: uint, x:T);
    fn remove(&mut self, i: uint) -> T;
    fn swap_remove(&mut self, index: uint) -> T;
    fn truncate(&mut self, newlen: uint);
    fn retain(&mut self, f: &fn(t: &T) -> bool);
    fn partition(self, f: &fn(&T) -> bool) -> (~[T], ~[T]);
    fn grow_fn(&mut self, n: uint, op: &fn(uint) -> T);
}

impl<T> OwnedVector<T> for ~[T] {
    /// Creates a consuming iterator, that is, one that moves each
    /// value out of the vector (from start to end). The vector cannot
    /// be used after calling this.
    ///
    /// Note that this performs O(n) swaps, and so `consume_rev_iter`
    /// (which just calls `pop` repeatedly) is more efficient.
    ///
    /// # Examples
    ///
    /// ~~~ {.rust}
    /// let v = ~[~"a", ~"b"];
    /// for v.consume_iter().advance |s| {
    ///   // s has type ~str, not &~str
    ///   println(s);
    /// }
    /// ~~~
    fn consume_iter(self) -> VecConsumeIterator<T> {
        VecConsumeIterator { v: self, idx: 0 }
    }
    /// Creates a consuming iterator that moves out of the vector in
    /// reverse order. Also see `consume_iter`, however note that this
    /// is more efficient.
    fn consume_rev_iter(self) -> VecConsumeRevIterator<T> {
        VecConsumeRevIterator { v: self }
    }

    /**
     * Reserves capacity for exactly `n` elements in the given vector.
     *
     * If the capacity for `self` is already equal to or greater than the requested
     * capacity, then no action is taken.
     *
     * # Arguments
     *
     * * n - The number of elements to reserve space for
     */
    #[inline]
    #[cfg(stage0)]
    fn reserve(&mut self, n: uint) {
        // Only make the (slow) call into the runtime if we have to
        use managed;
        if self.capacity() < n {
            unsafe {
                let ptr: *mut *mut raw::VecRepr = cast::transmute(self);
                let td = get_tydesc::<T>();
                if ((**ptr).box_header.ref_count ==
                    managed::raw::RC_MANAGED_UNIQUE) {
                    vec_reserve_shared_actual(td, ptr as **raw::VecRepr, n as libc::size_t);
                } else {
                    let alloc = n * sys::nonzero_size_of::<T>();
                    *ptr = realloc_raw(*ptr as *mut c_void, alloc + size_of::<raw::VecRepr>())
                           as *mut raw::VecRepr;
                    (**ptr).unboxed.alloc = alloc;
                }
            }
        }
    }

    /**
     * Reserves capacity for exactly `n` elements in the given vector.
     *
     * If the capacity for `self` is already equal to or greater than the requested
     * capacity, then no action is taken.
     *
     * # Arguments
     *
     * * n - The number of elements to reserve space for
     */
    #[inline]
    #[cfg(not(stage0))]
    fn reserve(&mut self, n: uint) {
        // Only make the (slow) call into the runtime if we have to
        if self.capacity() < n {
            unsafe {
                let ptr: *mut *mut raw::VecRepr = cast::transmute(self);
                let td = get_tydesc::<T>();
                if contains_managed::<T>() {
                    vec_reserve_shared_actual(td, ptr as **raw::VecRepr, n as libc::size_t);
                } else {
                    let alloc = n * sys::nonzero_size_of::<T>();
                    *ptr = realloc_raw(*ptr as *mut c_void, alloc + size_of::<raw::VecRepr>())
                           as *mut raw::VecRepr;
                    (**ptr).unboxed.alloc = alloc;
                }
            }
        }
    }

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
    fn reserve_at_least(&mut self, n: uint) {
        self.reserve(uint::next_power_of_two(n));
    }

    /// Returns the number of elements the vector can hold without reallocating.
    #[inline]
    fn capacity(&self) -> uint {
        unsafe {
            let repr: **raw::VecRepr = transmute(self);
            (**repr).unboxed.alloc / sys::nonzero_size_of::<T>()
        }
    }

    /// Append an element to a vector
    #[inline]
    fn push(&mut self, t: T) {
        unsafe {
            let repr: **raw::VecRepr = transmute(&mut *self);
            let fill = (**repr).unboxed.fill;
            if (**repr).unboxed.alloc <= fill {
                // need more space
                reserve_no_inline(self);
            }

            self.push_fast(t);
        }

        // this peculiar function is because reserve_at_least is very
        // large (because of reserve), and will be inlined, which
        // makes push too large.
        #[inline(never)]
        fn reserve_no_inline<T>(v: &mut ~[T]) {
            let new_len = v.len() + 1;
            v.reserve_at_least(new_len);
        }
    }

    // This doesn't bother to make sure we have space.
    #[inline] // really pretty please
    unsafe fn push_fast(&mut self, t: T) {
        let repr: **mut raw::VecRepr = transmute(self);
        let fill = (**repr).unboxed.fill;
        (**repr).unboxed.fill += sys::nonzero_size_of::<T>();
        let p = to_unsafe_ptr(&((**repr).unboxed.data));
        let p = ptr::offset(p, fill) as *mut T;
        intrinsics::move_val_init(&mut(*p), t);
    }

    /// Takes ownership of the vector `rhs`, moving all elements into
    /// the current vector. This does not copy any elements, and it is
    /// illegal to use the `rhs` vector after calling this method
    /// (because it is moved here).
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let mut a = ~[~1];
    /// a.push_all_move(~[~2, ~3, ~4]);
    /// assert!(a == ~[~1, ~2, ~3, ~4]);
    /// ~~~
    #[inline]
    fn push_all_move(&mut self, mut rhs: ~[T]) {
        let self_len = self.len();
        let rhs_len = rhs.len();
        let new_len = self_len + rhs_len;
        self.reserve(new_len);
        unsafe { // Note: infallible.
            let self_p = vec::raw::to_mut_ptr(*self);
            let rhs_p = vec::raw::to_ptr(rhs);
            ptr::copy_memory(ptr::mut_offset(self_p, self_len), rhs_p, rhs_len);
            raw::set_len(self, new_len);
            raw::set_len(&mut rhs, 0);
        }
    }

    /// Remove the last element from a vector and return it, or `None` if it is empty
    fn pop_opt(&mut self) -> Option<T> {
        match self.len() {
            0  => None,
            ln => {
                let valptr = ptr::to_mut_unsafe_ptr(&mut self[ln - 1u]);
                unsafe {
                    raw::set_len(self, ln - 1u);
                    Some(ptr::read_ptr(valptr))
                }
            }
        }
    }


    /// Remove the last element from a vector and return it, failing if it is empty
    #[inline]
    fn pop(&mut self) -> T {
        self.pop_opt().expect("pop: empty vector")
    }

    /// Removes the first element from a vector and return it
    #[inline]
    fn shift(&mut self) -> T {
        self.shift_opt().expect("shift: empty vector")
    }

    /// Removes the first element from a vector and return it, or `None` if it is empty
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
                raw::copy_memory(transmute(last_slice), first_slice, 1);
            }

            // Memcopy everything to the left one element
            {
                let init_slice = self.slice(0, next_ln);
                let tail_slice = self.slice(1, ln);
                raw::copy_memory(transmute(init_slice),
                                 tail_slice,
                                 next_ln);
            }

            // Set the new length. Now the vector is back to normal
            raw::set_len(self, next_ln);

            // Swap out the element we want from the end
            let vp = raw::to_mut_ptr(*self);
            let vp = ptr::mut_offset(vp, next_ln - 1);

            Some(ptr::replace_ptr(vp, work_elt))
        }
    }

    /// Prepend an element to the vector
    fn unshift(&mut self, x: T) {
        let v = util::replace(self, ~[x]);
        self.push_all_move(v);
    }

    /// Insert an element at position i within v, shifting all
    /// elements after position i one position to the right.
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

    /// Remove and return the element at position i within v, shifting
    /// all elements after position i one position to the left.
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

    /**
     * Remove an element from anywhere in the vector and return it, replacing it
     * with the last element. This does not preserve ordering, but is O(1).
     *
     * Fails if index >= length.
     */
    fn swap_remove(&mut self, index: uint) -> T {
        let ln = self.len();
        if index >= ln {
            fail!("vec::swap_remove - index %u >= length %u", index, ln);
        }
        if index < ln - 1 {
            self.swap(index, ln - 1);
        }
        self.pop()
    }

    /// Shorten a vector, dropping excess elements.
    fn truncate(&mut self, newlen: uint) {
        do self.as_mut_buf |p, oldlen| {
            assert!(newlen <= oldlen);
            unsafe {
                // This loop is optimized out for non-drop types.
                for uint::range(newlen, oldlen) |i| {
                    ptr::read_and_zero_ptr(ptr::mut_offset(p, i));
                }
            }
        }
        unsafe { raw::set_len(self, newlen); }
    }


    /**
     * Like `filter()`, but in place.  Preserves order of `v`.  Linear time.
     */
    fn retain(&mut self, f: &fn(t: &T) -> bool) {
        let len = self.len();
        let mut deleted: uint = 0;

        for uint::range(0, len) |i| {
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

    /**
     * Partitions the vector into those that satisfies the predicate, and
     * those that do not.
     */
    #[inline]
    fn partition(self, f: &fn(&T) -> bool) -> (~[T], ~[T]) {
        let mut lefts  = ~[];
        let mut rights = ~[];

        for self.consume_iter().advance |elt| {
            if f(&elt) {
                lefts.push(elt);
            } else {
                rights.push(elt);
            }
        }

        (lefts, rights)
    }

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
    fn grow_fn(&mut self, n: uint, op: &fn(uint) -> T) {
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

#[allow(missing_doc)]
pub trait OwnedCopyableVector<T:Copy> {
    fn push_all(&mut self, rhs: &[T]);
    fn grow(&mut self, n: uint, initval: &T);
    fn grow_set(&mut self, index: uint, initval: &T, val: T);
}

impl<T:Copy> OwnedCopyableVector<T> for ~[T] {
    /// Iterates over the slice `rhs`, copies each element, and then appends it to
    /// the vector provided `v`. The `rhs` vector is traversed in-order.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let mut a = ~[1];
    /// a.push_all([2, 3, 4]);
    /// assert!(a == ~[1, 2, 3, 4]);
    /// ~~~
    #[inline]
    fn push_all(&mut self, rhs: &[T]) {
        let new_len = self.len() + rhs.len();
        self.reserve(new_len);

        for uint::range(0u, rhs.len()) |i| {
            self.push(unsafe { raw::get(rhs, i) })
        }
    }

    /**
     * Expands a vector in place, initializing the new elements to a given value
     *
     * # Arguments
     *
     * * n - The number of elements to add
     * * initval - The value for the new elements
     */
    fn grow(&mut self, n: uint, initval: &T) {
        let new_len = self.len() + n;
        self.reserve_at_least(new_len);
        let mut i: uint = 0u;

        while i < n {
            self.push(copy *initval);
            i += 1u;
        }
    }

    /**
     * Sets the value of a vector element at a given index, growing the vector as
     * needed
     *
     * Sets the element at position `index` to `val`. If `index` is past the end
     * of the vector, expands the vector by replicating `initval` to fill the
     * intervening space.
     */
    fn grow_set(&mut self, index: uint, initval: &T, val: T) {
        let l = self.len();
        if index >= l { self.grow(index - l + 1u, initval); }
        self[index] = val;
    }
}

#[allow(missing_doc)]
pub trait OwnedEqVector<T:Eq> {
    fn dedup(&mut self);
}

impl<T:Eq> OwnedEqVector<T> for ~[T] {
    /**
    * Remove consecutive repeated elements from a vector; if the vector is
    * sorted, this removes all duplicates.
    */
    pub fn dedup(&mut self) {
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
                let p_r = ptr::mut_offset(p, r);
                let p_wm1 = ptr::mut_offset(p, w - 1);
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

#[allow(missing_doc)]
pub trait MutableVector<'self, T> {
    fn mut_slice(self, start: uint, end: uint) -> &'self mut [T];
    fn mut_iter(self) -> VecMutIterator<'self, T>;
    fn mut_rev_iter(self) -> VecMutRevIterator<'self, T>;

    fn swap(self, a: uint, b: uint);

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

    unsafe fn unsafe_mut_ref(&self, index: uint) -> *mut T;
    unsafe fn unsafe_set(&self, index: uint, val: T);

    fn as_mut_buf<U>(&self, f: &fn(*mut T, uint) -> U) -> U;
}

impl<'self,T> MutableVector<'self, T> for &'self mut [T] {
    /// Return a slice that points into another slice.
    #[inline]
    fn mut_slice(self, start: uint, end: uint) -> &'self mut [T] {
        assert!(start <= end);
        assert!(end <= self.len());
        do self.as_mut_buf |p, _len| {
            unsafe {
                transmute((ptr::mut_offset(p, start),
                           (end - start) * sys::nonzero_size_of::<T>()))
            }
        }
    }

    #[inline]
    fn mut_iter(self) -> VecMutIterator<'self, T> {
        unsafe {
            let p = vec::raw::to_mut_ptr(self);
            VecMutIterator{ptr: p, end: p.offset(self.len()),
                           lifetime: cast::transmute(p)}
        }
    }

    fn mut_rev_iter(self) -> VecMutRevIterator<'self, T> {
        unsafe {
            let p = vec::raw::to_mut_ptr(self);
            VecMutRevIterator{ptr: p.offset(self.len() - 1),
                              end: p.offset(-1),
                              lifetime: cast::transmute(p)}
        }
    }

    /**
     * Swaps two elements in a vector
     *
     * # Arguments
     *
     * * a - The index of the first element
     * * b - The index of the second element
     */
    fn swap(self, a: uint, b: uint) {
        unsafe {
            // Can't take two mutable loans from one vector, so instead just cast
            // them to their raw pointers to do the swap
            let pa: *mut T = &mut self[a];
            let pb: *mut T = &mut self[b];
            ptr::swap_ptr(pa, pb);
        }
    }

    /// Reverse the order of elements in a vector, in place
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
        for self.mut_iter().zip(src.mut_slice(start, end).mut_iter()).advance |(a, b)| {
            util::swap(a, b);
        }
        cmp::min(self.len(), end-start)
    }

    #[inline]
    unsafe fn unsafe_mut_ref(&self, index: uint) -> *mut T {
        let pair_ptr: &(*mut T, uint) = transmute(self);
        let (ptr, _) = *pair_ptr;
        ptr.offset(index)
    }

    #[inline]
    unsafe fn unsafe_set(&self, index: uint, val: T) {
        *self.unsafe_mut_ref(index) = val;
    }

    /// Similar to `as_imm_buf` but passing a `*mut T`
    #[inline]
    fn as_mut_buf<U>(&self, f: &fn(*mut T, uint) -> U) -> U {
        unsafe {
            let v : *(*mut T,uint) = transmute(self);
            let (buf,len) = *v;
            f(buf, len / sys::nonzero_size_of::<T>())
        }
    }

}

/// Trait for ~[T] where T is Cloneable
pub trait MutableCloneableVector<T> {
    /// Copies as many elements from `src` as it can into `self`
    /// (the shorter of self.len() and src.len()). Returns the number of elements copied.
    fn copy_from(self, &[T]) -> uint;
}

impl<'self, T:Clone> MutableCloneableVector<T> for &'self mut [T] {
    #[inline]
    fn copy_from(self, src: &[T]) -> uint {
        for self.mut_iter().zip(src.iter()).advance |(a, b)| {
            *a = b.clone();
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

/// The internal 'unboxed' representation of a vector
#[allow(missing_doc)]
pub struct UnboxedVecRepr {
    fill: uint,
    alloc: uint,
    data: u8
}

/// Unsafe operations
pub mod raw {
    use cast::transmute;
    use kinds::Copy;
    use managed;
    use option::{None, Some};
    use ptr;
    use sys;
    use unstable::intrinsics;
    use vec::{UnboxedVecRepr, with_capacity, ImmutableVector, MutableVector};
    use util;

    /// The internal representation of a (boxed) vector
    #[allow(missing_doc)]
    pub struct VecRepr {
        box_header: managed::raw::BoxHeaderRepr,
        unboxed: UnboxedVecRepr
    }

    /// The internal representation of a slice
    pub struct SliceRepr {
        /// Pointer to the base of this slice
        data: *u8,
        /// The length of the slice
        len: uint
    }

    /**
     * Sets the length of a vector
     *
     * This will explicitly set the size of the vector, without actually
     * modifing its buffers, so it is up to the caller to ensure that
     * the vector is actually the specified size.
     */
    #[inline]
    pub unsafe fn set_len<T>(v: &mut ~[T], new_len: uint) {
        let repr: **mut VecRepr = transmute(v);
        (**repr).unboxed.fill = new_len * sys::nonzero_size_of::<T>();
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
        unsafe {
            let repr: **SliceRepr = transmute(&v);
            transmute(&((**repr).data))
        }
    }

    /** see `to_ptr()` */
    #[inline]
    pub fn to_mut_ptr<T>(v: &mut [T]) -> *mut T {
        unsafe {
            let repr: **SliceRepr = transmute(&v);
            transmute(&((**repr).data))
        }
    }

    /**
     * Form a slice from a pointer and length (as a number of units,
     * not bytes).
     */
    #[inline]
    pub unsafe fn buf_as_slice<T,U>(p: *T,
                                    len: uint,
                                    f: &fn(v: &[T]) -> U) -> U {
        let pair = (p, len * sys::nonzero_size_of::<T>());
        let v : *(&'blk [T]) = transmute(&pair);
        f(*v)
    }

    /**
     * Form a slice from a pointer and length (as a number of units,
     * not bytes).
     */
    #[inline]
    pub unsafe fn mut_buf_as_slice<T,U>(p: *mut T,
                                        len: uint,
                                        f: &fn(v: &mut [T]) -> U) -> U {
        let pair = (p, len * sys::nonzero_size_of::<T>());
        let v : *(&'blk mut [T]) = transmute(&pair);
        f(*v)
    }

    /**
     * Unchecked vector indexing.
     */
    #[inline]
    pub unsafe fn get<T:Copy>(v: &[T], i: uint) -> T {
        v.as_imm_buf(|p, _len| copy *ptr::offset(p, i))
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
            let box2 = util::replace(&mut box, None);
            intrinsics::move_val_init(&mut(*ptr::mut_offset(p, i)),
                                      box2.unwrap());
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
    use uint;
    use vec::raw;
    use vec;
    use ptr;

    /// A trait for operations on mutable operations on `[u8]`
    pub trait MutableByteVector {
        /// Sets all bytes of the receiver to the given value.
        pub fn set_memory(self, value: u8);
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
        let n = uint::min(a_len, b_len) as libc::size_t;
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
}

impl<A:Clone> Clone for ~[A] {
    #[inline]
    fn clone(&self) -> ~[A] {
        self.iter().transform(|item| item.clone()).collect()
    }
}

// This works because every lifetime is a sub-lifetime of 'static
impl<'self, A> Zero for &'self [A] {
    fn zero() -> &'self [A] { &'self [] }
    fn is_zero(&self) -> bool { self.is_empty() }
}

impl<A> Zero for ~[A] {
    fn zero() -> ~[A] { ~[] }
    fn is_zero(&self) -> bool { self.len() == 0 }
}

impl<A> Zero for @[A] {
    fn zero() -> @[A] { @[] }
    fn is_zero(&self) -> bool { self.len() == 0 }
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
    (impl $name:ident -> $elem:ty, $step:expr) => {
        // could be implemented with &[T] with .slice(), but this avoids bounds checks
        impl<'self, T> Iterator<$elem> for $name<'self, T> {
            #[inline]
            fn next(&mut self) -> Option<$elem> {
                unsafe {
                    if self.ptr == self.end {
                        None
                    } else {
                        let old = self.ptr;
                        self.ptr = self.ptr.offset($step);
                        Some(cast::transmute(old))
                    }
                }
            }

            #[inline]
            fn size_hint(&self) -> (uint, Option<uint>) {
                let diff = if $step > 0 {
                    (self.end as uint) - (self.ptr as uint)
                } else {
                    (self.ptr as uint) - (self.end as uint)
                };
                let exact = diff / size_of::<$elem>();
                (exact, Some(exact))
            }
        }
    }
}

//iterator!{struct VecIterator -> *T, &'self T}
/// An iterator for iterating over a vector.
pub struct VecIterator<'self, T> {
    priv ptr: *T,
    priv end: *T,
    priv lifetime: &'self T // FIXME: #5922
}
iterator!{impl VecIterator -> &'self T, 1}

//iterator!{struct VecRevIterator -> *T, &'self T}
/// An iterator for iterating over a vector in reverse.
pub struct VecRevIterator<'self, T> {
    priv ptr: *T,
    priv end: *T,
    priv lifetime: &'self T // FIXME: #5922
}
iterator!{impl VecRevIterator -> &'self T, -1}

//iterator!{struct VecMutIterator -> *mut T, &'self mut T}
/// An iterator for mutating the elements of a vector.
pub struct VecMutIterator<'self, T> {
    priv ptr: *mut T,
    priv end: *mut T,
    priv lifetime: &'self mut T // FIXME: #5922
}
iterator!{impl VecMutIterator -> &'self mut T, 1}

//iterator!{struct VecMutRevIterator -> *mut T, &'self mut T}
/// An iterator for mutating the elements of a vector in reverse.
pub struct VecMutRevIterator<'self, T> {
    priv ptr: *mut T,
    priv end: *mut T,
    priv lifetime: &'self mut T // FIXME: #5922
}
iterator!{impl VecMutRevIterator -> &'self mut T, -1}

/// An iterator that moves out of a vector.
pub struct VecConsumeIterator<T> {
    priv v: ~[T],
    priv idx: uint,
}

impl<T> Iterator<T> for VecConsumeIterator<T> {
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
}

/// An iterator that moves out of a vector in reverse order.
pub struct VecConsumeRevIterator<T> {
    priv v: ~[T]
}

impl<T> Iterator<T> for VecConsumeRevIterator<T> {
    fn next(&mut self) -> Option<T> {
        self.v.pop_opt()
    }
}

#[cfg(stage0)]
impl<A, T: Iterator<A>> FromIterator<A, T> for ~[A] {
    pub fn from_iterator(iterator: &mut T) -> ~[A] {
        let mut xs = ~[];
        for iterator.advance |x| {
            xs.push(x);
        }
        xs
    }
}


#[cfg(not(stage0))]
impl<A, T: Iterator<A>> FromIterator<A, T> for ~[A] {
    pub fn from_iterator(iterator: &mut T) -> ~[A] {
        let (lower, _) = iterator.size_hint();
        let mut xs = with_capacity(lower);
        for iterator.advance |x| {
            xs.push(x);
        }
        xs
    }
}


#[cfg(test)]
mod tests {
    use option::{None, Option, Some};
    use sys;
    use vec::*;
    use cmp::*;

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
        assert_eq!(sys::size_of::<Z>(), 0);
        assert_eq!(v0.len(), 0);
        assert_eq!(v1.len(), 1);
        assert_eq!(v2.len(), 2);
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
    #[ignore(cfg(windows))]
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
    #[ignore(cfg(windows))]
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
    #[ignore(cfg(windows))]
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

    #[init]
    #[should_fail]
    #[ignore(cfg(windows))]
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

    #[init]
    #[should_fail]
    #[ignore(cfg(windows))]
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
    #[ignore(cfg(windows))]
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
        let mut v = ~[::unstable::sync::exclusive(()),
                      ::unstable::sync::exclusive(()),
                      ::unstable::sync::exclusive(())];
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
    fn test_each_permutation() {
        let mut results: ~[~[int]];

        results = ~[];
        for each_permutation([]) |v| { results.push(to_owned(v)); }
        assert_eq!(results, ~[~[]]);

        results = ~[];
        for each_permutation([7]) |v| { results.push(to_owned(v)); }
        assert_eq!(results, ~[~[7]]);

        results = ~[];
        for each_permutation([1,1]) |v| { results.push(to_owned(v)); }
        assert_eq!(results, ~[~[1,1],~[1,1]]);

        results = ~[];
        for each_permutation([5,2,0]) |v| { results.push(to_owned(v)); }
        assert!(results ==
            ~[~[5,2,0],~[5,0,2],~[2,5,0],~[2,0,5],~[0,5,2],~[0,2,5]]);
    }

    #[test]
    fn test_zip_unzip() {
        let v1 = ~[1, 2, 3];
        let v2 = ~[4, 5, 6];

        let z1 = zip(v1, v2);

        assert_eq!((1, 4), z1[0]);
        assert_eq!((2, 5), z1[1]);
        assert_eq!((3, 6), z1[2]);

        let (left, right) = unzip(z1);

        assert_eq!((1, 4), (left[0], right[0]));
        assert_eq!((2, 5), (left[1], right[1]));
        assert_eq!((3, 6), (left[2], right[2]));
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
    fn test_rposition() {
        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        fn g(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'd' }
        let v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert_eq!(v.rposition(f), Some(3u));
        assert!(v.rposition(g).is_none());
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
        assert_eq!(concat([~[1], ~[2,3]]), ~[1, 2, 3]);
        assert_eq!([~[1], ~[2,3]].concat_vec(), ~[1, 2, 3]);

        assert_eq!(concat_slices([&[1], &[2,3]]), ~[1, 2, 3]);
        assert_eq!([&[1], &[2,3]].concat_vec(), ~[1, 2, 3]);
    }

    #[test]
    fn test_connect() {
        assert_eq!(connect([], &0), ~[]);
        assert_eq!(connect([~[1], ~[2, 3]], &0), ~[1, 0, 2, 3]);
        assert_eq!(connect([~[1], ~[2], ~[3]], &0), ~[1, 0, 2, 0, 3]);
        assert_eq!([~[1], ~[2, 3]].connect_vec(&0), ~[1, 0, 2, 3]);
        assert_eq!([~[1], ~[2], ~[3]].connect_vec(&0), ~[1, 0, 2, 0, 3]);

        assert_eq!(connect_slices([], &0), ~[]);
        assert_eq!(connect_slices([&[1], &[2, 3]], &0), ~[1, 0, 2, 3]);
        assert_eq!(connect_slices([&[1], &[2], &[3]], &0), ~[1, 0, 2, 0, 3]);
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
    #[ignore(cfg(windows))]
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
    #[ignore(cfg(windows))]
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
    #[ignore(windows)]
    #[should_fail]
    fn test_from_fn_fail() {
        do from_fn(100) |v| {
            if v == 50 { fail!() }
            (~0, @0)
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_build_fail() {
        do build |push| {
            push((~0, @0));
            push((~0, @0));
            push((~0, @0));
            push((~0, @0));
            fail!();
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    #[allow(non_implicitly_copyable_typarams)]
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
    #[ignore(windows)]
    #[should_fail]
    fn test_map_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do v.map |_elt| {
            if i == 2 {
                fail!()
            }
            i += 0;
            ~[(~0, @0)]
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_flat_map_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do flat_map(v) |_elt| {
            if i == 2 {
                fail!()
            }
            i += 0;
            ~[(~0, @0)]
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_rposition_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do v.rposition |_elt| {
            if i == 2 {
                fail!()
            }
            i += 0;
            false
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    #[allow(non_implicitly_copyable_typarams)]
    fn test_permute_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        for each_permutation(v) |_elt| {
            if i == 2 {
                fail!()
            }
            i += 0;
        }
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_as_imm_buf_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        do v.as_imm_buf |_buf, _i| {
            fail!()
        }
    }

    #[test]
    #[ignore(cfg(windows))]
    #[should_fail]
    fn test_as_mut_buf_fail() {
        let mut v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        do v.as_mut_buf |_buf, _i| {
            fail!()
        }
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
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
        use iterator::*;
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
    fn test_iter_size_hints() {
        use iterator::*;
        let mut xs = [1, 2, 5, 10, 11];
        assert_eq!(xs.iter().size_hint(), (5, Some(5)));
        assert_eq!(xs.rev_iter().size_hint(), (5, Some(5)));
        assert_eq!(xs.mut_iter().size_hint(), (5, Some(5)));
        assert_eq!(xs.mut_rev_iter().size_hint(), (5, Some(5)));
    }

    #[test]
    fn test_mut_iterator() {
        use iterator::*;
        let mut xs = [1, 2, 3, 4, 5];
        for xs.mut_iter().advance |x| {
            *x += 1;
        }
        assert_eq!(xs, [2, 3, 4, 5, 6])
    }

    #[test]
    fn test_rev_iterator() {
        use iterator::*;

        let xs = [1, 2, 5, 10, 11];
        let ys = [11, 10, 5, 2, 1];
        let mut i = 0;
        for xs.rev_iter().advance |&x| {
            assert_eq!(x, ys[i]);
            i += 1;
        }
        assert_eq!(i, 5);
    }

    #[test]
    fn test_mut_rev_iterator() {
        use iterator::*;
        let mut xs = [1u, 2, 3, 4, 5];
        for xs.mut_rev_iter().enumerate().advance |(i,x)| {
            *x += i;
        }
        assert_eq!(xs, [5, 5, 5, 5, 5])
    }

    #[test]
    fn test_consume_iterator() {
        use iterator::*;
        let xs = ~[1u,2,3,4,5];
        assert_eq!(xs.consume_iter().fold(0, |a: uint, b: uint| 10*a + b), 12345);
    }

    #[test]
    fn test_consume_rev_iterator() {
        use iterator::*;
        let xs = ~[1u,2,3,4,5];
        assert_eq!(xs.consume_rev_iter().fold(0, |a: uint, b: uint| 10*a + b), 54321);
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
    #[ignore(cfg(windows))]
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
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
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
    fn test_permutations0() {
        let values = [];
        let mut v : ~[~[int]] = ~[];
        for each_permutation(values) |p| {
            v.push(p.to_owned());
        }
        assert_eq!(v, ~[~[]]);
    }

    #[test]
    fn test_permutations1() {
        let values = [1];
        let mut v : ~[~[int]] = ~[];
        for each_permutation(values) |p| {
            v.push(p.to_owned());
        }
        assert_eq!(v, ~[~[1]]);
    }

    #[test]
    fn test_permutations2() {
        let values = [1,2];
        let mut v : ~[~[int]] = ~[];
        for each_permutation(values) |p| {
            v.push(p.to_owned());
        }
        assert_eq!(v, ~[~[1,2],~[2,1]]);
    }

    #[test]
    fn test_permutations3() {
        let values = [1,2,3];
        let mut v : ~[~[int]] = ~[];
        for each_permutation(values) |p| {
            v.push(p.to_owned());
        }
        assert_eq!(v, ~[~[1,2,3],~[1,3,2],~[2,1,3],~[2,3,1],~[3,1,2],~[3,2,1]]);
    }

    #[test]
    fn test_vec_zero() {
        use num::Zero;
        macro_rules! t (
            ($ty:ty) => {{
                let v: $ty = Zero::zero();
                assert!(v.is_empty());
                assert!(v.is_zero());
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
}
