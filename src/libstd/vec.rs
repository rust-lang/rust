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
use cmp::{Eq, Ord, TotalEq, TotalOrd, Ordering, Less, Equal, Greater};
use clone::Clone;
use iterator::{FromIterator, Iterator, IteratorUtil};
use iter::FromIter;
use kinds::Copy;
use libc;
use num::Zero;
use ops::Add;
use option::{None, Option, Some};
use ptr::to_unsafe_ptr;
use ptr;
use ptr::RawPtr;
use sys;
use sys::size_of;
use uint;
use unstable::intrinsics;
#[cfg(stage0)]
use intrinsic::{get_tydesc};
#[cfg(not(stage0))]
use unstable::intrinsics::{get_tydesc};
use vec;
use util;

#[cfg(not(test))] use cmp::Equiv;

#[doc(hidden)]
pub mod rustrt {
    use libc;
    use vec::raw;
    #[cfg(stage0)]
    use intrinsic::{TyDesc};
    #[cfg(not(stage0))]
    use unstable::intrinsics::{TyDesc};

    #[abi = "cdecl"]
    pub extern {
        // These names are terrible. reserve_shared applies
        // to ~[] and reserve_shared_actual applies to @[].
        #[fast_ffi]
        unsafe fn vec_reserve_shared(t: *TyDesc,
                                     v: **raw::VecRepr,
                                     n: libc::size_t);
        #[fast_ffi]
        unsafe fn vec_reserve_shared_actual(t: *TyDesc,
                                            v: **raw::VecRepr,
                                            n: libc::size_t);
    }
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
        do as_mut_buf(v) |p, _len| {
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
        do as_mut_buf(v) |p, _len| {
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

// Accessors

/// Copies

/// Split the vector `v` by applying each element against the predicate `f`.
pub fn split<T:Copy>(v: &[T], f: &fn(t: &T) -> bool) -> ~[~[T]] {
    let ln = v.len();
    if (ln == 0u) { return ~[] }

    let mut start = 0u;
    let mut result = ~[];
    while start < ln {
        match position_between(v, start, ln, |t| f(t)) {
            None => break,
            Some(i) => {
                result.push(v.slice(start, i).to_owned());
                start = i + 1u;
            }
        }
    }
    result.push(v.slice(start, ln).to_owned());
    result
}

/**
 * Split the vector `v` by applying each element against the predicate `f` up
 * to `n` times.
 */
pub fn splitn<T:Copy>(v: &[T], n: uint, f: &fn(t: &T) -> bool) -> ~[~[T]] {
    let ln = v.len();
    if (ln == 0u) { return ~[] }

    let mut start = 0u;
    let mut count = n;
    let mut result = ~[];
    while start < ln && count > 0u {
        match position_between(v, start, ln, |t| f(t)) {
            None => break,
            Some(i) => {
                result.push(v.slice(start, i).to_owned());
                // Make sure to skip the separator.
                start = i + 1u;
                count -= 1u;
            }
        }
    }
    result.push(v.slice(start, ln).to_owned());
    result
}

/**
 * Reverse split the vector `v` by applying each element against the predicate
 * `f`.
 */
pub fn rsplit<T:Copy>(v: &[T], f: &fn(t: &T) -> bool) -> ~[~[T]] {
    let ln = v.len();
    if (ln == 0) { return ~[] }

    let mut end = ln;
    let mut result = ~[];
    while end > 0 {
        match rposition_between(v, 0, end, |t| f(t)) {
            None => break,
            Some(i) => {
                result.push(v.slice(i + 1, end).to_owned());
                end = i;
            }
        }
    }
    result.push(v.slice(0u, end).to_owned());
    reverse(result);
    result
}

/**
 * Reverse split the vector `v` by applying each element against the predicate
 * `f` up to `n times.
 */
pub fn rsplitn<T:Copy>(v: &[T], n: uint, f: &fn(t: &T) -> bool) -> ~[~[T]] {
    let ln = v.len();
    if (ln == 0u) { return ~[] }

    let mut end = ln;
    let mut count = n;
    let mut result = ~[];
    while end > 0u && count > 0u {
        match rposition_between(v, 0u, end, |t| f(t)) {
            None => break,
            Some(i) => {
                result.push(v.slice(i + 1u, end).to_owned());
                // Make sure to skip the separator.
                end = i;
                count -= 1u;
            }
        }
    }
    result.push(v.slice(0u, end).to_owned());
    reverse(result);
    result
}

/// Consumes all elements, in a vector, moving them out into the / closure
/// provided. The vector is traversed from the start to the end.
///
/// This method does not impose any requirements on the type of the vector being
/// consumed, but it prevents any usage of the vector after this function is
/// called.
///
/// # Examples
///
/// ~~~ {.rust}
/// let v = ~[~"a", ~"b"];
/// do vec::consume(v) |i, s| {
///   // s has type ~str, not &~str
///   io::println(s + fmt!(" %d", i));
/// }
/// ~~~
pub fn consume<T>(mut v: ~[T], f: &fn(uint, v: T)) {
    unsafe {
        do as_mut_buf(v) |p, ln| {
            for uint::range(0, ln) |i| {
                // NB: This unsafe operation counts on init writing 0s to the
                // holes we create in the vector. That ensures that, if the
                // iterator fails then we won't try to clean up the consumed
                // elements during unwinding
                let x = intrinsics::init();
                let p = ptr::mut_offset(p, i);
                f(i, ptr::replace_ptr(p, x));
            }
        }

        raw::set_len(&mut v, 0);
    }
}

/// Consumes all elements, in a vector, moving them out into the / closure
/// provided. The vectors is traversed in reverse order (from end to start).
///
/// This method does not impose any requirements on the type of the vector being
/// consumed, but it prevents any usage of the vector after this function is
/// called.
pub fn consume_reverse<T>(mut v: ~[T], f: &fn(uint, v: T)) {
    unsafe {
        do as_mut_buf(v) |p, ln| {
            let mut i = ln;
            while i > 0 {
                i -= 1;

                // NB: This unsafe operation counts on init writing 0s to the
                // holes we create in the vector. That ensures that, if the
                // iterator fails then we won't try to clean up the consumed
                // elements during unwinding
                let x = intrinsics::init();
                let p = ptr::mut_offset(p, i);
                f(i, ptr::replace_ptr(p, x));
            }
        }

        raw::set_len(&mut v, 0);
    }
}

/**
 * Remove consecutive repeated elements from a vector; if the vector is
 * sorted, this removes all duplicates.
 */
pub fn dedup<T:Eq>(v: &mut ~[T]) {
    unsafe {
        if v.len() < 1 { return; }
        let mut last_written = 0;
        let mut next_to_read = 1;
        do as_mut_buf(*v) |p, ln| {
            // last_written < next_to_read <= ln
            while next_to_read < ln {
                // last_written < next_to_read < ln
                if *ptr::mut_offset(p, next_to_read) ==
                    *ptr::mut_offset(p, last_written) {
                    ptr::replace_ptr(ptr::mut_offset(p, next_to_read),
                                     intrinsics::uninit());
                } else {
                    last_written += 1;
                    // last_written <= next_to_read < ln
                    if next_to_read != last_written {
                        ptr::swap_ptr(ptr::mut_offset(p, last_written),
                                      ptr::mut_offset(p, next_to_read));
                    }
                }
                // last_written <= next_to_read < ln
                next_to_read += 1;
                // last_written < next_to_read <= ln
            }
        }
        // last_written < next_to_read == ln
        raw::set_len(v, last_written + 1);
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

/**
 * Expands a vector in place, initializing the new elements to a given value
 *
 * # Arguments
 *
 * * v - The vector to grow
 * * n - The number of elements to add
 * * initval - The value for the new elements
 */
pub fn grow<T:Copy>(v: &mut ~[T], n: uint, initval: &T) {
    let new_len = v.len() + n;
    v.reserve_at_least(new_len);
    let mut i: uint = 0u;

    while i < n {
        v.push(copy *initval);
        i += 1u;
    }
}

/**
 * Expands a vector in place, initializing the new elements to the result of
 * a function
 *
 * Function `init_op` is called `n` times with the values [0..`n`)
 *
 * # Arguments
 *
 * * v - The vector to grow
 * * n - The number of elements to add
 * * init_op - A function to call to retreive each appended element's
 *             value
 */
pub fn grow_fn<T>(v: &mut ~[T], n: uint, op: &fn(uint) -> T) {
    let new_len = v.len() + n;
    v.reserve_at_least(new_len);
    let mut i: uint = 0u;
    while i < n {
        v.push(op(i));
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
pub fn grow_set<T:Copy>(v: &mut ~[T], index: uint, initval: &T, val: T) {
    let l = v.len();
    if index >= l { grow(&mut *v, index - l + 1u, initval); }
    v[index] = val;
}

// Functional utilities

/// Apply a function to each element of a vector and return the results
pub fn map<T, U>(v: &[T], f: &fn(t: &T) -> U) -> ~[U] {
    let mut result = with_capacity(v.len());
    for v.iter().advance |elem| {
        result.push(f(elem));
    }
    result
}

/// Consumes a vector, mapping it into a different vector. This function takes
/// ownership of the supplied vector `v`, moving each element into the closure
/// provided to generate a new element. The vector of new elements is then
/// returned.
///
/// The original vector `v` cannot be used after this function call (it is moved
/// inside), but there are no restrictions on the type of the vector.
pub fn map_consume<T, U>(v: ~[T], f: &fn(v: T) -> U) -> ~[U] {
    let mut result = ~[];
    do consume(v) |_i, x| {
        result.push(f(x));
    }
    result
}

/// Apply a function to each element of a vector and return the results
pub fn mapi<T, U>(v: &[T], f: &fn(uint, t: &T) -> U) -> ~[U] {
    let mut i = 0;
    do map(v) |e| {
        i += 1;
        f(i - 1, e)
    }
}

/**
 * Apply a function to each element of a vector and return a concatenation
 * of each result vector
 */
pub fn flat_map<T, U>(v: &[T], f: &fn(t: &T) -> ~[U]) -> ~[U] {
    let mut result = ~[];
    for v.iter().advance |elem| { result.push_all_move(f(elem)); }
    result
}

/**
 * Apply a function to each pair of elements and return the results.
 * Equivalent to `map(zip(v0, v1), f)`.
 */
pub fn map_zip<T:Copy,U:Copy,V>(v0: &[T], v1: &[U],
                                  f: &fn(t: &T, v: &U) -> V) -> ~[V] {
    let v0_len = v0.len();
    if v0_len != v1.len() { fail!(); }
    let mut u: ~[V] = ~[];
    let mut i = 0u;
    while i < v0_len {
        u.push(f(&v0[i], &v1[i]));
        i += 1u;
    }
    u
}

pub fn filter_map<T, U>(
    v: ~[T],
    f: &fn(t: T) -> Option<U>) -> ~[U]
{
    /*!
     *
     * Apply a function to each element of a vector and return the results.
     * Consumes the input vector.  If function `f` returns `None` then that
     * element is excluded from the resulting vector.
     */

    let mut result = ~[];
    do consume(v) |_, elem| {
        match f(elem) {
            None => {}
            Some(result_elem) => { result.push(result_elem); }
        }
    }
    result
}

pub fn filter_mapped<T, U: Copy>(
    v: &[T],
    f: &fn(t: &T) -> Option<U>) -> ~[U]
{
    /*!
     *
     * Like `filter_map()`, but operates on a borrowed slice
     * and does not consume the input.
     */

    let mut result = ~[];
    for v.iter().advance |elem| {
        match f(elem) {
          None => {/* no-op */ }
          Some(result_elem) => { result.push(result_elem); }
        }
    }
    result
}

/**
 * Construct a new vector from the elements of a vector for which some
 * predicate holds.
 *
 * Apply function `f` to each element of `v` and return a vector containing
 * only those elements for which `f` returned true.
 */
pub fn filter<T>(v: ~[T], f: &fn(t: &T) -> bool) -> ~[T] {
    let mut result = ~[];
    // FIXME (#4355 maybe): using v.consume here crashes
    // do v.consume |_, elem| {
    do consume(v) |_, elem| {
        if f(&elem) { result.push(elem); }
    }
    result
}

/**
 * Construct a new vector from the elements of a vector for which some
 * predicate holds.
 *
 * Apply function `f` to each element of `v` and return a vector containing
 * only those elements for which `f` returned true.
 */
pub fn filtered<T:Copy>(v: &[T], f: &fn(t: &T) -> bool) -> ~[T] {
    let mut result = ~[];
    for v.iter().advance |elem| {
        if f(elem) { result.push(copy *elem); }
    }
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
        self.flat_map(|&inner| inner)
    }

    /// Concatenate a vector of vectors, placing a given separator between each.
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

/// Return true if a vector contains an element with the given value
pub fn contains<T:Eq>(v: &[T], x: &T) -> bool {
    for v.iter().advance |elt| { if *x == *elt { return true; } }
    false
}

/**
 * Search for the first element that matches a given predicate within a range
 *
 * Apply function `f` to each element of `v` within the range
 * [`start`, `end`). When function `f` returns true then an option containing
 * the element is returned. If `f` matches no elements then none is returned.
 */
pub fn find_between<T:Copy>(v: &[T], start: uint, end: uint,
                      f: &fn(t: &T) -> bool) -> Option<T> {
    position_between(v, start, end, f).map(|i| copy v[*i])
}

/**
 * Search for the last element that matches a given predicate
 *
 * Apply function `f` to each element of `v` in reverse order. When function
 * `f` returns true then an option containing the element is returned. If `f`
 * matches no elements then none is returned.
 */
pub fn rfind<T:Copy>(v: &[T], f: &fn(t: &T) -> bool) -> Option<T> {
    rfind_between(v, 0u, v.len(), f)
}

/**
 * Search for the last element that matches a given predicate within a range
 *
 * Apply function `f` to each element of `v` in reverse order within the range
 * [`start`, `end`). When function `f` returns true then an option containing
 * the element is returned. If `f` matches no elements then none is return.
 */
pub fn rfind_between<T:Copy>(v: &[T],
                             start: uint,
                             end: uint,
                             f: &fn(t: &T) -> bool)
                          -> Option<T> {
    rposition_between(v, start, end, f).map(|i| copy v[*i])
}

/// Find the first index containing a matching value
pub fn position_elem<T:Eq>(v: &[T], x: &T) -> Option<uint> {
    v.iter().position_(|y| *x == *y)
}

/**
 * Find the first index matching some predicate within a range
 *
 * Apply function `f` to each element of `v` between the range
 * [`start`, `end`). When function `f` returns true then an option containing
 * the index is returned. If `f` matches no elements then none is returned.
 */
pub fn position_between<T>(v: &[T],
                           start: uint,
                           end: uint,
                           f: &fn(t: &T) -> bool)
                        -> Option<uint> {
    assert!(start <= end);
    assert!(end <= v.len());
    let mut i = start;
    while i < end { if f(&v[i]) { return Some::<uint>(i); } i += 1u; }
    None
}

/// Find the last index containing a matching value
pub fn rposition_elem<T:Eq>(v: &[T], x: &T) -> Option<uint> {
    rposition(v, |y| *x == *y)
}

/**
 * Find the last index matching some predicate
 *
 * Apply function `f` to each element of `v` in reverse order.  When function
 * `f` returns true then an option containing the index is returned. If `f`
 * matches no elements then none is returned.
 */
pub fn rposition<T>(v: &[T], f: &fn(t: &T) -> bool) -> Option<uint> {
    rposition_between(v, 0u, v.len(), f)
}

/**
 * Find the last index matching some predicate within a range
 *
 * Apply function `f` to each element of `v` in reverse order between the
 * range [`start`, `end`). When function `f` returns true then an option
 * containing the index is returned. If `f` matches no elements then none is
 * returned.
 */
pub fn rposition_between<T>(v: &[T], start: uint, end: uint,
                             f: &fn(t: &T) -> bool) -> Option<uint> {
    assert!(start <= end);
    assert!(end <= v.len());
    let mut i = end;
    while i > start {
        if f(&v[i - 1u]) { return Some::<uint>(i - 1u); }
        i -= 1u;
    }
    None
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
pub fn bsearch<T>(v: &[T], f: &fn(&T) -> Ordering) -> Option<uint> {
    let mut base : uint = 0;
    let mut lim : uint = v.len();

    while lim != 0 {
        let ix = base + (lim >> 1);
        match f(&v[ix]) {
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

/**
 * Binary search a sorted vector for a given element.
 *
 * Returns the index of the element or None if not found.
 */
pub fn bsearch_elem<T:TotalOrd>(v: &[T], x: &T) -> Option<uint> {
    bsearch(v, |p| p.cmp(x))
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
    do consume(v) |_i, p| {
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
    reverse(w);
    w
}

/**
 * Swaps two elements in a vector
 *
 * # Arguments
 *
 * * v  The input vector
 * * a - The index of the first element
 * * b - The index of the second element
 */
#[inline]
pub fn swap<T>(v: &mut [T], a: uint, b: uint) {
    unsafe {
        // Can't take two mutable loans from one vector, so instead just cast
        // them to their raw pointers to do the swap
        let pa: *mut T = &mut v[a];
        let pb: *mut T = &mut v[b];
        ptr::swap_ptr(pa, pb);
    }
}

/// Reverse the order of elements in a vector, in place
pub fn reverse<T>(v: &mut [T]) {
    let mut i: uint = 0;
    let ln = v.len();
    while i < ln / 2 {
        swap(v, i, ln - i - 1);
        i += 1;
    }
}

/// Returns a vector with the order of elements reversed
pub fn reversed<T:Copy>(v: &[T]) -> ~[T] {
    let mut rs: ~[T] = ~[];
    let mut i = v.len();
    if i == 0 { return (rs); } else { i -= 1; }
    while i != 0 { rs.push(copy v[i]); i -= 1; }
    rs.push(copy v[0]);
    rs
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
        vec::swap(indices, k, l);
        reverse(indices.mut_slice(k+1, length));
        // fixup permutation based on indices
        for uint::range(k, length) |i| {
            permutation[i] = copy values[indices[i]];
        }
    }
}

/**
 * Iterate over all contiguous windows of length `n` of the vector `v`.
 *
 * # Example
 *
 * Print the adjacent pairs of a vector (i.e. `[1,2]`, `[2,3]`, `[3,4]`)
 *
 * ~~~ {.rust}
 * for windowed(2, &[1,2,3,4]) |v| {
 *     io::println(fmt!("%?", v));
 * }
 * ~~~
 *
 */
pub fn windowed<'r, T>(n: uint, v: &'r [T], it: &fn(&'r [T]) -> bool) -> bool {
    assert!(1u <= n);
    if n > v.len() { return true; }
    for uint::range(0, v.len() - n + 1) |i| {
        if !it(v.slice(i, i + n)) { return false; }
    }
    return true;
}

/**
 * Work with the buffer of a vector.
 *
 * Allows for unsafe manipulation of vector contents, which is useful for
 * foreign interop.
 */
#[inline]
pub fn as_imm_buf<T,U>(s: &[T],
                       /* NB---this CANNOT be const, see below */
                       f: &fn(*T, uint) -> U) -> U {

    // NB---Do not change the type of s to `&const [T]`.  This is
    // unsound.  The reason is that we are going to create immutable pointers
    // into `s` and pass them to `f()`, but in fact they are potentially
    // pointing at *mutable memory*.  Use `as_const_buf` or `as_mut_buf`
    // instead!

    unsafe {
        let v : *(*T,uint) = transmute(&s);
        let (buf,len) = *v;
        f(buf, len / sys::nonzero_size_of::<T>())
    }
}

/// Similar to `as_imm_buf` but passing a `*mut T`
#[inline]
pub fn as_mut_buf<T,U>(s: &mut [T], f: &fn(*mut T, uint) -> U) -> U {
    unsafe {
        let v : *(*mut T,uint) = transmute(&s);
        let (buf,len) = *v;
        f(buf, len / sys::nonzero_size_of::<T>())
    }
}

// Equality

/// Tests whether two slices are equal to one another. This is only true if both
/// slices are of the same length, and each of the corresponding elements return
/// true when queried via the `eq` function.
fn eq<T: Eq>(a: &[T], b: &[T]) -> bool {
    let (a_len, b_len) = (a.len(), b.len());
    if a_len != b_len { return false; }

    let mut i = 0;
    while i < a_len {
        if a[i] != b[i] { return false; }
        i += 1;
    }
    true
}

/// Similar to the `vec::eq` function, but this is defined for types which
/// implement `TotalEq` as opposed to types which implement `Eq`. Equality
/// comparisons are done via the `equals` function instead of `eq`.
fn equals<T: TotalEq>(a: &[T], b: &[T]) -> bool {
    let (a_len, b_len) = (a.len(), b.len());
    if a_len != b_len { return false; }

    let mut i = 0;
    while i < a_len {
        if !a[i].equals(&b[i]) { return false; }
        i += 1;
    }
    true
}

#[cfg(not(test))]
impl<'self,T:Eq> Eq for &'self [T] {
    #[inline]
    fn eq(&self, other: & &'self [T]) -> bool { eq(*self, *other) }
    #[inline]
    fn ne(&self, other: & &'self [T]) -> bool { !self.eq(other) }
}

#[cfg(not(test))]
impl<T:Eq> Eq for ~[T] {
    #[inline]
    fn eq(&self, other: &~[T]) -> bool { eq(*self, *other) }
    #[inline]
    fn ne(&self, other: &~[T]) -> bool { !self.eq(other) }
}

#[cfg(not(test))]
impl<T:Eq> Eq for @[T] {
    #[inline]
    fn eq(&self, other: &@[T]) -> bool { eq(*self, *other) }
    #[inline]
    fn ne(&self, other: &@[T]) -> bool { !self.eq(other) }
}

#[cfg(not(test))]
impl<'self,T:TotalEq> TotalEq for &'self [T] {
    #[inline]
    fn equals(&self, other: & &'self [T]) -> bool { equals(*self, *other) }
}

#[cfg(not(test))]
impl<T:TotalEq> TotalEq for ~[T] {
    #[inline]
    fn equals(&self, other: &~[T]) -> bool { equals(*self, *other) }
}

#[cfg(not(test))]
impl<T:TotalEq> TotalEq for @[T] {
    #[inline]
    fn equals(&self, other: &@[T]) -> bool { equals(*self, *other) }
}

#[cfg(not(test))]
impl<'self,T:Eq> Equiv<~[T]> for &'self [T] {
    #[inline]
    fn equiv(&self, other: &~[T]) -> bool { eq(*self, *other) }
}

// Lexicographical comparison

fn cmp<T: TotalOrd>(a: &[T], b: &[T]) -> Ordering {
    let low = uint::min(a.len(), b.len());

    for uint::range(0, low) |idx| {
        match a[idx].cmp(&b[idx]) {
          Greater => return Greater,
          Less => return Less,
          Equal => ()
        }
    }

    a.len().cmp(&b.len())
}

#[cfg(not(test))]
impl<'self,T:TotalOrd> TotalOrd for &'self [T] {
    #[inline]
    fn cmp(&self, other: & &'self [T]) -> Ordering { cmp(*self, *other) }
}

#[cfg(not(test))]
impl<T: TotalOrd> TotalOrd for ~[T] {
    #[inline]
    fn cmp(&self, other: &~[T]) -> Ordering { cmp(*self, *other) }
}

#[cfg(not(test))]
impl<T: TotalOrd> TotalOrd for @[T] {
    #[inline]
    fn cmp(&self, other: &@[T]) -> Ordering { cmp(*self, *other) }
}

fn lt<T:Ord>(a: &[T], b: &[T]) -> bool {
    let (a_len, b_len) = (a.len(), b.len());
    let end = uint::min(a_len, b_len);

    let mut i = 0;
    while i < end {
        let (c_a, c_b) = (&a[i], &b[i]);
        if *c_a < *c_b { return true; }
        if *c_a > *c_b { return false; }
        i += 1;
    }

    a_len < b_len
}

fn le<T:Ord>(a: &[T], b: &[T]) -> bool { !lt(b, a) }
fn ge<T:Ord>(a: &[T], b: &[T]) -> bool { !lt(a, b) }
fn gt<T:Ord>(a: &[T], b: &[T]) -> bool { lt(b, a)  }

#[cfg(not(test))]
impl<'self,T:Ord> Ord for &'self [T] {
    #[inline]
    fn lt(&self, other: & &'self [T]) -> bool { lt((*self), (*other)) }
    #[inline]
    fn le(&self, other: & &'self [T]) -> bool { le((*self), (*other)) }
    #[inline]
    fn ge(&self, other: & &'self [T]) -> bool { ge((*self), (*other)) }
    #[inline]
    fn gt(&self, other: & &'self [T]) -> bool { gt((*self), (*other)) }
}

#[cfg(not(test))]
impl<T:Ord> Ord for ~[T] {
    #[inline]
    fn lt(&self, other: &~[T]) -> bool { lt((*self), (*other)) }
    #[inline]
    fn le(&self, other: &~[T]) -> bool { le((*self), (*other)) }
    #[inline]
    fn ge(&self, other: &~[T]) -> bool { ge((*self), (*other)) }
    #[inline]
    fn gt(&self, other: &~[T]) -> bool { gt((*self), (*other)) }
}

#[cfg(not(test))]
impl<T:Ord> Ord for @[T] {
    #[inline]
    fn lt(&self, other: &@[T]) -> bool { lt((*self), (*other)) }
    #[inline]
    fn le(&self, other: &@[T]) -> bool { le((*self), (*other)) }
    #[inline]
    fn ge(&self, other: &@[T]) -> bool { ge((*self), (*other)) }
    #[inline]
    fn gt(&self, other: &@[T]) -> bool { gt((*self), (*other)) }
}

#[cfg(not(test))]
impl<'self,T:Copy> Add<&'self [T], ~[T]> for ~[T] {
    #[inline]
    fn add(&self, rhs: & &'self [T]) -> ~[T] {
        append(copy *self, (*rhs))
    }
}

impl<'self, T> Container for &'self [T] {
    /// Returns true if a vector contains no elements
    #[inline]
    fn is_empty(&self) -> bool {
        as_imm_buf(*self, |_p, len| len == 0u)
    }

    /// Returns the length of a vector
    #[inline]
    fn len(&self) -> uint {
        as_imm_buf(*self, |_p, len| len)
    }
}

impl<T> Container for ~[T] {
    /// Returns true if a vector contains no elements
    #[inline]
    fn is_empty(&self) -> bool {
        as_imm_buf(*self, |_p, len| len == 0u)
    }

    /// Returns the length of a vector
    #[inline]
    fn len(&self) -> uint {
        as_imm_buf(*self, |_p, len| len)
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
    fn head(&self) -> &'self T;
    fn head_opt(&self) -> Option<&'self T>;
    fn tail(&self) -> &'self [T];
    fn tailn(&self, n: uint) -> &'self [T];
    fn init(&self) -> &'self [T];
    fn initn(&self, n: uint) -> &'self [T];
    fn last(&self) -> &'self T;
    fn last_opt(&self) -> Option<&'self T>;
    fn rposition(&self, f: &fn(t: &T) -> bool) -> Option<uint>;
    fn map<U>(&self, f: &fn(t: &T) -> U) -> ~[U];
    fn mapi<U>(&self, f: &fn(uint, t: &T) -> U) -> ~[U];
    fn map_r<U>(&self, f: &fn(x: &T) -> U) -> ~[U];
    fn flat_map<U>(&self, f: &fn(t: &T) -> ~[U]) -> ~[U];
    fn filter_mapped<U:Copy>(&self, f: &fn(t: &T) -> Option<U>) -> ~[U];
    unsafe fn unsafe_ref(&self, index: uint) -> *T;
}

/// Extension methods for vectors
impl<'self,T> ImmutableVector<'self, T> for &'self [T] {
    /// Return a slice that points into another slice.
    #[inline]
    fn slice(&self, start: uint, end: uint) -> &'self [T] {
    assert!(start <= end);
    assert!(end <= self.len());
        do as_imm_buf(*self) |p, _len| {
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
     * returned. If `f` matches no elements then none is returned.
     */
    #[inline]
    fn rposition(&self, f: &fn(t: &T) -> bool) -> Option<uint> {
        rposition(*self, f)
    }

    /// Apply a function to each element of a vector and return the results
    #[inline]
    fn map<U>(&self, f: &fn(t: &T) -> U) -> ~[U] { map(*self, f) }

    /**
     * Apply a function to the index and value of each element in the vector
     * and return the results
     */
    fn mapi<U>(&self, f: &fn(uint, t: &T) -> U) -> ~[U] {
        mapi(*self, f)
    }

    #[inline]
    fn map_r<U>(&self, f: &fn(x: &T) -> U) -> ~[U] {
        let mut r = ~[];
        let mut i = 0;
        while i < self.len() {
            r.push(f(&self[i]));
            i += 1;
        }
        r
    }

    /**
     * Apply a function to each element of a vector and return a concatenation
     * of each result vector
     */
    #[inline]
    fn flat_map<U>(&self, f: &fn(t: &T) -> ~[U]) -> ~[U] {
        flat_map(*self, f)
    }
    /**
     * Apply a function to each element of a vector and return the results
     *
     * If function `f` returns `none` then that element is excluded from
     * the resulting vector.
     */
    #[inline]
    fn filter_mapped<U:Copy>(&self, f: &fn(t: &T) -> Option<U>) -> ~[U] {
        filter_mapped(*self, f)
    }

    /// Returns a pointer to the element at the given index, without doing
    /// bounds checking.
    #[inline]
    unsafe fn unsafe_ref(&self, index: uint) -> *T {
        let (ptr, _): (*T, uint) = transmute(*self);
        ptr.offset(index)
    }
}

#[allow(missing_doc)]
pub trait ImmutableEqVector<T:Eq> {
    fn position_elem(&self, t: &T) -> Option<uint>;
    fn rposition_elem(&self, t: &T) -> Option<uint>;
}

impl<'self,T:Eq> ImmutableEqVector<T> for &'self [T] {
    /// Find the first index containing a matching value
    #[inline]
    fn position_elem(&self, x: &T) -> Option<uint> {
        position_elem(*self, x)
    }

    /// Find the last index containing a matching value
    #[inline]
    fn rposition_elem(&self, t: &T) -> Option<uint> {
        rposition_elem(*self, t)
    }
}

#[allow(missing_doc)]
pub trait ImmutableCopyableVector<T> {
    fn filtered(&self, f: &fn(&T) -> bool) -> ~[T];
    fn rfind(&self, f: &fn(t: &T) -> bool) -> Option<T>;
    fn partitioned(&self, f: &fn(&T) -> bool) -> (~[T], ~[T]);
    unsafe fn unsafe_get(&self, elem: uint) -> T;
}

/// Extension methods for vectors
impl<'self,T:Copy> ImmutableCopyableVector<T> for &'self [T] {
    /**
     * Construct a new vector from the elements of a vector for which some
     * predicate holds.
     *
     * Apply function `f` to each element of `v` and return a vector
     * containing only those elements for which `f` returned true.
     */
    #[inline]
    fn filtered(&self, f: &fn(t: &T) -> bool) -> ~[T] {
        filtered(*self, f)
    }

    /**
     * Search for the last element that matches a given predicate
     *
     * Apply function `f` to each element of `v` in reverse order. When
     * function `f` returns true then an option containing the element is
     * returned. If `f` matches no elements then none is returned.
     */
    #[inline]
    fn rfind(&self, f: &fn(t: &T) -> bool) -> Option<T> {
        rfind(*self, f)
    }

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
    fn reserve(&mut self, n: uint);
    fn reserve_at_least(&mut self, n: uint);
    fn capacity(&self) -> uint;

    fn push(&mut self, t: T);
    unsafe fn push_fast(&mut self, t: T);

    fn push_all_move(&mut self, rhs: ~[T]);
    fn pop(&mut self) -> T;
    fn shift(&mut self) -> T;
    fn unshift(&mut self, x: T);
    fn insert(&mut self, i: uint, x:T);
    fn remove(&mut self, i: uint) -> T;
    fn swap_remove(&mut self, index: uint) -> T;
    fn truncate(&mut self, newlen: uint);
    fn retain(&mut self, f: &fn(t: &T) -> bool);
    fn consume(self, f: &fn(uint, v: T));
    fn consume_reverse(self, f: &fn(uint, v: T));
    fn filter(self, f: &fn(t: &T) -> bool) -> ~[T];
    fn partition(self, f: &fn(&T) -> bool) -> (~[T], ~[T]);
    fn grow_fn(&mut self, n: uint, op: &fn(uint) -> T);
}

impl<T> OwnedVector<T> for ~[T] {
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
    fn reserve(&mut self, n: uint) {
        // Only make the (slow) call into the runtime if we have to
        use managed;
        if self.capacity() < n {
            unsafe {
                let ptr: **raw::VecRepr = cast::transmute(self);
                let td = get_tydesc::<T>();
                if ((**ptr).box_header.ref_count ==
                    managed::raw::RC_MANAGED_UNIQUE) {
                    rustrt::vec_reserve_shared_actual(td, ptr, n as libc::size_t);
                } else {
                    rustrt::vec_reserve_shared(td, ptr, n as libc::size_t);
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
        let new_len = self.len() + rhs.len();
        self.reserve(new_len);
        unsafe {
            do as_mut_buf(rhs) |p, len| {
                for uint::range(0, len) |i| {
                    let x = ptr::replace_ptr(ptr::mut_offset(p, i),
                                             intrinsics::uninit());
                    self.push(x);
                }
            }
            raw::set_len(&mut rhs, 0);
        }
    }

    /// Remove the last element from a vector and return it
    fn pop(&mut self) -> T {
        let ln = self.len();
        if ln == 0 {
            fail!("sorry, cannot pop an empty vector")
        }
        let valptr = ptr::to_mut_unsafe_ptr(&mut self[ln - 1u]);
        unsafe {
            let val = ptr::replace_ptr(valptr, intrinsics::init());
            raw::set_len(self, ln - 1u);
            val
        }
    }

    /// Removes the first element from a vector and return it
    fn shift(&mut self) -> T {
        unsafe {
            assert!(!self.is_empty());

            if self.len() == 1 { return self.pop() }

            if self.len() == 2 {
                let last = self.pop();
                let first = self.pop();
                self.push(last);
                return first;
            }

            let ln = self.len();
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

            ptr::replace_ptr(vp, work_elt)
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
            swap(*self, j, j - 1);
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
            swap(*self, j, j + 1);
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
            swap(*self, index, ln - 1);
        }
        self.pop()
    }

    /// Shorten a vector, dropping excess elements.
    fn truncate(&mut self, newlen: uint) {
        do as_mut_buf(*self) |p, oldlen| {
            assert!(newlen <= oldlen);
            unsafe {
                // This loop is optimized out for non-drop types.
                for uint::range(newlen, oldlen) |i| {
                    ptr::replace_ptr(ptr::mut_offset(p, i), intrinsics::uninit());
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
                swap(*self, i - deleted, i);
            }
        }

        if deleted > 0 {
            self.truncate(len - deleted);
        }
    }

    #[inline]
    fn consume(self, f: &fn(uint, v: T)) {
        consume(self, f)
    }

    #[inline]
    fn consume_reverse(self, f: &fn(uint, v: T)) {
        consume_reverse(self, f)
    }

    #[inline]
    fn filter(self, f: &fn(&T) -> bool) -> ~[T] {
        filter(self, f)
    }

    /**
     * Partitions the vector into those that satisfies the predicate, and
     * those that do not.
     */
    #[inline]
    fn partition(self, f: &fn(&T) -> bool) -> (~[T], ~[T]) {
        let mut lefts  = ~[];
        let mut rights = ~[];

        do self.consume |_, elt| {
            if f(&elt) {
                lefts.push(elt);
            } else {
                rights.push(elt);
            }
        }

        (lefts, rights)
    }

    #[inline]
    fn grow_fn(&mut self, n: uint, op: &fn(uint) -> T) {
        grow_fn(self, n, op);
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

    #[inline]
    fn grow(&mut self, n: uint, initval: &T) {
        grow(self, n, initval);
    }

    #[inline]
    fn grow_set(&mut self, index: uint, initval: &T, val: T) {
        grow_set(self, index, initval, val);
    }
}

#[allow(missing_doc)]
trait OwnedEqVector<T:Eq> {
    fn dedup(&mut self);
}

impl<T:Eq> OwnedEqVector<T> for ~[T] {
    #[inline]
    fn dedup(&mut self) {
        dedup(self)
    }
}

#[allow(missing_doc)]
pub trait MutableVector<'self, T> {
    fn mut_slice(self, start: uint, end: uint) -> &'self mut [T];
    fn mut_iter(self) -> VecMutIterator<'self, T>;
    fn mut_rev_iter(self) -> VecMutRevIterator<'self, T>;

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
}

impl<'self,T> MutableVector<'self, T> for &'self mut [T] {
    /// Return a slice that points into another slice.
    #[inline]
    fn mut_slice(self, start: uint, end: uint) -> &'self mut [T] {
        assert!(start <= end);
        assert!(end <= self.len());
        do as_mut_buf(self) |p, _len| {
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
    use vec::{UnboxedVecRepr, as_imm_buf, as_mut_buf, with_capacity};
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
        as_imm_buf(v, |p, _len| copy *ptr::offset(p, i))
    }

    /**
     * Unchecked vector index assignment.  Does not drop the
     * old value and hence is only suitable when the vector
     * is newly allocated.
     */
    #[inline]
    pub unsafe fn init_elem<T>(v: &mut [T], i: uint, val: T) {
        let mut box = Some(val);
        do as_mut_buf(v) |p, _len| {
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
        as_mut_buf(dst, |p_dst, _len_dst| ptr::copy_memory(p_dst, ptr, elts));
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

        do as_mut_buf(dst) |p_dst, _len_dst| {
            do as_imm_buf(src) |p_src, _len_src| {
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
            do vec::as_mut_buf(self) |p, len| {
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
        self.map(|item| item.clone())
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
            fn size_hint(&self) -> (Option<uint>, Option<uint>) {
                let exact = Some(((self.end as uint) - (self.ptr as uint)) / size_of::<$elem>());
                (exact, exact)
            }
        }
    }
}

//iterator!{struct VecIterator -> *T, &'self T}
/// An iterator for iterating over a vector
pub struct VecIterator<'self, T> {
    priv ptr: *T,
    priv end: *T,
    priv lifetime: &'self T // FIXME: #5922
}
iterator!{impl VecIterator -> &'self T, 1}

//iterator!{struct VecRevIterator -> *T, &'self T}
/// An iterator for iterating over a vector in reverse
pub struct VecRevIterator<'self, T> {
    priv ptr: *T,
    priv end: *T,
    priv lifetime: &'self T // FIXME: #5922
}
iterator!{impl VecRevIterator -> &'self T, -1}

//iterator!{struct VecMutIterator -> *mut T, &'self mut T}
/// An iterator for mutating the elements of a vector
pub struct VecMutIterator<'self, T> {
    priv ptr: *mut T,
    priv end: *mut T,
    priv lifetime: &'self mut T // FIXME: #5922
}
iterator!{impl VecMutIterator -> &'self mut T, 1}

//iterator!{struct VecMutRevIterator -> *mut T, &'self mut T}
/// An iterator for mutating the elements of a vector in reverse
pub struct VecMutRevIterator<'self, T> {
    priv ptr: *mut T,
    priv end: *mut T,
    priv lifetime: &'self mut T // FIXME: #5922
}
iterator!{impl VecMutRevIterator -> &'self mut T, -1}

impl<T> FromIter<T> for ~[T]{
    #[inline]
    pub fn from_iter(iter: &fn(f: &fn(T) -> bool) -> bool) -> ~[T] {
        let mut v = ~[];
        for iter |x| { v.push(x) }
        v
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
        let mut xs = with_capacity(lower.get_or_zero());
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
        let mut v = ~[1u, 2u, 3u];
        let mut w = map(v, square_ref);
        assert_eq!(w.len(), 3u);
        assert_eq!(w[0], 1u);
        assert_eq!(w[1], 4u);
        assert_eq!(w[2], 9u);

        // Test on-heap map.
        v = ~[1u, 2u, 3u, 4u, 5u];
        w = map(v, square_ref);
        assert_eq!(w.len(), 5u);
        assert_eq!(w[0], 1u);
        assert_eq!(w[1], 4u);
        assert_eq!(w[2], 9u);
        assert_eq!(w[3], 16u);
        assert_eq!(w[4], 25u);
    }

    #[test]
    fn test_map_zip() {
        fn times(x: &int, y: &int) -> int { *x * *y }
        let f = times;
        let v0 = ~[1, 2, 3, 4, 5];
        let v1 = ~[5, 4, 3, 2, 1];
        let u = map_zip::<int, int, int>(v0, v1, f);
        let mut i = 0;
        while i < 5 { assert!(v0[i] * v1[i] == u[i]); i += 1; }
    }

    #[test]
    fn test_filter_mapped() {
        // Test on-stack filter-map.
        let mut v = ~[1u, 2u, 3u];
        let mut w = filter_mapped(v, square_if_odd_r);
        assert_eq!(w.len(), 2u);
        assert_eq!(w[0], 1u);
        assert_eq!(w[1], 9u);

        // Test on-heap filter-map.
        v = ~[1u, 2u, 3u, 4u, 5u];
        w = filter_mapped(v, square_if_odd_r);
        assert_eq!(w.len(), 3u);
        assert_eq!(w[0], 1u);
        assert_eq!(w[1], 9u);
        assert_eq!(w[2], 25u);

        fn halve(i: &int) -> Option<int> {
            if *i % 2 == 0 {
                Some::<int>(*i / 2)
            } else {
                None::<int>
            }
        }
        fn halve_for_sure(i: &int) -> int { *i / 2 }
        let all_even: ~[int] = ~[0, 2, 8, 6];
        let all_odd1: ~[int] = ~[1, 7, 3];
        let all_odd2: ~[int] = ~[];
        let mix: ~[int] = ~[9, 2, 6, 7, 1, 0, 0, 3];
        let mix_dest: ~[int] = ~[1, 3, 0, 0];
        assert!(filter_mapped(all_even, halve) ==
                     map(all_even, halve_for_sure));
        assert_eq!(filter_mapped(all_odd1, halve), ~[]);
        assert_eq!(filter_mapped(all_odd2, halve), ~[]);
        assert_eq!(filter_mapped(mix, halve), mix_dest);
    }

    #[test]
    fn test_filter_map() {
        // Test on-stack filter-map.
        let mut v = ~[1u, 2u, 3u];
        let mut w = filter_map(v, square_if_odd_v);
        assert_eq!(w.len(), 2u);
        assert_eq!(w[0], 1u);
        assert_eq!(w[1], 9u);

        // Test on-heap filter-map.
        v = ~[1u, 2u, 3u, 4u, 5u];
        w = filter_map(v, square_if_odd_v);
        assert_eq!(w.len(), 3u);
        assert_eq!(w[0], 1u);
        assert_eq!(w[1], 9u);
        assert_eq!(w[2], 25u);

        fn halve(i: int) -> Option<int> {
            if i % 2 == 0 {
                Some::<int>(i / 2)
            } else {
                None::<int>
            }
        }
        fn halve_for_sure(i: &int) -> int { *i / 2 }
        let all_even: ~[int] = ~[0, 2, 8, 6];
        let all_even0: ~[int] = copy all_even;
        let all_odd1: ~[int] = ~[1, 7, 3];
        let all_odd2: ~[int] = ~[];
        let mix: ~[int] = ~[9, 2, 6, 7, 1, 0, 0, 3];
        let mix_dest: ~[int] = ~[1, 3, 0, 0];
        assert!(filter_map(all_even, halve) ==
                     map(all_even0, halve_for_sure));
        assert_eq!(filter_map(all_odd1, halve), ~[]);
        assert_eq!(filter_map(all_odd2, halve), ~[]);
        assert_eq!(filter_map(mix, halve), mix_dest);
    }

    #[test]
    fn test_filter() {
        assert_eq!(filter(~[1u, 2u, 3u], is_odd), ~[1u, 3u]);
        assert_eq!(filter(~[1u, 2u, 4u, 8u, 16u], is_three), ~[]);
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
        assert!(position_elem([], &1).is_none());

        let v1 = ~[1, 2, 3, 3, 2, 5];
        assert_eq!(position_elem(v1, &1), Some(0u));
        assert_eq!(position_elem(v1, &2), Some(1u));
        assert_eq!(position_elem(v1, &5), Some(5u));
        assert!(position_elem(v1, &4).is_none());
    }

    #[test]
    fn test_position_between() {
        assert!(position_between([], 0u, 0u, f).is_none());

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        let v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert!(position_between(v, 0u, 0u, f).is_none());
        assert!(position_between(v, 0u, 1u, f).is_none());
        assert_eq!(position_between(v, 0u, 2u, f), Some(1u));
        assert_eq!(position_between(v, 0u, 3u, f), Some(1u));
        assert_eq!(position_between(v, 0u, 4u, f), Some(1u));

        assert!(position_between(v, 1u, 1u, f).is_none());
        assert_eq!(position_between(v, 1u, 2u, f), Some(1u));
        assert_eq!(position_between(v, 1u, 3u, f), Some(1u));
        assert_eq!(position_between(v, 1u, 4u, f), Some(1u));

        assert!(position_between(v, 2u, 2u, f).is_none());
        assert!(position_between(v, 2u, 3u, f).is_none());
        assert_eq!(position_between(v, 2u, 4u, f), Some(3u));

        assert!(position_between(v, 3u, 3u, f).is_none());
        assert_eq!(position_between(v, 3u, 4u, f), Some(3u));

        assert!(position_between(v, 4u, 4u, f).is_none());
    }

    #[test]
    fn test_find_between() {
        assert!(find_between([], 0u, 0u, f).is_none());

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        let v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert!(find_between(v, 0u, 0u, f).is_none());
        assert!(find_between(v, 0u, 1u, f).is_none());
        assert_eq!(find_between(v, 0u, 2u, f), Some((1, 'b')));
        assert_eq!(find_between(v, 0u, 3u, f), Some((1, 'b')));
        assert_eq!(find_between(v, 0u, 4u, f), Some((1, 'b')));

        assert!(find_between(v, 1u, 1u, f).is_none());
        assert_eq!(find_between(v, 1u, 2u, f), Some((1, 'b')));
        assert_eq!(find_between(v, 1u, 3u, f), Some((1, 'b')));
        assert_eq!(find_between(v, 1u, 4u, f), Some((1, 'b')));

        assert!(find_between(v, 2u, 2u, f).is_none());
        assert!(find_between(v, 2u, 3u, f).is_none());
        assert_eq!(find_between(v, 2u, 4u, f), Some((3, 'b')));

        assert!(find_between(v, 3u, 3u, f).is_none());
        assert_eq!(find_between(v, 3u, 4u, f), Some((3, 'b')));

        assert!(find_between(v, 4u, 4u, f).is_none());
    }

    #[test]
    fn test_rposition() {
        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        fn g(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'd' }
        let v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert_eq!(rposition(v, f), Some(3u));
        assert!(rposition(v, g).is_none());
    }

    #[test]
    fn test_rposition_between() {
        assert!(rposition_between([], 0u, 0u, f).is_none());

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        let v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert!(rposition_between(v, 0u, 0u, f).is_none());
        assert!(rposition_between(v, 0u, 1u, f).is_none());
        assert_eq!(rposition_between(v, 0u, 2u, f), Some(1u));
        assert_eq!(rposition_between(v, 0u, 3u, f), Some(1u));
        assert_eq!(rposition_between(v, 0u, 4u, f), Some(3u));

        assert!(rposition_between(v, 1u, 1u, f).is_none());
        assert_eq!(rposition_between(v, 1u, 2u, f), Some(1u));
        assert_eq!(rposition_between(v, 1u, 3u, f), Some(1u));
        assert_eq!(rposition_between(v, 1u, 4u, f), Some(3u));

        assert!(rposition_between(v, 2u, 2u, f).is_none());
        assert!(rposition_between(v, 2u, 3u, f).is_none());
        assert_eq!(rposition_between(v, 2u, 4u, f), Some(3u));

        assert!(rposition_between(v, 3u, 3u, f).is_none());
        assert_eq!(rposition_between(v, 3u, 4u, f), Some(3u));

        assert!(rposition_between(v, 4u, 4u, f).is_none());
    }

    #[test]
    fn test_rfind() {
        assert!(rfind([], f).is_none());

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        fn g(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'd' }
        let v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert_eq!(rfind(v, f), Some((3, 'b')));
        assert!(rfind(v, g).is_none());
    }

    #[test]
    fn test_rfind_between() {
        assert!(rfind_between([], 0u, 0u, f).is_none());

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        let v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert!(rfind_between(v, 0u, 0u, f).is_none());
        assert!(rfind_between(v, 0u, 1u, f).is_none());
        assert_eq!(rfind_between(v, 0u, 2u, f), Some((1, 'b')));
        assert_eq!(rfind_between(v, 0u, 3u, f), Some((1, 'b')));
        assert_eq!(rfind_between(v, 0u, 4u, f), Some((3, 'b')));

        assert!(rfind_between(v, 1u, 1u, f).is_none());
        assert_eq!(rfind_between(v, 1u, 2u, f), Some((1, 'b')));
        assert_eq!(rfind_between(v, 1u, 3u, f), Some((1, 'b')));
        assert_eq!(rfind_between(v, 1u, 4u, f), Some((3, 'b')));

        assert!(rfind_between(v, 2u, 2u, f).is_none());
        assert!(rfind_between(v, 2u, 3u, f).is_none());
        assert_eq!(rfind_between(v, 2u, 4u, f), Some((3, 'b')));

        assert!(rfind_between(v, 3u, 3u, f).is_none());
        assert_eq!(rfind_between(v, 3u, 4u, f), Some((3, 'b')));

        assert!(rfind_between(v, 4u, 4u, f).is_none());
    }

    #[test]
    fn test_bsearch_elem() {
        assert_eq!(bsearch_elem([1,2,3,4,5], &5), Some(4));
        assert_eq!(bsearch_elem([1,2,3,4,5], &4), Some(3));
        assert_eq!(bsearch_elem([1,2,3,4,5], &3), Some(2));
        assert_eq!(bsearch_elem([1,2,3,4,5], &2), Some(1));
        assert_eq!(bsearch_elem([1,2,3,4,5], &1), Some(0));

        assert_eq!(bsearch_elem([2,4,6,8,10], &1), None);
        assert_eq!(bsearch_elem([2,4,6,8,10], &5), None);
        assert_eq!(bsearch_elem([2,4,6,8,10], &4), Some(1));
        assert_eq!(bsearch_elem([2,4,6,8,10], &10), Some(4));

        assert_eq!(bsearch_elem([2,4,6,8], &1), None);
        assert_eq!(bsearch_elem([2,4,6,8], &5), None);
        assert_eq!(bsearch_elem([2,4,6,8], &4), Some(1));
        assert_eq!(bsearch_elem([2,4,6,8], &8), Some(3));

        assert_eq!(bsearch_elem([2,4,6], &1), None);
        assert_eq!(bsearch_elem([2,4,6], &5), None);
        assert_eq!(bsearch_elem([2,4,6], &4), Some(1));
        assert_eq!(bsearch_elem([2,4,6], &6), Some(2));

        assert_eq!(bsearch_elem([2,4], &1), None);
        assert_eq!(bsearch_elem([2,4], &5), None);
        assert_eq!(bsearch_elem([2,4], &2), Some(0));
        assert_eq!(bsearch_elem([2,4], &4), Some(1));

        assert_eq!(bsearch_elem([2], &1), None);
        assert_eq!(bsearch_elem([2], &5), None);
        assert_eq!(bsearch_elem([2], &2), Some(0));

        assert_eq!(bsearch_elem([], &1), None);
        assert_eq!(bsearch_elem([], &5), None);

        assert!(bsearch_elem([1,1,1,1,1], &1) != None);
        assert!(bsearch_elem([1,1,1,1,2], &1) != None);
        assert!(bsearch_elem([1,1,1,2,2], &1) != None);
        assert!(bsearch_elem([1,1,2,2,2], &1) != None);
        assert_eq!(bsearch_elem([1,2,2,2,2], &1), Some(0));

        assert_eq!(bsearch_elem([1,2,3,4,5], &6), None);
        assert_eq!(bsearch_elem([1,2,3,4,5], &0), None);
    }

    #[test]
    fn reverse_and_reversed() {
        let mut v: ~[int] = ~[10, 20];
        assert_eq!(v[0], 10);
        assert_eq!(v[1], 20);
        reverse(v);
        assert_eq!(v[0], 20);
        assert_eq!(v[1], 10);
        let v2 = reversed::<int>([10, 20]);
        assert_eq!(v2[0], 20);
        assert_eq!(v2[1], 10);
        v[0] = 30;
        assert_eq!(v2[0], 20);
        // Make sure they work with 0-length vectors too.

        let v4 = reversed::<int>([]);
        assert_eq!(v4, ~[]);
        let mut v3: ~[int] = ~[];
        reverse::<int>(v3);
    }

    #[test]
    fn reversed_mut() {
        let v2 = reversed::<int>([10, 20]);
        assert_eq!(v2[0], 20);
        assert_eq!(v2[1], 10);
    }

    #[test]
    fn test_split() {
        fn f(x: &int) -> bool { *x == 3 }

        assert_eq!(split([], f), ~[]);
        assert_eq!(split([1, 2], f), ~[~[1, 2]]);
        assert_eq!(split([3, 1, 2], f), ~[~[], ~[1, 2]]);
        assert_eq!(split([1, 2, 3], f), ~[~[1, 2], ~[]]);
        assert_eq!(split([1, 2, 3, 4, 3, 5], f), ~[~[1, 2], ~[4], ~[5]]);
    }

    #[test]
    fn test_splitn() {
        fn f(x: &int) -> bool { *x == 3 }

        assert_eq!(splitn([], 1u, f), ~[]);
        assert_eq!(splitn([1, 2], 1u, f), ~[~[1, 2]]);
        assert_eq!(splitn([3, 1, 2], 1u, f), ~[~[], ~[1, 2]]);
        assert_eq!(splitn([1, 2, 3], 1u, f), ~[~[1, 2], ~[]]);
        assert!(splitn([1, 2, 3, 4, 3, 5], 1u, f) ==
                      ~[~[1, 2], ~[4, 3, 5]]);
    }

    #[test]
    fn test_rsplit() {
        fn f(x: &int) -> bool { *x == 3 }

        assert_eq!(rsplit([], f), ~[]);
        assert_eq!(rsplit([1, 2], f), ~[~[1, 2]]);
        assert_eq!(rsplit([1, 2, 3], f), ~[~[1, 2], ~[]]);
        assert!(rsplit([1, 2, 3, 4, 3, 5], f) ==
            ~[~[1, 2], ~[4], ~[5]]);
    }

    #[test]
    fn test_rsplitn() {
        fn f(x: &int) -> bool { *x == 3 }

        assert_eq!(rsplitn([], 1u, f), ~[]);
        assert_eq!(rsplitn([1, 2], 1u, f), ~[~[1, 2]]);
        assert_eq!(rsplitn([1, 2, 3], 1u, f), ~[~[1, 2], ~[]]);
        assert_eq!(rsplitn([1, 2, 3, 4, 3, 5], 1u, f), ~[~[1, 2, 3, 4], ~[5]]);
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
    fn test_windowed () {
        fn t(n: uint, expected: &[&[int]]) {
            let mut i = 0;
            for windowed(n, [1,2,3,4,5,6]) |v| {
                assert_eq!(v, expected[i]);
                i += 1;
            }

            // check that we actually iterated the right number of times
            assert_eq!(i, expected.len());
        }
        t(3, &[&[1,2,3],&[2,3,4],&[3,4,5],&[4,5,6]]);
        t(4, &[&[1,2,3,4],&[2,3,4,5],&[3,4,5,6]]);
        t(7, &[]);
        t(8, &[]);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_windowed_() {
        for windowed (0u, [1u,2u,3u,4u,5u,6u]) |_v| {}
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
    fn test_split_fail_ret_true() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do split(v) |_elt| {
            if i == 2 {
                fail!()
            }
            i += 1;

            true
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    #[allow(non_implicitly_copyable_typarams)]
    fn test_split_fail_ret_false() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do split(v) |_elt| {
            if i == 2 {
                fail!()
            }
            i += 1;

            false
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    #[allow(non_implicitly_copyable_typarams)]
    fn test_splitn_fail_ret_true() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do splitn(v, 100) |_elt| {
            if i == 2 {
                fail!()
            }
            i += 1;

            true
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    #[allow(non_implicitly_copyable_typarams)]
    fn test_splitn_fail_ret_false() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do split(v) |_elt| {
            if i == 2 {
                fail!()
            }
            i += 1;

            false
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    #[allow(non_implicitly_copyable_typarams)]
    fn test_rsplit_fail_ret_true() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do rsplit(v) |_elt| {
            if i == 2 {
                fail!()
            }
            i += 1;

            true
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    #[allow(non_implicitly_copyable_typarams)]
    fn test_rsplit_fail_ret_false() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do rsplit(v) |_elt| {
            if i == 2 {
                fail!()
            }
            i += 1;

            false
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    #[allow(non_implicitly_copyable_typarams)]
    fn test_rsplitn_fail_ret_true() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do rsplitn(v, 100) |_elt| {
            if i == 2 {
                fail!()
            }
            i += 1;

            true
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    #[allow(non_implicitly_copyable_typarams)]
    fn test_rsplitn_fail_ret_false() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do rsplitn(v, 100) |_elt| {
            if i == 2 {
                fail!()
            }
            i += 1;

            false
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_consume_fail() {
        let v = ~[(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do consume(v) |_i, _elt| {
            if i == 2 {
                fail!()
            }
            i += 1;
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
        do map(v) |_elt| {
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
    fn test_map_consume_fail() {
        let v = ~[(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do map_consume(v) |_elt| {
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
    fn test_mapi_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do mapi(v) |_i, _elt| {
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
        do map(v) |_elt| {
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
    #[allow(non_implicitly_copyable_typarams)]
    fn test_map_zip_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do map_zip(v, v) |_elt1, _elt2| {
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
    #[allow(non_implicitly_copyable_typarams)]
    fn test_filter_mapped_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do filter_mapped(v) |_elt| {
            if i == 2 {
                fail!()
            }
            i += 0;
            Some((~0, @0))
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    #[allow(non_implicitly_copyable_typarams)]
    fn test_filter_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do v.filtered |_elt| {
            if i == 2 {
                fail!()
            }
            i += 0;
            true
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_rposition_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do rposition(v) |_elt| {
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
        do as_imm_buf(v) |_buf, _i| {
            fail!()
        }
    }

    #[test]
    #[ignore(cfg(windows))]
    #[should_fail]
    fn test_as_mut_buf_fail() {
        let mut v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        do as_mut_buf(v) |_buf, _i| {
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
        assert_eq!(it.size_hint(), (Some(5), Some(5)));
        assert_eq!(it.next().unwrap(), &1);
        assert_eq!(it.size_hint(), (Some(4), Some(4)));
        assert_eq!(it.next().unwrap(), &2);
        assert_eq!(it.size_hint(), (Some(3), Some(3)));
        assert_eq!(it.next().unwrap(), &5);
        assert_eq!(it.size_hint(), (Some(2), Some(2)));
        assert_eq!(it.next().unwrap(), &10);
        assert_eq!(it.size_hint(), (Some(1), Some(1)));
        assert_eq!(it.next().unwrap(), &11);
        assert_eq!(it.size_hint(), (Some(0), Some(0)));
        assert!(it.next().is_none());
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
        reverse(values.mut_slice(1, 4));
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
