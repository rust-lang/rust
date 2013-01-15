// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Vectors

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];
#[warn(non_camel_case_types)];

use cast;
use cmp::{Eq, Ord};
use iter::BaseIter;
use iter;
use kinds::Copy;
use libc;
use libc::size_t;
use option::{None, Option, Some};
use ptr;
use ptr::addr_of;
use sys;
use uint;
use vec;

#[abi = "cdecl"]
pub extern mod rustrt {
    unsafe fn vec_reserve_shared(++t: *sys::TypeDesc,
                                 ++v: **raw::VecRepr,
                                 ++n: libc::size_t);
}

#[abi = "rust-intrinsic"]
pub extern mod rusti {
    fn move_val_init<T>(dst: &mut T, -src: T);
    fn init<T>() -> T;
}


/// Returns true if a vector contains no elements
pub pure fn is_empty<T>(v: &[const T]) -> bool {
    as_const_buf(v, |_p, len| len == 0u)
}

/// Returns true if a vector contains some elements
pub pure fn is_not_empty<T>(v: &[const T]) -> bool {
    as_const_buf(v, |_p, len| len > 0u)
}

/// Returns true if two vectors have the same length
pub pure fn same_length<T, U>(xs: &[const T], ys: &[const U]) -> bool {
    len(xs) == len(ys)
}

/**
 * Reserves capacity for exactly `n` elements in the given vector.
 *
 * If the capacity for `v` is already equal to or greater than the requested
 * capacity, then no action is taken.
 *
 * # Arguments
 *
 * * v - A vector
 * * n - The number of elements to reserve space for
 */
pub fn reserve<T>(v: &mut ~[T], n: uint) {
    // Only make the (slow) call into the runtime if we have to
    if capacity(v) < n {
        unsafe {
            let ptr: **raw::VecRepr = cast::transmute(v);
            rustrt::vec_reserve_shared(sys::get_type_desc::<T>(),
                                       ptr, n as size_t);
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
 * If the capacity for `v` is already equal to or greater than the requested
 * capacity, then no action is taken.
 *
 * # Arguments
 *
 * * v - A vector
 * * n - The number of elements to reserve space for
 */
pub fn reserve_at_least<T>(v: &mut ~[T], n: uint) {
    reserve(v, uint::next_power_of_two(n));
}

/// Returns the number of elements the vector can hold without reallocating
#[inline(always)]
pub pure fn capacity<T>(v: &const ~[T]) -> uint {
    unsafe {
        let repr: **raw::VecRepr = ::cast::transmute(v);
        (**repr).unboxed.alloc / sys::nonzero_size_of::<T>()
    }
}

/// Returns the length of a vector
#[inline(always)]
pub pure fn len<T>(v: &[const T]) -> uint {
    as_const_buf(v, |_p, len| len)
}

/**
 * Creates and initializes an immutable vector.
 *
 * Creates an immutable vector of size `n_elts` and initializes the elements
 * to the value returned by the function `op`.
 */
pub pure fn from_fn<T>(n_elts: uint, op: iter::InitOp<T>) -> ~[T] {
    unsafe {
        let mut v = with_capacity(n_elts);
        do as_mut_buf(v) |p, _len| {
            let mut i: uint = 0u;
            while i < n_elts {
                rusti::move_val_init(&mut(*ptr::mut_offset(p, i)), op(i));
                i += 1u;
            }
        }
        raw::set_len(&mut v, n_elts);
        return v;
    }
}

/**
 * Creates and initializes an immutable vector.
 *
 * Creates an immutable vector of size `n_elts` and initializes the elements
 * to the value `t`.
 */
pub pure fn from_elem<T: Copy>(n_elts: uint, t: T) -> ~[T] {
    from_fn(n_elts, |_i| copy t)
}

/// Creates a new unique vector with the same contents as the slice
pub pure fn from_slice<T: Copy>(t: &[T]) -> ~[T] {
    from_fn(t.len(), |i| t[i])
}

pub pure fn with_capacity<T>(capacity: uint) -> ~[T] {
    let mut vec = ~[];
    unsafe { reserve(&mut vec, capacity); }
    return vec;
}

/**
 * Builds a vector by calling a provided function with an argument
 * function that pushes an element to the back of a vector.
 * This version takes an initial size for the vector.
 *
 * # Arguments
 *
 * * size - An initial size of the vector to reserve
 * * builder - A function that will construct the vector. It recieves
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline(always)]
pub pure fn build_sized<A>(size: uint,
                       builder: fn(push: pure fn(v: A))) -> ~[A] {
    let mut vec = with_capacity(size);
    builder(|x| unsafe { vec.push(x) });
    vec
}

/**
 * Builds a vector by calling a provided function with an argument
 * function that pushes an element to the back of a vector.
 *
 * # Arguments
 *
 * * builder - A function that will construct the vector. It recieves
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline(always)]
pub pure fn build<A>(builder: fn(push: pure fn(v: A))) -> ~[A] {
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
 * * builder - A function that will construct the vector. It recieves
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline(always)]
pub pure fn build_sized_opt<A>(size: Option<uint>,
                           builder: fn(push: pure fn(v: A))) -> ~[A] {
    build_sized(size.get_or_default(4), builder)
}

/// Produces a mut vector from an immutable vector.
pub pure fn to_mut<T>(v: ~[T]) -> ~[mut T] {
    unsafe { ::cast::transmute(v) }
}

/// Produces an immutable vector from a mut vector.
pub pure fn from_mut<T>(v: ~[mut T]) -> ~[T] {
    unsafe { ::cast::transmute(v) }
}

// Accessors

/// Returns the first element of a vector
pub pure fn head<T: Copy>(v: &[const T]) -> T { v[0] }

/// Returns a vector containing all but the first element of a slice
pub pure fn tail<T: Copy>(v: &[const T]) -> ~[T] {
    return slice(v, 1u, len(v));
}

/**
 * Returns a vector containing all but the first `n` \
 * elements of a slice
 */
pub pure fn tailn<T: Copy>(v: &[const T], n: uint) -> ~[T] {
    slice(v, n, len(v))
}

/// Returns a vector containing all but the last element of a slice
pub pure fn init<T: Copy>(v: &[const T]) -> ~[T] {
    assert len(v) != 0u;
    slice(v, 0u, len(v) - 1u)
}

/// Returns the last element of the slice `v`, failing if the slice is empty.
pub pure fn last<T: Copy>(v: &[const T]) -> T {
    if len(v) == 0u { fail ~"last_unsafe: empty vector" }
    v[len(v) - 1u]
}

/**
 * Returns `Some(x)` where `x` is the last element of the slice `v`,
 * or `none` if the vector is empty.
 */
pub pure fn last_opt<T: Copy>(v: &[const T]) -> Option<T> {
    if len(v) == 0u { return None; }
    Some(v[len(v) - 1u])
}

/// Returns a copy of the elements from [`start`..`end`) from `v`.
pub pure fn slice<T: Copy>(v: &[const T], start: uint, end: uint) -> ~[T] {
    assert (start <= end);
    assert (end <= len(v));
    let mut result = ~[];
    unsafe {
        for uint::range(start, end) |i| { result.push(v[i]) }
    }
    result
}

/// Return a slice that points into another slice.
pub pure fn view<T>(v: &r/[T], start: uint, end: uint) -> &r/[T] {
    assert (start <= end);
    assert (end <= len(v));
    do as_imm_buf(v) |p, _len| {
        unsafe {
            ::cast::reinterpret_cast(
                &(ptr::offset(p, start),
                  (end - start) * sys::nonzero_size_of::<T>()))
        }
    }
}

/// Return a slice that points into another slice.
pub pure fn mut_view<T>(v: &r/[mut T], start: uint, end: uint) -> &r/[mut T] {
    assert (start <= end);
    assert (end <= len(v));
    do as_mut_buf(v) |p, _len| {
        unsafe {
            ::cast::reinterpret_cast(
                &(ptr::mut_offset(p, start),
                  (end - start) * sys::nonzero_size_of::<T>()))
        }
    }
}

/// Return a slice that points into another slice.
pub pure fn const_view<T>(v: &r/[const T], start: uint,
                      end: uint) -> &r/[const T] {
    assert (start <= end);
    assert (end <= len(v));
    do as_const_buf(v) |p, _len| {
        unsafe {
            ::cast::reinterpret_cast(
                &(ptr::const_offset(p, start),
                  (end - start) * sys::nonzero_size_of::<T>()))
        }
    }
}

/// Split the vector `v` by applying each element against the predicate `f`.
pub fn split<T: Copy>(v: &[T], f: fn(t: &T) -> bool) -> ~[~[T]] {
    let ln = len(v);
    if (ln == 0u) { return ~[] }

    let mut start = 0u;
    let mut result = ~[];
    while start < ln {
        match position_between(v, start, ln, f) {
            None => break,
            Some(i) => {
                result.push(slice(v, start, i));
                start = i + 1u;
            }
        }
    }
    result.push(slice(v, start, ln));
    result
}

/**
 * Split the vector `v` by applying each element against the predicate `f` up
 * to `n` times.
 */
pub fn splitn<T: Copy>(v: &[T], n: uint, f: fn(t: &T) -> bool) -> ~[~[T]] {
    let ln = len(v);
    if (ln == 0u) { return ~[] }

    let mut start = 0u;
    let mut count = n;
    let mut result = ~[];
    while start < ln && count > 0u {
        match position_between(v, start, ln, f) {
            None => break,
            Some(i) => {
                result.push(slice(v, start, i));
                // Make sure to skip the separator.
                start = i + 1u;
                count -= 1u;
            }
        }
    }
    result.push(slice(v, start, ln));
    result
}

/**
 * Reverse split the vector `v` by applying each element against the predicate
 * `f`.
 */
pub fn rsplit<T: Copy>(v: &[T], f: fn(t: &T) -> bool) -> ~[~[T]] {
    let ln = len(v);
    if (ln == 0) { return ~[] }

    let mut end = ln;
    let mut result = ~[];
    while end > 0 {
        match rposition_between(v, 0, end, f) {
            None => break,
            Some(i) => {
                result.push(slice(v, i + 1, end));
                end = i;
            }
        }
    }
    result.push(slice(v, 0u, end));
    reverse(result);
    return result;
}

/**
 * Reverse split the vector `v` by applying each element against the predicate
 * `f` up to `n times.
 */
pub fn rsplitn<T: Copy>(v: &[T], n: uint, f: fn(t: &T) -> bool) -> ~[~[T]] {
    let ln = len(v);
    if (ln == 0u) { return ~[] }

    let mut end = ln;
    let mut count = n;
    let mut result = ~[];
    while end > 0u && count > 0u {
        match rposition_between(v, 0u, end, f) {
            None => break,
            Some(i) => {
                result.push(slice(v, i + 1u, end));
                // Make sure to skip the separator.
                end = i;
                count -= 1u;
            }
        }
    }
    result.push(slice(v, 0u, end));
    reverse(result);
    result
}

/**
 * Partitions a vector into two new vectors: those that satisfies the
 * predicate, and those that do not.
 */
pub fn partition<T>(v: ~[T], f: fn(&T) -> bool) -> (~[T], ~[T]) {
    let mut lefts  = ~[];
    let mut rights = ~[];

    // FIXME (#4355 maybe): using v.consume here crashes
    // do v.consume |_, elt| {
    do consume(v) |_, elt| {
        if f(&elt) {
            lefts.push(elt);
        } else {
            rights.push(elt);
        }
    }

    (lefts, rights)
}

/**
 * Partitions a vector into two new vectors: those that satisfies the
 * predicate, and those that do not.
 */
pub pure fn partitioned<T: Copy>(v: &[T], f: fn(&T) -> bool) -> (~[T], ~[T]) {
    let mut lefts  = ~[];
    let mut rights = ~[];

    for each(v) |elt| {
        unsafe {
            if f(elt) {
                lefts.push(*elt);
            } else {
                rights.push(*elt);
            }
        }
    }

    (lefts, rights)
}

// Mutators

/// Removes the first element from a vector and return it
pub fn shift<T>(v: &mut ~[T]) -> T unsafe {

    assert v.is_not_empty();

    if v.len() == 1 { return v.pop() }

    if v.len() == 2 {
        let last = v.pop();
        let first = v.pop();
        v.push(last);
        return first;
    }

    let ln = v.len();
    let next_ln = v.len() - 1;

    // Save the last element. We're going to overwrite its position
    let mut work_elt = v.pop();
    // We still should have room to work where what last element was
    assert capacity(v) >= ln;
    // Pretend like we have the original length so we can use
    // the vector copy_memory to overwrite the hole we just made
    raw::set_len(&mut *v, ln);

    // Memcopy the head element (the one we want) to the location we just
    // popped. For the moment it unsafely exists at both the head and last
    // positions
    let first_slice = view(*v, 0, 1);
    let last_slice = mut_view(*v, next_ln, ln);
    raw::copy_memory(last_slice, first_slice, 1);

    // Memcopy everything to the left one element
    let init_slice = mut_view(*v, 0, next_ln);
    let tail_slice = view(*v, 1, ln);
    raw::copy_memory(init_slice, tail_slice, next_ln);

    // Set the new length. Now the vector is back to normal
    raw::set_len(&mut *v, next_ln);

    // Swap out the element we want from the end
    let vp = raw::to_mut_ptr(*v);
    let vp = ptr::mut_offset(vp, next_ln - 1);
    *vp <-> work_elt;

    return work_elt;
}

/// Prepend an element to the vector
pub fn unshift<T>(v: &mut ~[T], x: T) {
    let mut vv = ~[x];
    *v <-> vv;
    v.push_all_move(vv);
}

/// Insert an element at position i within v, shifting all
/// elements after position i one position to the right.
pub fn insert<T>(v: &mut ~[T], i: uint, x: T) {
    let len = v.len();
    assert i <= len;

    v.push(x);
    let mut j = len;
    while j > i {
        v[j] <-> v[j - 1];
        j -= 1;
    }
}

/// Remove and return the element at position i within v, shifting
/// all elements after position i one position to the left.
pub fn remove<T>(v: &mut ~[T], i: uint) -> T {
    let len = v.len();
    assert i < len;

    let mut j = i;
    while j < len - 1 {
        v[j] <-> v[j + 1];
        j += 1;
    }
    v.pop()
}

pub fn consume<T>(v: ~[T], f: fn(uint, v: T)) unsafe {
    let mut v = v; // FIXME(#3488)

    do as_mut_buf(v) |p, ln| {
        for uint::range(0, ln) |i| {
            // NB: This unsafe operation counts on init writing 0s to the
            // holes we create in the vector. That ensures that, if the
            // iterator fails then we won't try to clean up the consumed
            // elements during unwinding
            let mut x = rusti::init();
            let p = ptr::mut_offset(p, i);
            x <-> *p;
            f(i, x);
        }
    }

    raw::set_len(&mut v, 0);
}

pub fn consume_mut<T>(v: ~[mut T], f: fn(uint, v: T)) {
    consume(vec::from_mut(v), f)
}

/// Remove the last element from a vector and return it
pub fn pop<T>(v: &mut ~[T]) -> T {
    let ln = v.len();
    if ln == 0 {
        fail ~"sorry, cannot vec::pop an empty vector"
    }
    let valptr = ptr::to_mut_unsafe_ptr(&mut v[ln - 1u]);
    unsafe {
        // FIXME #4204: Should be rusti::uninit() - we don't need this zeroed
        let mut val = rusti::init();
        val <-> *valptr;
        raw::set_len(v, ln - 1u);
        val
    }
}

/**
 * Remove an element from anywhere in the vector and return it, replacing it
 * with the last element. This does not preserve ordering, but is O(1).
 *
 * Fails if index >= length.
 */
pub fn swap_remove<T>(v: &mut ~[T], index: uint) -> T {
    let ln = v.len();
    if index >= ln {
        fail fmt!("vec::swap_remove - index %u >= length %u", index, ln);
    }
    if index < ln - 1 {
        v[index] <-> v[ln - 1];
    }
    vec::pop(v)
}

/// Append an element to a vector
#[inline(always)]
pub fn push<T>(v: &mut ~[T], initval: T) {
    unsafe {
        let repr: **raw::VecRepr = ::cast::transmute(&mut *v);
        let fill = (**repr).unboxed.fill;
        if (**repr).unboxed.alloc > fill {
            push_fast(v, initval);
        }
        else {
            push_slow(v, initval);
        }
    }
}

// This doesn't bother to make sure we have space.
#[inline(always)] // really pretty please
unsafe fn push_fast<T>(v: &mut ~[T], initval: T) {
    let repr: **raw::VecRepr = ::cast::transmute(v);
    let fill = (**repr).unboxed.fill;
    (**repr).unboxed.fill += sys::nonzero_size_of::<T>();
    let p = addr_of(&((**repr).unboxed.data));
    let p = ptr::offset(p, fill) as *mut T;
    rusti::move_val_init(&mut(*p), move initval);
}

#[inline(never)]
fn push_slow<T>(v: &mut ~[T], initval: T) {
    reserve_at_least(&mut *v, v.len() + 1u);
    unsafe { push_fast(v, initval) }
}

#[inline(always)]
pub fn push_all<T: Copy>(v: &mut ~[T], rhs: &[const T]) {
    reserve(&mut *v, v.len() + rhs.len());

    for uint::range(0u, rhs.len()) |i| {
        push(&mut *v, unsafe { raw::get(rhs, i) })
    }
}

#[inline(always)]
pub fn push_all_move<T>(v: &mut ~[T], rhs: ~[T]) {
    let mut rhs = rhs; // FIXME(#3488)
    reserve(&mut *v, v.len() + rhs.len());
    unsafe {
        do as_mut_buf(rhs) |p, len| {
            for uint::range(0, len) |i| {
                // FIXME #4204 Should be rusti::uninit() - don't need to zero
                let mut x = rusti::init();
                x <-> *ptr::mut_offset(p, i);
                push(&mut *v, x);
            }
        }
        raw::set_len(&mut rhs, 0);
    }
}

/// Shorten a vector, dropping excess elements.
pub fn truncate<T>(v: &mut ~[T], newlen: uint) {
    do as_mut_buf(*v) |p, oldlen| {
        assert(newlen <= oldlen);
        unsafe {
            // This loop is optimized out for non-drop types.
            for uint::range(newlen, oldlen) |i| {
                // FIXME #4204 Should be rusti::uninit() - don't need to zero
                let mut dropped = rusti::init();
                dropped <-> *ptr::mut_offset(p, i);
            }
            raw::set_len(&mut *v, newlen);
        }
    }
}

/**
 * Remove consecutive repeated elements from a vector; if the vector is
 * sorted, this removes all duplicates.
 */
pub fn dedup<T: Eq>(v: &mut ~[T]) unsafe {
    if v.len() < 1 { return; }
    let mut last_written = 0, next_to_read = 1;
    do as_const_buf(*v) |p, ln| {
        // We have a mutable reference to v, so we can make arbitrary changes.
        // (cf. push and pop)
        let p = p as *mut T;
        // last_written < next_to_read <= ln
        while next_to_read < ln {
            // last_written < next_to_read < ln
            if *ptr::mut_offset(p, next_to_read) ==
                *ptr::mut_offset(p, last_written) {
                // FIXME #4204 Should be rusti::uninit() - don't need to zero
                let mut dropped = rusti::init();
                dropped <-> *ptr::mut_offset(p, next_to_read);
            } else {
                last_written += 1;
                // last_written <= next_to_read < ln
                if next_to_read != last_written {
                    *ptr::mut_offset(p, last_written) <->
                        *ptr::mut_offset(p, next_to_read);
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


// Appending
#[inline(always)]
pub pure fn append<T: Copy>(lhs: ~[T], rhs: &[const T]) -> ~[T] {
    let mut v = lhs;
    unsafe {
        v.push_all(rhs);
    }
    v
}

#[inline(always)]
pub pure fn append_one<T>(lhs: ~[T], x: T) -> ~[T] {
    let mut v = lhs;
    unsafe { v.push(x); }
    v
}

#[inline(always)]
pub pure fn append_mut<T: Copy>(lhs: ~[mut T], rhs: &[const T]) -> ~[mut T] {
    to_mut(append(from_mut(lhs), rhs))
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
pub fn grow<T: Copy>(v: &mut ~[T], n: uint, initval: &T) {
    reserve_at_least(&mut *v, v.len() + n);
    let mut i: uint = 0u;

    while i < n {
        v.push(*initval);
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
pub fn grow_fn<T>(v: &mut ~[T], n: uint, op: iter::InitOp<T>) {
    reserve_at_least(&mut *v, v.len() + n);
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
pub fn grow_set<T: Copy>(v: &mut ~[T], index: uint, initval: &T, val: T) {
    let l = v.len();
    if index >= l { grow(&mut *v, index - l + 1u, initval); }
    v[index] = val;
}

// Functional utilities

/// Apply a function to each element of a vector and return the results
pub pure fn map<T, U>(v: &[T], f: fn(t: &T) -> U) -> ~[U] {
    let mut result = with_capacity(len(v));
    for each(v) |elem| {
        unsafe {
            result.push(f(elem));
        }
    }
    result
}

pub fn map_consume<T, U>(v: ~[T], f: fn(v: T) -> U) -> ~[U] {
    let mut result = ~[];
    do consume(v) |_i, x| {
        result.push(f(x));
    }
    result
}

/// Apply a function to each element of a vector and return the results
pub pure fn mapi<T, U>(v: &[T], f: fn(uint, t: &T) -> U) -> ~[U] {
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
pub pure fn flat_map<T, U>(v: &[T], f: fn(t: &T) -> ~[U]) -> ~[U] {
    let mut result = ~[];
    for each(v) |elem| { unsafe{ result.push_all_move(f(elem)); } }
    result
}

/// Apply a function to each pair of elements and return the results
pub pure fn map2<T: Copy, U: Copy, V>(v0: &[T], v1: &[U],
                                  f: fn(t: &T, v: &U) -> V) -> ~[V] {
    let v0_len = len(v0);
    if v0_len != len(v1) { fail; }
    let mut u: ~[V] = ~[];
    let mut i = 0u;
    while i < v0_len {
        unsafe { u.push(f(&v0[i], &v1[i])) };
        i += 1u;
    }
    u
}

/**
 * Apply a function to each element of a vector and return the results
 *
 * If function `f` returns `none` then that element is excluded from
 * the resulting vector.
 */
pub pure fn filter_map<T, U: Copy>(v: &[T], f: fn(t: &T) -> Option<U>)
    -> ~[U] {
    let mut result = ~[];
    for each(v) |elem| {
        match f(elem) {
          None => {/* no-op */ }
          Some(result_elem) => unsafe { result.push(result_elem); }
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
pub fn filter<T>(v: ~[T], f: fn(t: &T) -> bool) -> ~[T] {
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
pub pure fn filtered<T: Copy>(v: &[T], f: fn(t: &T) -> bool) -> ~[T] {
    let mut result = ~[];
    for each(v) |elem| {
        if f(elem) { unsafe { result.push(*elem); } }
    }
    result
}

/**
 * Like `filter()`, but in place.  Preserves order of `v`.  Linear time.
 */
pub fn retain<T>(v: &mut ~[T], f: pure fn(t: &T) -> bool) {
    let len = v.len();
    let mut deleted: uint = 0;

    for uint::range(0, len) |i| {
        if !f(&v[i]) {
            deleted += 1;
        } else if deleted > 0 {
            v[i - deleted] <-> v[i];
        }
    }

    if deleted > 0 {
        v.truncate(len - deleted);
    }
}

/**
 * Concatenate a vector of vectors.
 *
 * Flattens a vector of vectors of T into a single vector of T.
 */
pub pure fn concat<T: Copy>(v: &[~[T]]) -> ~[T] {
    let mut r = ~[];
    for each(v) |inner| { unsafe { r.push_all(*inner); } }
    r
}

/// Concatenate a vector of vectors, placing a given separator between each
pub pure fn connect<T: Copy>(v: &[~[T]], sep: &T) -> ~[T] {
    let mut r: ~[T] = ~[];
    let mut first = true;
    for each(v) |inner| {
        if first { first = false; } else { unsafe { r.push(*sep); } }
        unsafe { r.push_all(*inner) };
    }
    r
}

/// Reduce a vector from left to right
pub pure fn foldl<T, U>(z: T, v: &[U], p: fn(t: T, u: &U) -> T) -> T {
    let mut accum = z;
    let mut i = 0;
    let l = v.len();
    while i < l {
        // Use a while loop so that liveness analysis can handle moving
        // the accumulator.
        accum = p(accum, &v[i]);
        i += 1;
    }
    return accum;
}

/// Reduce a vector from right to left
pub pure fn foldr<T, U: Copy>(v: &[T], z: U, p: fn(t: &T, u: U) -> U) -> U {
    let mut accum = z;
    for rev_each(v) |elt| {
        accum = p(elt, accum);
    }
    return accum;
}

/**
 * Return true if a predicate matches any elements
 *
 * If the vector contains no elements then false is returned.
 */
pub pure fn any<T>(v: &[T], f: fn(t: &T) -> bool) -> bool {
    for each(v) |elem| { if f(elem) { return true; } }
    return false;
}

/**
 * Return true if a predicate matches any elements in both vectors.
 *
 * If the vectors contains no elements then false is returned.
 */
pub pure fn any2<T, U>(v0: &[T], v1: &[U],
                   f: fn(a: &T, b: &U) -> bool) -> bool {
    let v0_len = len(v0);
    let v1_len = len(v1);
    let mut i = 0u;
    while i < v0_len && i < v1_len {
        if f(&v0[i], &v1[i]) { return true; };
        i += 1u;
    }
    return false;
}

/**
 * Return true if a predicate matches all elements
 *
 * If the vector contains no elements then true is returned.
 */
pub pure fn all<T>(v: &[T], f: fn(t: &T) -> bool) -> bool {
    for each(v) |elem| { if !f(elem) { return false; } }
    return true;
}

/**
 * Return true if a predicate matches all elements
 *
 * If the vector contains no elements then true is returned.
 */
pub pure fn alli<T>(v: &[T], f: fn(uint, t: &T) -> bool) -> bool {
    for eachi(v) |i, elem| { if !f(i, elem) { return false; } }
    return true;
}

/**
 * Return true if a predicate matches all elements in both vectors.
 *
 * If the vectors are not the same size then false is returned.
 */
pub pure fn all2<T, U>(v0: &[T], v1: &[U],
                   f: fn(t: &T, u: &U) -> bool) -> bool {
    let v0_len = len(v0);
    if v0_len != len(v1) { return false; }
    let mut i = 0u;
    while i < v0_len { if !f(&v0[i], &v1[i]) { return false; }; i += 1u; }
    return true;
}

/// Return true if a vector contains an element with the given value
pub pure fn contains<T: Eq>(v: &[T], x: &T) -> bool {
    for each(v) |elt| { if *x == *elt { return true; } }
    return false;
}

/// Returns the number of elements that are equal to a given value
pub pure fn count<T: Eq>(v: &[T], x: &T) -> uint {
    let mut cnt = 0u;
    for each(v) |elt| { if *x == *elt { cnt += 1u; } }
    return cnt;
}

/**
 * Search for the first element that matches a given predicate
 *
 * Apply function `f` to each element of `v`, starting from the first.
 * When function `f` returns true then an option containing the element
 * is returned. If `f` matches no elements then none is returned.
 */
pub pure fn find<T: Copy>(v: &[T], f: fn(t: &T) -> bool) -> Option<T> {
    find_between(v, 0u, len(v), f)
}

/**
 * Search for the first element that matches a given predicate within a range
 *
 * Apply function `f` to each element of `v` within the range
 * [`start`, `end`). When function `f` returns true then an option containing
 * the element is returned. If `f` matches no elements then none is returned.
 */
pub pure fn find_between<T: Copy>(v: &[T], start: uint, end: uint,
                      f: fn(t: &T) -> bool) -> Option<T> {
    position_between(v, start, end, f).map(|i| v[*i])
}

/**
 * Search for the last element that matches a given predicate
 *
 * Apply function `f` to each element of `v` in reverse order. When function
 * `f` returns true then an option containing the element is returned. If `f`
 * matches no elements then none is returned.
 */
pub pure fn rfind<T: Copy>(v: &[T], f: fn(t: &T) -> bool) -> Option<T> {
    rfind_between(v, 0u, len(v), f)
}

/**
 * Search for the last element that matches a given predicate within a range
 *
 * Apply function `f` to each element of `v` in reverse order within the range
 * [`start`, `end`). When function `f` returns true then an option containing
 * the element is returned. If `f` matches no elements then none is return.
 */
pub pure fn rfind_between<T: Copy>(v: &[T], start: uint, end: uint,
                               f: fn(t: &T) -> bool) -> Option<T> {
    rposition_between(v, start, end, f).map(|i| v[*i])
}

/// Find the first index containing a matching value
pub pure fn position_elem<T: Eq>(v: &[T], x: &T) -> Option<uint> {
    position(v, |y| *x == *y)
}

/**
 * Find the first index matching some predicate
 *
 * Apply function `f` to each element of `v`.  When function `f` returns true
 * then an option containing the index is returned. If `f` matches no elements
 * then none is returned.
 */
pub pure fn position<T>(v: &[T], f: fn(t: &T) -> bool) -> Option<uint> {
    position_between(v, 0u, len(v), f)
}

/**
 * Find the first index matching some predicate within a range
 *
 * Apply function `f` to each element of `v` between the range
 * [`start`, `end`). When function `f` returns true then an option containing
 * the index is returned. If `f` matches no elements then none is returned.
 */
pub pure fn position_between<T>(v: &[T], start: uint, end: uint,
                            f: fn(t: &T) -> bool) -> Option<uint> {
    assert start <= end;
    assert end <= len(v);
    let mut i = start;
    while i < end { if f(&v[i]) { return Some::<uint>(i); } i += 1u; }
    return None;
}

/// Find the last index containing a matching value
pure fn rposition_elem<T: Eq>(v: &[T], x: &T) -> Option<uint> {
    rposition(v, |y| *x == *y)
}

/**
 * Find the last index matching some predicate
 *
 * Apply function `f` to each element of `v` in reverse order.  When function
 * `f` returns true then an option containing the index is returned. If `f`
 * matches no elements then none is returned.
 */
pub pure fn rposition<T>(v: &[T], f: fn(t: &T) -> bool) -> Option<uint> {
    rposition_between(v, 0u, len(v), f)
}

/**
 * Find the last index matching some predicate within a range
 *
 * Apply function `f` to each element of `v` in reverse order between the
 * range [`start`, `end`). When function `f` returns true then an option
 * containing the index is returned. If `f` matches no elements then none is
 * returned.
 */
pub pure fn rposition_between<T>(v: &[T], start: uint, end: uint,
                             f: fn(t: &T) -> bool) -> Option<uint> {
    assert start <= end;
    assert end <= len(v);
    let mut i = end;
    while i > start {
        if f(&v[i - 1u]) { return Some::<uint>(i - 1u); }
        i -= 1u;
    }
    return None;
}

// FIXME: if issue #586 gets implemented, could have a postcondition
// saying the two result lists have the same length -- or, could
// return a nominal record with a constraint saying that, instead of
// returning a tuple (contingent on issue #869)
/**
 * Convert a vector of pairs into a pair of vectors, by reference. As unzip().
 */
pure fn unzip_slice<T: Copy, U: Copy>(v: &[(T, U)]) -> (~[T], ~[U]) {
    let mut ts = ~[], us = ~[];
    for each(v) |p| {
        let (t, u) = *p;
        unsafe {
            ts.push(t);
            us.push(u);
        }
    }
    return (ts, us);
}

/**
 * Convert a vector of pairs into a pair of vectors.
 *
 * Returns a tuple containing two vectors where the i-th element of the first
 * vector contains the first element of the i-th tuple of the input vector,
 * and the i-th element of the second vector contains the second element
 * of the i-th tuple of the input vector.
 */
pub pure fn unzip<T,U>(v: ~[(T, U)]) -> (~[T], ~[U]) {
    let mut ts = ~[], us = ~[];
    unsafe {
        do consume(v) |_i, p| {
            let (t, u) = p;
            ts.push(t);
            us.push(u);
        }
    }
    (ts, us)
}

/**
 * Convert two vectors to a vector of pairs, by reference. As zip().
 */
pub pure fn zip_slice<T: Copy, U: Copy>(v: &[const T], u: &[const U])
        -> ~[(T, U)] {
    let mut zipped = ~[];
    let sz = len(v);
    let mut i = 0u;
    assert sz == len(u);
    while i < sz unsafe { zipped.push((v[i], u[i])); i += 1u; }
    zipped
}

/**
 * Convert two vectors to a vector of pairs.
 *
 * Returns a vector of tuples, where the i-th tuple contains contains the
 * i-th elements from each of the input vectors.
 */
pub pure fn zip<T, U>(v: ~[T], u: ~[U]) -> ~[(T, U)] {
    let mut v = v, u = u; // FIXME(#3488)
    let mut i = len(v);
    assert i == len(u);
    let mut w = with_capacity(i);
    while i > 0 {
        unsafe { w.push((v.pop(),u.pop())); }
        i -= 1;
    }
    unsafe { reverse(w); }
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
pub fn swap<T>(v: &[mut T], a: uint, b: uint) {
    v[a] <-> v[b];
}

/// Reverse the order of elements in a vector, in place
pub fn reverse<T>(v: &[mut T]) {
    let mut i: uint = 0;
    let ln = len::<T>(v);
    while i < ln / 2 { v[i] <-> v[ln - i - 1]; i += 1; }
}

/// Returns a vector with the order of elements reversed
pub pure fn reversed<T: Copy>(v: &[const T]) -> ~[T] {
    let mut rs: ~[T] = ~[];
    let mut i = len::<T>(v);
    if i == 0 { return (rs); } else { i -= 1; }
    unsafe {
        while i != 0 { rs.push(v[i]); i -= 1; }
        rs.push(v[0]);
    }
    rs
}

/**
 * Iterates over a vector, yielding each element to a closure.
 *
 * # Arguments
 *
 * * `v` - A vector, to be iterated over
 * * `f` - A closure to do the iterating. Within this closure, return true to
 * * continue iterating, false to break.
 *
 * # Examples
 * ~~~
 * [1,2,3].each(|&i| {
 *     io::println(int::str(i));
 *     true
 * });
 * ~~~
 *
 * ~~~
 * [1,2,3,4,5].each(|&i| {
 *     if i < 4 {
 *         io::println(int::str(i));
 *         true
 *     }
 *     else {
 *         false
 *     }
 * });
 * ~~~
 *
 * You probably will want to use each with a `for`/`do` expression, depending
 * on your iteration needs:
 *
 * ~~~
 * for [1,2,3].each |&i| {
 *     io::println(int::str(i));
 * }
 * ~~~
 */
#[inline(always)]
pub pure fn each<T>(v: &r/[T], f: fn(&r/T) -> bool) {
    //             ^^^^
    // NB---this CANNOT be &[const T]!  The reason
    // is that you are passing it to `f()` using
    // an immutable.

    do vec::as_imm_buf(v) |p, n| {
        let mut n = n;
        let mut p = p;
        while n > 0u {
            unsafe {
                let q = cast::copy_lifetime_vec(v, &*p);
                if !f(q) { break; }
                p = ptr::offset(p, 1u);
            }
            n -= 1u;
        }
    }
}

/// Like `each()`, but for the case where you have
/// a vector with mutable contents and you would like
/// to mutate the contents as you iterate.
#[inline(always)]
pub fn each_mut<T>(v: &[mut T], f: fn(elem: &mut T) -> bool) {
    let mut i = 0;
    let n = v.len();
    while i < n {
        if !f(&mut v[i]) {
            return;
        }
        i += 1;
    }
}

/// Like `each()`, but for the case where you have a vector that *may or may
/// not* have mutable contents.
#[inline(always)]
pub pure fn each_const<T>(v: &[const T], f: fn(elem: &const T) -> bool) {
    let mut i = 0;
    let n = v.len();
    while i < n {
        if !f(&const v[i]) {
            return;
        }
        i += 1;
    }
}

/**
 * Iterates over a vector's elements and indices
 *
 * Return true to continue, false to break.
 */
#[inline(always)]
pub pure fn eachi<T>(v: &r/[T], f: fn(uint, v: &r/T) -> bool) {
    let mut i = 0;
    for each(v) |p| {
        if !f(i, p) { return; }
        i += 1;
    }
}

/**
 * Iterates over a vector's elements in reverse
 *
 * Return true to continue, false to break.
 */
#[inline(always)]
pub pure fn rev_each<T>(v: &r/[T], blk: fn(v: &r/T) -> bool) {
    rev_eachi(v, |_i, v| blk(v))
}

/**
 * Iterates over a vector's elements and indices in reverse
 *
 * Return true to continue, false to break.
 */
#[inline(always)]
pub pure fn rev_eachi<T>(v: &r/[T], blk: fn(i: uint, v: &r/T) -> bool) {
    let mut i = v.len();
    while i > 0 {
        i -= 1;
        if !blk(i, &v[i]) {
            return;
        }
    }
}

/**
 * Iterates over two vectors simultaneously
 *
 * # Failure
 *
 * Both vectors must have the same length
 */
#[inline]
pub fn each2<U, T>(v1: &[U], v2: &[T], f: fn(u: &U, t: &T) -> bool) {
    assert len(v1) == len(v2);
    for uint::range(0u, len(v1)) |i| {
        if !f(&v1[i], &v2[i]) {
            return;
        }
    }
}

/**
 * Iterate over all permutations of vector `v`.
 *
 * Permutations are produced in lexicographic order with respect to the order
 * of elements in `v` (so if `v` is sorted then the permutations are
 * lexicographically sorted).
 *
 * The total number of permutations produced is `len(v)!`.  If `v` contains
 * repeated elements, then some permutations are repeated.
 */
pure fn each_permutation<T: Copy>(v: &[T], put: fn(ts: &[T]) -> bool) {
    let ln = len(v);
    if ln <= 1 {
        put(v);
    } else {
        // This does not seem like the most efficient implementation.  You
        // could make far fewer copies if you put your mind to it.
        let mut i = 0u;
        while i < ln {
            let elt = v[i];
            let mut rest = slice(v, 0u, i);
            unsafe {
                rest.push_all(const_view(v, i+1u, ln));
                for each_permutation(rest) |permutation| {
                    if !put(append(~[elt], permutation)) {
                        return;
                    }
                }
            }
            i += 1u;
        }
    }
}

pub pure fn windowed<TT: Copy>(nn: uint, xx: &[TT]) -> ~[~[TT]] {
    let mut ww = ~[];
    assert 1u <= nn;
    for vec::eachi (xx) |ii, _x| {
        let len = vec::len(xx);
        if ii+nn <= len unsafe {
            ww.push(vec::slice(xx, ii, ii+nn));
        }
    }
    ww
}

/**
 * Work with the buffer of a vector.
 *
 * Allows for unsafe manipulation of vector contents, which is useful for
 * foreign interop.
 */
#[inline(always)]
pub pure fn as_imm_buf<T,U>(s: &[T],
                            /* NB---this CANNOT be const, see below */
                            f: fn(*T, uint) -> U) -> U {

    // NB---Do not change the type of s to `&[const T]`.  This is
    // unsound.  The reason is that we are going to create immutable pointers
    // into `s` and pass them to `f()`, but in fact they are potentially
    // pointing at *mutable memory*.  Use `as_const_buf` or `as_mut_buf`
    // instead!

    unsafe {
        let v : *(*T,uint) =
            ::cast::reinterpret_cast(&addr_of(&s));
        let (buf,len) = *v;
        f(buf, len / sys::nonzero_size_of::<T>())
    }
}

/// Similar to `as_imm_buf` but passing a `*const T`
#[inline(always)]
pub pure fn as_const_buf<T,U>(s: &[const T],
                          f: fn(*const T, uint) -> U) -> U {

    unsafe {
        let v : *(*const T,uint) =
            ::cast::reinterpret_cast(&addr_of(&s));
        let (buf,len) = *v;
        f(buf, len / sys::nonzero_size_of::<T>())
    }
}

/// Similar to `as_imm_buf` but passing a `*mut T`
#[inline(always)]
pub pure fn as_mut_buf<T,U>(s: &[mut T],
                        f: fn(*mut T, uint) -> U) -> U {

    unsafe {
        let v : *(*mut T,uint) =
            ::cast::reinterpret_cast(&addr_of(&s));
        let (buf,len) = *v;
        f(buf, len / sys::nonzero_size_of::<T>())
    }
}

// Equality

pure fn eq<T: Eq>(a: &[T], b: &[T]) -> bool {
    let (a_len, b_len) = (a.len(), b.len());
    if a_len != b_len { return false; }

    let mut i = 0;
    while i < a_len {
        if a[i] != b[i] { return false; }
        i += 1;
    }

    return true;
}

#[cfg(notest)]
impl<T: Eq> &[T] : Eq {
    #[inline(always)]
    pure fn eq(&self, other: & &self/[T]) -> bool { eq((*self), (*other)) }
    #[inline(always)]
    pure fn ne(&self, other: & &self/[T]) -> bool { !(*self).eq(other) }
}


#[cfg(notest)]
impl<T: Eq> ~[T] : Eq {
    #[inline(always)]
    pure fn eq(&self, other: &~[T]) -> bool { eq((*self), (*other)) }
    #[inline(always)]
    pure fn ne(&self, other: &~[T]) -> bool { !(*self).eq(other) }
}

#[cfg(notest)]
impl<T: Eq> @[T] : Eq {
    #[inline(always)]
    pure fn eq(&self, other: &@[T]) -> bool { eq((*self), (*other)) }
    #[inline(always)]
    pure fn ne(&self, other: &@[T]) -> bool { !(*self).eq(other) }
}

// Lexicographical comparison

pure fn lt<T: Ord>(a: &[T], b: &[T]) -> bool {
    let (a_len, b_len) = (a.len(), b.len());
    let mut end = uint::min(a_len, b_len);

    let mut i = 0;
    while i < end {
        let (c_a, c_b) = (&a[i], &b[i]);
        if *c_a < *c_b { return true; }
        if *c_a > *c_b { return false; }
        i += 1;
    }

    return a_len < b_len;
}

pure fn le<T: Ord>(a: &[T], b: &[T]) -> bool { !lt(b, a) }
pure fn ge<T: Ord>(a: &[T], b: &[T]) -> bool { !lt(a, b) }
pure fn gt<T: Ord>(a: &[T], b: &[T]) -> bool { lt(b, a)  }

#[cfg(notest)]
impl<T: Ord> &[T] : Ord {
    #[inline(always)]
    pure fn lt(&self, other: & &self/[T]) -> bool { lt((*self), (*other)) }
    #[inline(always)]
    pure fn le(&self, other: & &self/[T]) -> bool { le((*self), (*other)) }
    #[inline(always)]
    pure fn ge(&self, other: & &self/[T]) -> bool { ge((*self), (*other)) }
    #[inline(always)]
    pure fn gt(&self, other: & &self/[T]) -> bool { gt((*self), (*other)) }
}

#[cfg(notest)]
impl<T: Ord> ~[T] : Ord {
    #[inline(always)]
    pure fn lt(&self, other: &~[T]) -> bool { lt((*self), (*other)) }
    #[inline(always)]
    pure fn le(&self, other: &~[T]) -> bool { le((*self), (*other)) }
    #[inline(always)]
    pure fn ge(&self, other: &~[T]) -> bool { ge((*self), (*other)) }
    #[inline(always)]
    pure fn gt(&self, other: &~[T]) -> bool { gt((*self), (*other)) }
}

#[cfg(notest)]
impl<T: Ord> @[T] : Ord {
    #[inline(always)]
    pure fn lt(&self, other: &@[T]) -> bool { lt((*self), (*other)) }
    #[inline(always)]
    pure fn le(&self, other: &@[T]) -> bool { le((*self), (*other)) }
    #[inline(always)]
    pure fn ge(&self, other: &@[T]) -> bool { ge((*self), (*other)) }
    #[inline(always)]
    pure fn gt(&self, other: &@[T]) -> bool { gt((*self), (*other)) }
}

#[cfg(notest)]
pub mod traits {
    use kinds::Copy;
    use ops::Add;
    use vec::{append, append_mut};

    impl<T: Copy> ~[T] : Add<&[const T],~[T]> {
        #[inline(always)]
        pure fn add(&self, rhs: & &self/[const T]) -> ~[T] {
            append(copy *self, (*rhs))
        }
    }

    impl<T: Copy> ~[mut T] : Add<&[const T],~[mut T]> {
        #[inline(always)]
        pure fn add(&self, rhs: & &self/[const T]) -> ~[mut T] {
            append_mut(copy *self, (*rhs))
        }
    }
}

pub trait ConstVector {
    pure fn is_empty(&self) -> bool;
    pure fn is_not_empty(&self) -> bool;
    pure fn len(&self) -> uint;
}

/// Extension methods for vectors
impl<T> &[const T]: ConstVector {
    /// Returns true if a vector contains no elements
    #[inline]
    pure fn is_empty(&self) -> bool { is_empty(*self) }
    /// Returns true if a vector contains some elements
    #[inline]
    pure fn is_not_empty(&self) -> bool { is_not_empty(*self) }
    /// Returns the length of a vector
    #[inline]
    pure fn len(&self) -> uint { len(*self) }
}

pub trait CopyableVector<T> {
    pure fn head(&self) -> T;
    pure fn init(&self) -> ~[T];
    pure fn last(&self) -> T;
    pure fn slice(&self, start: uint, end: uint) -> ~[T];
    pure fn tail(&self) -> ~[T];
}

/// Extension methods for vectors
impl<T: Copy> &[const T]: CopyableVector<T> {
    /// Returns the first element of a vector
    #[inline]
    pure fn head(&self) -> T { head(*self) }

    /// Returns all but the last elemnt of a vector
    #[inline]
    pure fn init(&self) -> ~[T] { init(*self) }

    /// Returns the last element of a `v`, failing if the vector is empty.
    #[inline]
    pure fn last(&self) -> T { last(*self) }

    /// Returns a copy of the elements from [`start`..`end`) from `v`.
    #[inline]
    pure fn slice(&self, start: uint, end: uint) -> ~[T] {
        slice(*self, start, end)
    }

    /// Returns all but the first element of a vector
    #[inline]
    pure fn tail(&self) -> ~[T] { tail(*self) }
}

pub trait ImmutableVector<T> {
    pure fn view(&self, start: uint, end: uint) -> &self/[T];
    pure fn foldr<U: Copy>(&self, z: U, p: fn(t: &T, u: U) -> U) -> U;
    pure fn map<U>(&self, f: fn(t: &T) -> U) -> ~[U];
    pure fn mapi<U>(&self, f: fn(uint, t: &T) -> U) -> ~[U];
    fn map_r<U>(&self, f: fn(x: &T) -> U) -> ~[U];
    pure fn alli(&self, f: fn(uint, t: &T) -> bool) -> bool;
    pure fn flat_map<U>(&self, f: fn(t: &T) -> ~[U]) -> ~[U];
    pure fn filter_map<U: Copy>(&self, f: fn(t: &T) -> Option<U>) -> ~[U];
}

/// Extension methods for vectors
impl<T> &[T]: ImmutableVector<T> {
    /// Return a slice that points into another slice.
    #[inline]
    pure fn view(&self, start: uint, end: uint) -> &self/[T] {
        view(*self, start, end)
    }

    /// Reduce a vector from right to left
    #[inline]
    pure fn foldr<U: Copy>(&self, z: U, p: fn(t: &T, u: U) -> U) -> U {
        foldr(*self, z, p)
    }

    /// Apply a function to each element of a vector and return the results
    #[inline]
    pure fn map<U>(&self, f: fn(t: &T) -> U) -> ~[U] { map(*self, f) }

    /**
     * Apply a function to the index and value of each element in the vector
     * and return the results
     */
    pure fn mapi<U>(&self, f: fn(uint, t: &T) -> U) -> ~[U] {
        mapi(*self, f)
    }

    #[inline]
    fn map_r<U>(&self, f: fn(x: &T) -> U) -> ~[U] {
        let mut r = ~[];
        let mut i = 0;
        while i < self.len() {
            r.push(f(&self[i]));
            i += 1;
        }
        r
    }

    /**
     * Returns true if the function returns true for all elements.
     *
     *     If the vector is empty, true is returned.
     */
    pure fn alli(&self, f: fn(uint, t: &T) -> bool) -> bool {
        alli(*self, f)
    }
    /**
     * Apply a function to each element of a vector and return a concatenation
     * of each result vector
     */
    #[inline]
    pure fn flat_map<U>(&self, f: fn(t: &T) -> ~[U]) -> ~[U] {
        flat_map(*self, f)
    }
    /**
     * Apply a function to each element of a vector and return the results
     *
     * If function `f` returns `none` then that element is excluded from
     * the resulting vector.
     */
    #[inline]
    pure fn filter_map<U: Copy>(&self, f: fn(t: &T) -> Option<U>) -> ~[U] {
        filter_map(*self, f)
    }
}

pub trait ImmutableEqVector<T: Eq> {
    pure fn position(&self, f: fn(t: &T) -> bool) -> Option<uint>;
    pure fn position_elem(&self, t: &T) -> Option<uint>;
    pure fn rposition(&self, f: fn(t: &T) -> bool) -> Option<uint>;
    pure fn rposition_elem(&self, t: &T) -> Option<uint>;
}

impl<T: Eq> &[T]: ImmutableEqVector<T> {
    /**
     * Find the first index matching some predicate
     *
     * Apply function `f` to each element of `v`.  When function `f` returns
     * true then an option containing the index is returned. If `f` matches no
     * elements then none is returned.
     */
    #[inline]
    pure fn position(&self, f: fn(t: &T) -> bool) -> Option<uint> {
        position(*self, f)
    }

    /// Find the first index containing a matching value
    #[inline]
    pure fn position_elem(&self, x: &T) -> Option<uint> {
        position_elem(*self, x)
    }

    /**
     * Find the last index matching some predicate
     *
     * Apply function `f` to each element of `v` in reverse order.  When
     * function `f` returns true then an option containing the index is
     * returned. If `f` matches no elements then none is returned.
     */
    #[inline]
    pure fn rposition(&self, f: fn(t: &T) -> bool) -> Option<uint> {
        rposition(*self, f)
    }

    /// Find the last index containing a matching value
    #[inline]
    pure fn rposition_elem(&self, t: &T) -> Option<uint> {
        rposition_elem(*self, t)
    }
}

pub trait ImmutableCopyableVector<T> {
    pure fn filtered(&self, f: fn(&T) -> bool) -> ~[T];
    pure fn rfind(&self, f: fn(t: &T) -> bool) -> Option<T>;
    pure fn partitioned(&self, f: fn(&T) -> bool) -> (~[T], ~[T]);
}

/// Extension methods for vectors
impl<T: Copy> &[T]: ImmutableCopyableVector<T> {
    /**
     * Construct a new vector from the elements of a vector for which some
     * predicate holds.
     *
     * Apply function `f` to each element of `v` and return a vector
     * containing only those elements for which `f` returned true.
     */
    #[inline]
    pure fn filtered(&self, f: fn(t: &T) -> bool) -> ~[T] {
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
    pure fn rfind(&self, f: fn(t: &T) -> bool) -> Option<T> {
        rfind(*self, f)
    }

    /**
     * Partitions the vector into those that satisfies the predicate, and
     * those that do not.
     */
    #[inline]
    pure fn partitioned(&self, f: fn(&T) -> bool) -> (~[T], ~[T]) {
        partitioned(*self, f)
    }
}

pub trait OwnedVector<T> {
    fn push(&mut self, t: T);
    fn push_all_move(&mut self, rhs: ~[T]);
    fn pop(&mut self) -> T;
    fn shift(&mut self) -> T;
    fn unshift(&mut self, x: T);
    fn insert(&mut self, i: uint, x:T);
    fn remove(&mut self, i: uint) -> T;
    fn swap_remove(&mut self, index: uint) -> T;
    fn truncate(&mut self, newlen: uint);
    fn retain(&mut self, f: pure fn(t: &T) -> bool);
    fn consume(self, f: fn(uint, v: T));
    fn filter(self, f: fn(t: &T) -> bool) -> ~[T];
    fn partition(self, f: pure fn(&T) -> bool) -> (~[T], ~[T]);
}

impl<T> ~[T]: OwnedVector<T> {
    #[inline]
    fn push(&mut self, t: T) {
        push(self, t);
    }

    #[inline]
    fn push_all_move(&mut self, rhs: ~[T]) {
        push_all_move(self, rhs);
    }

    #[inline]
    fn pop(&mut self) -> T {
        pop(self)
    }

    #[inline]
    fn shift(&mut self) -> T {
        shift(self)
    }

    #[inline]
    fn unshift(&mut self, x: T) {
        unshift(self, x)
    }

    #[inline]
    fn insert(&mut self, i: uint, x:T) {
        insert(self, i, x)
    }

    #[inline]
    fn remove(&mut self, i: uint) -> T {
        remove(self, i)
    }

    #[inline]
    fn swap_remove(&mut self, index: uint) -> T {
        swap_remove(self, index)
    }

    #[inline]
    fn truncate(&mut self, newlen: uint) {
        truncate(self, newlen);
    }

    #[inline]
    fn retain(&mut self, f: pure fn(t: &T) -> bool) {
        retain(self, f);
    }

    #[inline]
    fn consume(self, f: fn(uint, v: T)) {
        consume(self, f)
    }

    #[inline]
    fn filter(self, f: fn(&T) -> bool) -> ~[T] {
        filter(self, f)
    }

    /**
     * Partitions the vector into those that satisfies the predicate, and
     * those that do not.
     */
    #[inline]
    fn partition(self, f: fn(&T) -> bool) -> (~[T], ~[T]) {
        partition(self, f)
    }
}

pub trait OwnedCopyableVector<T: Copy> {
    fn push_all(&mut self, rhs: &[const T]);
    fn grow(&mut self, n: uint, initval: &T);
    fn grow_fn(&mut self, n: uint, op: iter::InitOp<T>);
    fn grow_set(&mut self, index: uint, initval: &T, val: T);
}

impl<T: Copy> ~[T]: OwnedCopyableVector<T> {
    #[inline]
    fn push_all(&mut self, rhs: &[const T]) {
        push_all(self, rhs);
    }

    #[inline]
    fn grow(&mut self, n: uint, initval: &T) {
        grow(self, n, initval);
    }

    #[inline]
    fn grow_fn(&mut self, n: uint, op: iter::InitOp<T>) {
        grow_fn(self, n, op);
    }

    #[inline]
    fn grow_set(&mut self, index: uint, initval: &T, val: T) {
        grow_set(self, index, initval, val);
    }
}

trait OwnedEqVector<T: Eq> {
    fn dedup(&mut self);
}

impl<T: Eq> ~[T]: OwnedEqVector<T> {
    #[inline]
    fn dedup(&mut self) {
        dedup(self)
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
pub struct UnboxedVecRepr {
    mut fill: uint,
    mut alloc: uint,
    data: u8
}

/// Unsafe operations
pub mod raw {
    use kinds::Copy;
    use managed;
    use option::{None, Some};
    use option;
    use ptr::addr_of;
    use ptr;
    use sys;
    use vec::{UnboxedVecRepr, as_const_buf, as_mut_buf, len, with_capacity};
    use vec::rusti;

    /// The internal representation of a (boxed) vector
    pub struct VecRepr {
        box_header: managed::raw::BoxHeaderRepr,
        unboxed: UnboxedVecRepr
    }

    pub type SliceRepr = {
        mut data: *u8,
        mut len: uint
    };

    /**
     * Sets the length of a vector
     *
     * This will explicitly set the size of the vector, without actually
     * modifing its buffers, so it is up to the caller to ensure that
     * the vector is actually the specified size.
     */
    #[inline(always)]
    pub unsafe fn set_len<T>(v: &mut ~[T], new_len: uint) {
        let repr: **VecRepr = ::cast::transmute(v);
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
    #[inline(always)]
    pub unsafe fn to_ptr<T>(v: &[T]) -> *T {
        let repr: **SliceRepr = ::cast::transmute(&v);
        return ::cast::reinterpret_cast(&addr_of(&((**repr).data)));
    }

    /** see `to_ptr()` */
    #[inline(always)]
    pub unsafe fn to_const_ptr<T>(v: &[const T]) -> *const T {
        let repr: **SliceRepr = ::cast::transmute(&v);
        return ::cast::reinterpret_cast(&addr_of(&((**repr).data)));
    }

    /** see `to_ptr()` */
    #[inline(always)]
    pub unsafe fn to_mut_ptr<T>(v: &[mut T]) -> *mut T {
        let repr: **SliceRepr = ::cast::transmute(&v);
        return ::cast::reinterpret_cast(&addr_of(&((**repr).data)));
    }

    /**
     * Form a slice from a pointer and length (as a number of units,
     * not bytes).
     */
    #[inline(always)]
    pub unsafe fn buf_as_slice<T,U>(p: *T,
                                    len: uint,
                                    f: fn(v: &[T]) -> U) -> U {
        let pair = (p, len * sys::nonzero_size_of::<T>());
        let v : *(&blk/[T]) =
            ::cast::reinterpret_cast(&addr_of(&pair));
        f(*v)
    }

    /**
     * Unchecked vector indexing.
     */
    #[inline(always)]
    pub unsafe fn get<T: Copy>(v: &[const T], i: uint) -> T {
        as_const_buf(v, |p, _len| *ptr::const_offset(p, i))
    }

    /**
     * Unchecked vector index assignment.  Does not drop the
     * old value and hence is only suitable when the vector
     * is newly allocated.
     */
    #[inline(always)]
    pub unsafe fn init_elem<T>(v: &[mut T], i: uint, val: T) {
        let mut box = Some(val);
        do as_mut_buf(v) |p, _len| {
            let mut box2 = None;
            box2 <-> box;
            rusti::move_val_init(&mut(*ptr::mut_offset(p, i)),
                                 option::unwrap(box2));
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
    #[inline(always)]
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
    pub unsafe fn copy_memory<T>(dst: &[mut T], src: &[const T],
                                 count: uint) {
        assert dst.len() >= count;
        assert src.len() >= count;

        do as_mut_buf(dst) |p_dst, _len_dst| {
            do as_const_buf(src) |p_src, _len_src| {
                ptr::copy_memory(p_dst, p_src, count)
            }
        }
    }
}

/// Operations on `[u8]`
pub mod bytes {
    use libc;
    use uint;
    use vec::len;
    use vec::raw;
    use vec;

    /// Bytewise string comparison
    pub pure fn cmp(a: &~[u8], b: &~[u8]) -> int {
        let a_len = len(*a);
        let b_len = len(*b);
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
    pub pure fn lt(a: &~[u8], b: &~[u8]) -> bool { cmp(a, b) < 0 }

    /// Bytewise less than or equal
    pub pure fn le(a: &~[u8], b: &~[u8]) -> bool { cmp(a, b) <= 0 }

    /// Bytewise equality
    pub pure fn eq(a: &~[u8], b: &~[u8]) -> bool { cmp(a, b) == 0 }

    /// Bytewise inequality
    pub pure fn ne(a: &~[u8], b: &~[u8]) -> bool { cmp(a, b) != 0 }

    /// Bytewise greater than or equal
    pub pure fn ge(a: &~[u8], b: &~[u8]) -> bool { cmp(a, b) >= 0 }

    /// Bytewise greater than
    pub pure fn gt(a: &~[u8], b: &~[u8]) -> bool { cmp(a, b) > 0 }

    /**
      * Copies data from one vector to another.
      *
      * Copies `count` bytes from `src` to `dst`. The source and destination
      * may overlap.
      */
    pub fn copy_memory(dst: &[mut u8], src: &[const u8], count: uint) {
        // Bound checks are done at vec::raw::copy_memory.
        unsafe { vec::raw::copy_memory(dst, src, count) }
    }
}

// ___________________________________________________________________________
// ITERATION TRAIT METHODS
//
// This cannot be used with iter-trait.rs because of the region pointer
// required in the slice.

impl<A> &[A]: iter::BaseIter<A> {
    pub pure fn each(&self, blk: fn(v: &A) -> bool) {
        // FIXME(#2263)---should be able to call each(self, blk)
        for each(*self) |e| {
            if (!blk(e)) {
                return;
            }
        }
    }
    pure fn size_hint(&self) -> Option<uint> { Some(len(*self)) }
}

// FIXME(#4148): This should be redundant
impl<A> ~[A]: iter::BaseIter<A> {
    pub pure fn each(&self, blk: fn(v: &A) -> bool) {
        // FIXME(#2263)---should be able to call each(self, blk)
        for each(*self) |e| {
            if (!blk(e)) {
                return;
            }
        }
    }
    pure fn size_hint(&self) -> Option<uint> { Some(len(*self)) }
}

// FIXME(#4148): This should be redundant
impl<A> @[A]: iter::BaseIter<A> {
    pub pure fn each(&self, blk: fn(v: &A) -> bool) {
        // FIXME(#2263)---should be able to call each(self, blk)
        for each(*self) |e| {
            if (!blk(e)) {
                return;
            }
        }
    }
    pure fn size_hint(&self) -> Option<uint> { Some(len(*self)) }
}

impl<A> &[A]: iter::ExtendedIter<A> {
    pub pure fn eachi(&self, blk: fn(uint, v: &A) -> bool) {
        iter::eachi(self, blk)
    }
    pub pure fn all(&self, blk: fn(&A) -> bool) -> bool {
        iter::all(self, blk)
    }
    pub pure fn any(&self, blk: fn(&A) -> bool) -> bool {
        iter::any(self, blk)
    }
    pub pure fn foldl<B>(&self, b0: B, blk: fn(&B, &A) -> B) -> B {
        iter::foldl(self, b0, blk)
    }
    pub pure fn position(&self, f: fn(&A) -> bool) -> Option<uint> {
        iter::position(self, f)
    }
    pure fn map_to_vec<B>(&self, op: fn(&A) -> B) -> ~[B] {
        iter::map_to_vec(self, op)
    }
    pure fn flat_map_to_vec<B,IB:BaseIter<B>>(&self, op: fn(&A) -> IB)
        -> ~[B] {
        iter::flat_map_to_vec(self, op)
    }
}

// FIXME(#4148): This should be redundant
impl<A> ~[A]: iter::ExtendedIter<A> {
    pub pure fn eachi(&self, blk: fn(uint, v: &A) -> bool) {
        iter::eachi(self, blk)
    }
    pub pure fn all(&self, blk: fn(&A) -> bool) -> bool {
        iter::all(self, blk)
    }
    pub pure fn any(&self, blk: fn(&A) -> bool) -> bool {
        iter::any(self, blk)
    }
    pub pure fn foldl<B>(&self, b0: B, blk: fn(&B, &A) -> B) -> B {
        iter::foldl(self, b0, blk)
    }
    pub pure fn position(&self, f: fn(&A) -> bool) -> Option<uint> {
        iter::position(self, f)
    }
    pure fn map_to_vec<B>(&self, op: fn(&A) -> B) -> ~[B] {
        iter::map_to_vec(self, op)
    }
    pure fn flat_map_to_vec<B,IB:BaseIter<B>>(&self, op: fn(&A) -> IB)
        -> ~[B] {
        iter::flat_map_to_vec(self, op)
    }
}

// FIXME(#4148): This should be redundant
impl<A> @[A]: iter::ExtendedIter<A> {
    pub pure fn eachi(&self, blk: fn(uint, v: &A) -> bool) {
        iter::eachi(self, blk)
    }
    pub pure fn all(&self, blk: fn(&A) -> bool) -> bool {
        iter::all(self, blk)
    }
    pub pure fn any(&self, blk: fn(&A) -> bool) -> bool {
        iter::any(self, blk)
    }
    pub pure fn foldl<B>(&self, b0: B, blk: fn(&B, &A) -> B) -> B {
        iter::foldl(self, b0, blk)
    }
    pub pure fn position(&self, f: fn(&A) -> bool) -> Option<uint> {
        iter::position(self, f)
    }
    pure fn map_to_vec<B>(&self, op: fn(&A) -> B) -> ~[B] {
        iter::map_to_vec(self, op)
    }
    pure fn flat_map_to_vec<B,IB:BaseIter<B>>(&self, op: fn(&A) -> IB)
        -> ~[B] {
        iter::flat_map_to_vec(self, op)
    }
}

impl<A: Eq> &[A]: iter::EqIter<A> {
    pub pure fn contains(&self, x: &A) -> bool { iter::contains(self, x) }
    pub pure fn count(&self, x: &A) -> uint { iter::count(self, x) }
}

// FIXME(#4148): This should be redundant
impl<A: Eq> ~[A]: iter::EqIter<A> {
    pub pure fn contains(&self, x: &A) -> bool { iter::contains(self, x) }
    pub pure fn count(&self, x: &A) -> uint { iter::count(self, x) }
}

// FIXME(#4148): This should be redundant
impl<A: Eq> @[A]: iter::EqIter<A> {
    pub pure fn contains(&self, x: &A) -> bool { iter::contains(self, x) }
    pub pure fn count(&self, x: &A) -> uint { iter::count(self, x) }
}

impl<A: Copy> &[A]: iter::CopyableIter<A> {
    pure fn filter_to_vec(&self, pred: fn(&A) -> bool) -> ~[A] {
        iter::filter_to_vec(self, pred)
    }
    pure fn to_vec(&self) -> ~[A] { iter::to_vec(self) }
    pub pure fn find(&self, f: fn(&A) -> bool) -> Option<A> {
        iter::find(self, f)
    }
}

// FIXME(#4148): This should be redundant
impl<A: Copy> ~[A]: iter::CopyableIter<A> {
    pure fn filter_to_vec(&self, pred: fn(&A) -> bool) -> ~[A] {
        iter::filter_to_vec(self, pred)
    }
    pure fn to_vec(&self) -> ~[A] { iter::to_vec(self) }
    pub pure fn find(&self, f: fn(&A) -> bool) -> Option<A> {
        iter::find(self, f)
    }
}

// FIXME(#4148): This should be redundant
impl<A: Copy> @[A]: iter::CopyableIter<A> {
    pure fn filter_to_vec(&self, pred: fn(&A) -> bool) -> ~[A] {
        iter::filter_to_vec(self, pred)
    }
    pure fn to_vec(&self) -> ~[A] { iter::to_vec(self) }
    pub pure fn find(&self, f: fn(&A) -> bool) -> Option<A> {
        iter::find(self, f)
    }
}

impl<A: Copy Ord> &[A]: iter::CopyableOrderedIter<A> {
    pure fn min(&self) -> A { iter::min(self) }
    pure fn max(&self) -> A { iter::max(self) }
}

// FIXME(#4148): This should be redundant
impl<A: Copy Ord> ~[A]: iter::CopyableOrderedIter<A> {
    pure fn min(&self) -> A { iter::min(self) }
    pure fn max(&self) -> A { iter::max(self) }
}

// FIXME(#4148): This should be redundant
impl<A: Copy Ord> @[A]: iter::CopyableOrderedIter<A> {
    pure fn min(&self) -> A { iter::min(self) }
    pure fn max(&self) -> A { iter::max(self) }
}

impl<A:Copy> &[A] : iter::CopyableNonstrictIter<A> {
    pure fn each_val(&const self, f: fn(A) -> bool) {
        let mut i = 0;
        while i < self.len() {
            if !f(copy self[i]) { break; }
            i += 1;
        }
    }
}

// FIXME(#4148): This should be redundant
impl<A:Copy> ~[A] : iter::CopyableNonstrictIter<A> {
    pure fn each_val(&const self, f: fn(A) -> bool) {
        let mut i = 0;
        while i < self.len() {
            if !f(copy self[i]) { break; }
            i += 1;
        }
    }
}

// FIXME(#4148): This should be redundant
impl<A:Copy> @[A] : iter::CopyableNonstrictIter<A> {
    pure fn each_val(&const self, f: fn(A) -> bool) {
        let mut i = 0;
        while i < self.len() {
            if !f(copy self[i]) { break; }
            i += 1;
        }
    }
}

// ___________________________________________________________________________

#[cfg(test)]
mod tests {
    use option::{None, Option, Some};
    use option;
    use vec::*;

    fn square(n: uint) -> uint { return n * n; }

    fn square_ref(n: &uint) -> uint { return square(*n); }

    pure fn is_three(n: &uint) -> bool { return *n == 3u; }

    pure fn is_odd(n: &uint) -> bool { return *n % 2u == 1u; }

    pure fn is_equal(x: &uint, y:&uint) -> bool { return *x == *y; }

    fn square_if_odd(n: &uint) -> Option<uint> {
        return if *n % 2u == 1u { Some(*n * *n) } else { None };
    }

    fn add(x: uint, y: &uint) -> uint { return x + *y; }

    #[test]
    fn test_unsafe_ptrs() {
        unsafe {
            // Test on-stack copy-from-buf.
            let a = ~[1, 2, 3];
            let mut ptr = raw::to_ptr(a);
            let b = from_buf(ptr, 3u);
            assert (len(b) == 3u);
            assert (b[0] == 1);
            assert (b[1] == 2);
            assert (b[2] == 3);

            // Test on-heap copy-from-buf.
            let c = ~[1, 2, 3, 4, 5];
            ptr = raw::to_ptr(c);
            let d = from_buf(ptr, 5u);
            assert (len(d) == 5u);
            assert (d[0] == 1);
            assert (d[1] == 2);
            assert (d[2] == 3);
            assert (d[3] == 4);
            assert (d[4] == 5);
        }
    }

    #[test]
    fn test_from_fn() {
        // Test on-stack from_fn.
        let mut v = from_fn(3u, square);
        assert (len(v) == 3u);
        assert (v[0] == 0u);
        assert (v[1] == 1u);
        assert (v[2] == 4u);

        // Test on-heap from_fn.
        v = from_fn(5u, square);
        assert (len(v) == 5u);
        assert (v[0] == 0u);
        assert (v[1] == 1u);
        assert (v[2] == 4u);
        assert (v[3] == 9u);
        assert (v[4] == 16u);
    }

    #[test]
    fn test_from_elem() {
        // Test on-stack from_elem.
        let mut v = from_elem(2u, 10u);
        assert (len(v) == 2u);
        assert (v[0] == 10u);
        assert (v[1] == 10u);

        // Test on-heap from_elem.
        v = from_elem(6u, 20u);
        assert (v[0] == 20u);
        assert (v[1] == 20u);
        assert (v[2] == 20u);
        assert (v[3] == 20u);
        assert (v[4] == 20u);
        assert (v[5] == 20u);
    }

    #[test]
    fn test_is_empty() {
        assert (is_empty::<int>(~[]));
        assert (!is_empty(~[0]));
    }

    #[test]
    fn test_is_not_empty() {
        assert (is_not_empty(~[0]));
        assert (!is_not_empty::<int>(~[]));
    }

    #[test]
    fn test_len_divzero() {
        type Z = [i8 * 0];
        let v0 : &[Z] = &[];
        let v1 : &[Z] = &[[]];
        let v2 : &[Z] = &[[], []];
        assert(sys::size_of::<Z>() == 0);
        assert(len(v0) == 0);
        assert(len(v1) == 1);
        assert(len(v2) == 2);
    }

    #[test]
    fn test_head() {
        let a = ~[11, 12];
        assert (head(a) == 11);
    }

    #[test]
    fn test_tail() {
        let mut a = ~[11];
        assert (tail(a) == ~[]);

        a = ~[11, 12];
        assert (tail(a) == ~[12]);
    }

    #[test]
    fn test_last() {
        let mut n = last_opt(~[]);
        assert (n.is_none());
        n = last_opt(~[1, 2, 3]);
        assert (n == Some(3));
        n = last_opt(~[1, 2, 3, 4, 5]);
        assert (n == Some(5));
    }

    #[test]
    fn test_slice() {
        // Test on-stack -> on-stack slice.
        let mut v = slice(~[1, 2, 3], 1u, 3u);
        assert (len(v) == 2u);
        assert (v[0] == 2);
        assert (v[1] == 3);

        // Test on-heap -> on-stack slice.
        v = slice(~[1, 2, 3, 4, 5], 0u, 3u);
        assert (len(v) == 3u);
        assert (v[0] == 1);
        assert (v[1] == 2);
        assert (v[2] == 3);

        // Test on-heap -> on-heap slice.
        v = slice(~[1, 2, 3, 4, 5, 6], 1u, 6u);
        assert (len(v) == 5u);
        assert (v[0] == 2);
        assert (v[1] == 3);
        assert (v[2] == 4);
        assert (v[3] == 5);
        assert (v[4] == 6);
    }

    #[test]
    fn test_pop() {
        // Test on-stack pop.
        let mut v = ~[1, 2, 3];
        let mut e = v.pop();
        assert (len(v) == 2u);
        assert (v[0] == 1);
        assert (v[1] == 2);
        assert (e == 3);

        // Test on-heap pop.
        v = ~[1, 2, 3, 4, 5];
        e = v.pop();
        assert (len(v) == 4u);
        assert (v[0] == 1);
        assert (v[1] == 2);
        assert (v[2] == 3);
        assert (v[3] == 4);
        assert (e == 5);
    }

    #[test]
    fn test_swap_remove() {
        let mut v = ~[1, 2, 3, 4, 5];
        let mut e = v.swap_remove(0);
        assert (len(v) == 4);
        assert e == 1;
        assert (v[0] == 5);
        e = v.swap_remove(3);
        assert (len(v) == 3);
        assert e == 4;
        assert (v[0] == 5);
        assert (v[1] == 2);
        assert (v[2] == 3);
    }

    #[test]
    fn test_swap_remove_noncopyable() {
        // Tests that we don't accidentally run destructors twice.
        let mut v = ~[::private::exclusive(()), ::private::exclusive(()),
                      ::private::exclusive(())];
        let mut _e = v.swap_remove(0);
        assert (len(v) == 2);
        _e = v.swap_remove(1);
        assert (len(v) == 1);
        _e = v.swap_remove(0);
        assert (len(v) == 0);
    }

    #[test]
    fn test_push() {
        // Test on-stack push().
        let mut v = ~[];
        v.push(1);
        assert (len(v) == 1u);
        assert (v[0] == 1);

        // Test on-heap push().
        v.push(2);
        assert (len(v) == 2u);
        assert (v[0] == 1);
        assert (v[1] == 2);
    }

    #[test]
    fn test_grow() {
        // Test on-stack grow().
        let mut v = ~[];
        v.grow(2u, &1);
        assert (len(v) == 2u);
        assert (v[0] == 1);
        assert (v[1] == 1);

        // Test on-heap grow().
        v.grow(3u, &2);
        assert (len(v) == 5u);
        assert (v[0] == 1);
        assert (v[1] == 1);
        assert (v[2] == 2);
        assert (v[3] == 2);
        assert (v[4] == 2);
    }

    #[test]
    fn test_grow_fn() {
        let mut v = ~[];
        v.grow_fn(3u, square);
        assert (len(v) == 3u);
        assert (v[0] == 0u);
        assert (v[1] == 1u);
        assert (v[2] == 4u);
    }

    #[test]
    fn test_grow_set() {
        let mut v = ~[1, 2, 3];
        v.grow_set(4u, &4, 5);
        assert (len(v) == 5u);
        assert (v[0] == 1);
        assert (v[1] == 2);
        assert (v[2] == 3);
        assert (v[3] == 4);
        assert (v[4] == 5);
    }

    #[test]
    fn test_truncate() {
        let mut v = ~[@6,@5,@4];
        v.truncate(1);
        assert(v.len() == 1);
        assert(*(v[0]) == 6);
        // If the unsafe block didn't drop things properly, we blow up here.
    }

    #[test]
    fn test_dedup() {
        fn case(a: ~[uint], b: ~[uint]) {
            let mut v = a;
            v.dedup();
            assert(v == b);
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
        assert (len(w) == 3u);
        assert (w[0] == 1u);
        assert (w[1] == 4u);
        assert (w[2] == 9u);

        // Test on-heap map.
        v = ~[1u, 2u, 3u, 4u, 5u];
        w = map(v, square_ref);
        assert (len(w) == 5u);
        assert (w[0] == 1u);
        assert (w[1] == 4u);
        assert (w[2] == 9u);
        assert (w[3] == 16u);
        assert (w[4] == 25u);
    }

    #[test]
    fn test_map2() {
        fn times(x: &int, y: &int) -> int { return *x * *y; }
        let f = times;
        let v0 = ~[1, 2, 3, 4, 5];
        let v1 = ~[5, 4, 3, 2, 1];
        let u = map2::<int, int, int>(v0, v1, f);
        let mut i = 0;
        while i < 5 { assert (v0[i] * v1[i] == u[i]); i += 1; }
    }

    #[test]
    fn test_filter_map() {
        // Test on-stack filter-map.
        let mut v = ~[1u, 2u, 3u];
        let mut w = filter_map(v, square_if_odd);
        assert (len(w) == 2u);
        assert (w[0] == 1u);
        assert (w[1] == 9u);

        // Test on-heap filter-map.
        v = ~[1u, 2u, 3u, 4u, 5u];
        w = filter_map(v, square_if_odd);
        assert (len(w) == 3u);
        assert (w[0] == 1u);
        assert (w[1] == 9u);
        assert (w[2] == 25u);

        fn halve(i: &int) -> Option<int> {
            if *i % 2 == 0 {
                return option::Some::<int>(*i / 2);
            } else {
                return option::None::<int>;
            }
        }
        fn halve_for_sure(i: &int) -> int { return *i / 2; }
        let all_even: ~[int] = ~[0, 2, 8, 6];
        let all_odd1: ~[int] = ~[1, 7, 3];
        let all_odd2: ~[int] = ~[];
        let mix: ~[int] = ~[9, 2, 6, 7, 1, 0, 0, 3];
        let mix_dest: ~[int] = ~[1, 3, 0, 0];
        assert (filter_map(all_even, halve) == map(all_even, halve_for_sure));
        assert (filter_map(all_odd1, halve) == ~[]);
        assert (filter_map(all_odd2, halve) == ~[]);
        assert (filter_map(mix, halve) == mix_dest);
    }

    #[test]
    fn test_filter() {
        assert filter(~[1u, 2u, 3u], is_odd) == ~[1u, 3u];
        assert filter(~[1u, 2u, 4u, 8u, 16u], is_three) == ~[];
    }

    #[test]
    fn test_retain() {
        let mut v = ~[1, 2, 3, 4, 5];
        v.retain(is_odd);
        assert v == ~[1, 3, 5];
    }

    #[test]
    fn test_foldl() {
        // Test on-stack fold.
        let mut v = ~[1u, 2u, 3u];
        let mut sum = foldl(0u, v, add);
        assert (sum == 6u);

        // Test on-heap fold.
        v = ~[1u, 2u, 3u, 4u, 5u];
        sum = foldl(0u, v, add);
        assert (sum == 15u);
    }

    #[test]
    fn test_foldl2() {
        fn sub(a: int, b: &int) -> int {
            a - *b
        }
        let mut v = ~[1, 2, 3, 4];
        let sum = foldl(0, v, sub);
        assert sum == -10;
    }

    #[test]
    fn test_foldr() {
        fn sub(a: &int, b: int) -> int {
            *a - b
        }
        let mut v = ~[1, 2, 3, 4];
        let sum = foldr(v, 0, sub);
        assert sum == -2;
    }

    #[test]
    fn test_each_empty() {
        for each::<int>(~[]) |_v| {
            fail; // should never be executed
        }
    }

    #[test]
    fn test_iter_nonempty() {
        let mut i = 0;
        for each(~[1, 2, 3]) |v| {
            i += *v;
        }
        assert i == 6;
    }

    #[test]
    fn test_iteri() {
        let mut i = 0;
        for eachi(~[1, 2, 3]) |j, v| {
            if i == 0 { assert *v == 1; }
            assert j + 1u == *v as uint;
            i += *v;
        }
        assert i == 6;
    }

    #[test]
    fn test_reach_empty() {
        for rev_each::<int>(~[]) |_v| {
            fail; // should never execute
        }
    }

    #[test]
    fn test_reach_nonempty() {
        let mut i = 0;
        for rev_each(~[1, 2, 3]) |v| {
            if i == 0 { assert *v == 3; }
            i += *v
        }
        assert i == 6;
    }

    #[test]
    fn test_reachi() {
        let mut i = 0;
        for rev_eachi(~[0, 1, 2]) |j, v| {
            if i == 0 { assert *v == 2; }
            assert j == *v as uint;
            i += *v;
        }
        assert i == 3;
    }

    #[test]
    fn test_each_permutation() {
        let mut results: ~[~[int]];

        results = ~[];
        for each_permutation(~[]) |v| { results.push(from_slice(v)); }
        assert results == ~[~[]];

        results = ~[];
        for each_permutation(~[7]) |v| { results.push(from_slice(v)); }
        assert results == ~[~[7]];

        results = ~[];
        for each_permutation(~[1,1]) |v| { results.push(from_slice(v)); }
        assert results == ~[~[1,1],~[1,1]];

        results = ~[];
        for each_permutation(~[5,2,0]) |v| { results.push(from_slice(v)); }
        assert results ==
            ~[~[5,2,0],~[5,0,2],~[2,5,0],~[2,0,5],~[0,5,2],~[0,2,5]];
    }

    #[test]
    fn test_any_and_all() {
        assert (any(~[1u, 2u, 3u], is_three));
        assert (!any(~[0u, 1u, 2u], is_three));
        assert (any(~[1u, 2u, 3u, 4u, 5u], is_three));
        assert (!any(~[1u, 2u, 4u, 5u, 6u], is_three));

        assert (all(~[3u, 3u, 3u], is_three));
        assert (!all(~[3u, 3u, 2u], is_three));
        assert (all(~[3u, 3u, 3u, 3u, 3u], is_three));
        assert (!all(~[3u, 3u, 0u, 1u, 2u], is_three));
    }

    #[test]
    fn test_any2_and_all2() {

        assert (any2(~[2u, 4u, 6u], ~[2u, 4u, 6u], is_equal));
        assert (any2(~[1u, 2u, 3u], ~[4u, 5u, 3u], is_equal));
        assert (!any2(~[1u, 2u, 3u], ~[4u, 5u, 6u], is_equal));
        assert (any2(~[2u, 4u, 6u], ~[2u, 4u], is_equal));

        assert (all2(~[2u, 4u, 6u], ~[2u, 4u, 6u], is_equal));
        assert (!all2(~[1u, 2u, 3u], ~[4u, 5u, 3u], is_equal));
        assert (!all2(~[1u, 2u, 3u], ~[4u, 5u, 6u], is_equal));
        assert (!all2(~[2u, 4u, 6u], ~[2u, 4u], is_equal));
    }

    #[test]
    fn test_zip_unzip() {
        let v1 = ~[1, 2, 3];
        let v2 = ~[4, 5, 6];

        let z1 = zip(v1, v2);

        assert ((1, 4) == z1[0]);
        assert ((2, 5) == z1[1]);
        assert ((3, 6) == z1[2]);

        let (left, right) = unzip(z1);

        assert ((1, 4) == (left[0], right[0]));
        assert ((2, 5) == (left[1], right[1]));
        assert ((3, 6) == (left[2], right[2]));
    }

    #[test]
    fn test_position_elem() {
        assert position_elem(~[], &1).is_none();

        let v1 = ~[1, 2, 3, 3, 2, 5];
        assert position_elem(v1, &1) == Some(0u);
        assert position_elem(v1, &2) == Some(1u);
        assert position_elem(v1, &5) == Some(5u);
        assert position_elem(v1, &4).is_none();
    }

    #[test]
    fn test_position() {
        fn less_than_three(i: &int) -> bool { return *i < 3; }
        fn is_eighteen(i: &int) -> bool { return *i == 18; }

        assert position(~[], less_than_three).is_none();

        let v1 = ~[5, 4, 3, 2, 1];
        assert position(v1, less_than_three) == Some(3u);
        assert position(v1, is_eighteen).is_none();
    }

    #[test]
    fn test_position_between() {
        assert position_between(~[], 0u, 0u, f).is_none();

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert position_between(v, 0u, 0u, f).is_none();
        assert position_between(v, 0u, 1u, f).is_none();
        assert position_between(v, 0u, 2u, f) == Some(1u);
        assert position_between(v, 0u, 3u, f) == Some(1u);
        assert position_between(v, 0u, 4u, f) == Some(1u);

        assert position_between(v, 1u, 1u, f).is_none();
        assert position_between(v, 1u, 2u, f) == Some(1u);
        assert position_between(v, 1u, 3u, f) == Some(1u);
        assert position_between(v, 1u, 4u, f) == Some(1u);

        assert position_between(v, 2u, 2u, f).is_none();
        assert position_between(v, 2u, 3u, f).is_none();
        assert position_between(v, 2u, 4u, f) == Some(3u);

        assert position_between(v, 3u, 3u, f).is_none();
        assert position_between(v, 3u, 4u, f) == Some(3u);

        assert position_between(v, 4u, 4u, f).is_none();
    }

    #[test]
    fn test_find() {
        assert find(~[], f).is_none();

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        fn g(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'd' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert find(v, f) == Some((1, 'b'));
        assert find(v, g).is_none();
    }

    #[test]
    fn test_find_between() {
        assert find_between(~[], 0u, 0u, f).is_none();

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert find_between(v, 0u, 0u, f).is_none();
        assert find_between(v, 0u, 1u, f).is_none();
        assert find_between(v, 0u, 2u, f) == Some((1, 'b'));
        assert find_between(v, 0u, 3u, f) == Some((1, 'b'));
        assert find_between(v, 0u, 4u, f) == Some((1, 'b'));

        assert find_between(v, 1u, 1u, f).is_none();
        assert find_between(v, 1u, 2u, f) == Some((1, 'b'));
        assert find_between(v, 1u, 3u, f) == Some((1, 'b'));
        assert find_between(v, 1u, 4u, f) == Some((1, 'b'));

        assert find_between(v, 2u, 2u, f).is_none();
        assert find_between(v, 2u, 3u, f).is_none();
        assert find_between(v, 2u, 4u, f) == Some((3, 'b'));

        assert find_between(v, 3u, 3u, f).is_none();
        assert find_between(v, 3u, 4u, f) == Some((3, 'b'));

        assert find_between(v, 4u, 4u, f).is_none();
    }

    #[test]
    fn test_rposition() {
        assert find(~[], f).is_none();

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        fn g(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'd' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert position(v, f) == Some(1u);
        assert position(v, g).is_none();
    }

    #[test]
    fn test_rposition_between() {
        assert rposition_between(~[], 0u, 0u, f).is_none();

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert rposition_between(v, 0u, 0u, f).is_none();
        assert rposition_between(v, 0u, 1u, f).is_none();
        assert rposition_between(v, 0u, 2u, f) == Some(1u);
        assert rposition_between(v, 0u, 3u, f) == Some(1u);
        assert rposition_between(v, 0u, 4u, f) == Some(3u);

        assert rposition_between(v, 1u, 1u, f).is_none();
        assert rposition_between(v, 1u, 2u, f) == Some(1u);
        assert rposition_between(v, 1u, 3u, f) == Some(1u);
        assert rposition_between(v, 1u, 4u, f) == Some(3u);

        assert rposition_between(v, 2u, 2u, f).is_none();
        assert rposition_between(v, 2u, 3u, f).is_none();
        assert rposition_between(v, 2u, 4u, f) == Some(3u);

        assert rposition_between(v, 3u, 3u, f).is_none();
        assert rposition_between(v, 3u, 4u, f) == Some(3u);

        assert rposition_between(v, 4u, 4u, f).is_none();
    }

    #[test]
    fn test_rfind() {
        assert rfind(~[], f).is_none();

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        fn g(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'd' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert rfind(v, f) == Some((3, 'b'));
        assert rfind(v, g).is_none();
    }

    #[test]
    fn test_rfind_between() {
        assert rfind_between(~[], 0u, 0u, f).is_none();

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert rfind_between(v, 0u, 0u, f).is_none();
        assert rfind_between(v, 0u, 1u, f).is_none();
        assert rfind_between(v, 0u, 2u, f) == Some((1, 'b'));
        assert rfind_between(v, 0u, 3u, f) == Some((1, 'b'));
        assert rfind_between(v, 0u, 4u, f) == Some((3, 'b'));

        assert rfind_between(v, 1u, 1u, f).is_none();
        assert rfind_between(v, 1u, 2u, f) == Some((1, 'b'));
        assert rfind_between(v, 1u, 3u, f) == Some((1, 'b'));
        assert rfind_between(v, 1u, 4u, f) == Some((3, 'b'));

        assert rfind_between(v, 2u, 2u, f).is_none();
        assert rfind_between(v, 2u, 3u, f).is_none();
        assert rfind_between(v, 2u, 4u, f) == Some((3, 'b'));

        assert rfind_between(v, 3u, 3u, f).is_none();
        assert rfind_between(v, 3u, 4u, f) == Some((3, 'b'));

        assert rfind_between(v, 4u, 4u, f).is_none();
    }

    #[test]
    fn reverse_and_reversed() {
        let v: ~[mut int] = ~[mut 10, 20];
        assert (v[0] == 10);
        assert (v[1] == 20);
        reverse(v);
        assert (v[0] == 20);
        assert (v[1] == 10);
        let v2 = reversed::<int>(~[10, 20]);
        assert (v2[0] == 20);
        assert (v2[1] == 10);
        v[0] = 30;
        assert (v2[0] == 20);
        // Make sure they work with 0-length vectors too.

        let v4 = reversed::<int>(~[]);
        assert (v4 == ~[]);
        let v3: ~[mut int] = ~[mut];
        reverse::<int>(v3);
    }

    #[test]
    fn reversed_mut() {
        let v2 = reversed::<int>(~[mut 10, 20]);
        assert (v2[0] == 20);
        assert (v2[1] == 10);
    }

    #[test]
    fn test_init() {
        let v = init(~[1, 2, 3]);
        assert v == ~[1, 2];
    }

    #[test]
    fn test_split() {
        fn f(x: &int) -> bool { *x == 3 }

        assert split(~[], f) == ~[];
        assert split(~[1, 2], f) == ~[~[1, 2]];
        assert split(~[3, 1, 2], f) == ~[~[], ~[1, 2]];
        assert split(~[1, 2, 3], f) == ~[~[1, 2], ~[]];
        assert split(~[1, 2, 3, 4, 3, 5], f) == ~[~[1, 2], ~[4], ~[5]];
    }

    #[test]
    fn test_splitn() {
        fn f(x: &int) -> bool { *x == 3 }

        assert splitn(~[], 1u, f) == ~[];
        assert splitn(~[1, 2], 1u, f) == ~[~[1, 2]];
        assert splitn(~[3, 1, 2], 1u, f) == ~[~[], ~[1, 2]];
        assert splitn(~[1, 2, 3], 1u, f) == ~[~[1, 2], ~[]];
        assert splitn(~[1, 2, 3, 4, 3, 5], 1u, f) ==
                      ~[~[1, 2], ~[4, 3, 5]];
    }

    #[test]
    fn test_rsplit() {
        fn f(x: &int) -> bool { *x == 3 }

        assert rsplit(~[], f) == ~[];
        assert rsplit(~[1, 2], f) == ~[~[1, 2]];
        assert rsplit(~[1, 2, 3], f) == ~[~[1, 2], ~[]];
        assert rsplit(~[1, 2, 3, 4, 3, 5], f) == ~[~[1, 2], ~[4], ~[5]];
    }

    #[test]
    fn test_rsplitn() {
        fn f(x: &int) -> bool { *x == 3 }

        assert rsplitn(~[], 1u, f) == ~[];
        assert rsplitn(~[1, 2], 1u, f) == ~[~[1, 2]];
        assert rsplitn(~[1, 2, 3], 1u, f) == ~[~[1, 2], ~[]];
        assert rsplitn(~[1, 2, 3, 4, 3, 5], 1u, f) ==
                       ~[~[1, 2, 3, 4], ~[5]];
    }

    #[test]
    fn test_partition() {
        // FIXME (#4355 maybe): using v.partition here crashes
        assert partition(~[], |x: &int| *x < 3) == (~[], ~[]);
        assert partition(~[1, 2, 3], |x: &int| *x < 4) == (~[1, 2, 3], ~[]);
        assert partition(~[1, 2, 3], |x: &int| *x < 2) == (~[1], ~[2, 3]);
        assert partition(~[1, 2, 3], |x: &int| *x < 0) == (~[], ~[1, 2, 3]);
    }

    #[test]
    fn test_partitioned() {
        assert (~[]).partitioned(|x: &int| *x < 3) == (~[], ~[]);
        assert (~[1, 2, 3]).partitioned(|x: &int| *x < 4) ==
               (~[1, 2, 3], ~[]);
        assert (~[1, 2, 3]).partitioned(|x: &int| *x < 2) ==
               (~[1], ~[2, 3]);
        assert (~[1, 2, 3]).partitioned(|x: &int| *x < 0) ==
               (~[], ~[1, 2, 3]);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_init_empty() {
        init::<int>(~[]);
    }

    #[test]
    fn test_concat() {
        assert concat(~[~[1], ~[2,3]]) == ~[1, 2, 3];
    }

    #[test]
    fn test_connect() {
        assert connect(~[], &0) == ~[];
        assert connect(~[~[1], ~[2, 3]], &0) == ~[1, 0, 2, 3];
        assert connect(~[~[1], ~[2], ~[3]], &0) == ~[1, 0, 2, 0, 3];
    }

    #[test]
    fn test_windowed () {
        assert ~[~[1u,2u,3u],~[2u,3u,4u],~[3u,4u,5u],~[4u,5u,6u]]
            == windowed (3u, ~[1u,2u,3u,4u,5u,6u]);

        assert ~[~[1u,2u,3u,4u],~[2u,3u,4u,5u],~[3u,4u,5u,6u]]
            == windowed (4u, ~[1u,2u,3u,4u,5u,6u]);

        assert ~[] == windowed (7u, ~[1u,2u,3u,4u,5u,6u]);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_windowed_() {
        let _x = windowed (0u, ~[1u,2u,3u,4u,5u,6u]);
    }

    #[test]
    fn to_mut_no_copy() {
        unsafe {
            let x = ~[1, 2, 3];
            let addr = raw::to_ptr(x);
            let x_mut = to_mut(x);
            let addr_mut = raw::to_ptr(x_mut);
            assert addr == addr_mut;
        }
    }

    #[test]
    fn from_mut_no_copy() {
        unsafe {
            let x = ~[mut 1, 2, 3];
            let addr = raw::to_ptr(x);
            let x_imm = from_mut(x);
            let addr_imm = raw::to_ptr(x_imm);
            assert addr == addr_imm;
        }
    }

    #[test]
    fn test_unshift() {
        let mut x = ~[1, 2, 3];
        x.unshift(0);
        assert x == ~[0, 1, 2, 3];
    }

    #[test]
    fn test_insert() {
        let mut a = ~[1, 2, 4];
        a.insert(2, 3);
        assert a == ~[1, 2, 3, 4];

        let mut a = ~[1, 2, 3];
        a.insert(0, 0);
        assert a == ~[0, 1, 2, 3];

        let mut a = ~[1, 2, 3];
        a.insert(3, 4);
        assert a == ~[1, 2, 3, 4];

        let mut a = ~[];
        a.insert(0, 1);
        assert a == ~[1];
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
        assert a == ~[1, 2, 4];

        let mut a = ~[1, 2, 3];
        a.remove(0);
        assert a == ~[2, 3];

        let mut a = ~[1];
        a.remove(0);
        assert a == ~[];
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
        reserve(&mut v, 10u);
        assert capacity(&v) == 10u;
        let mut v = ~[0u32];
        reserve(&mut v, 10u);
        assert capacity(&v) == 10u;
    }

    #[test]
    fn test_view() {
        let v = ~[1, 2, 3, 4, 5];
        let v = v.view(1u, 3u);
        assert(len(v) == 2u);
        assert(v[0] == 2);
        assert(v[1] == 3);
    }


    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_from_fn_fail() {
        do from_fn(100) |v| {
            if v == 50 { fail }
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
            fail;
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
                fail
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
                fail
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
                fail
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
                fail
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
                fail
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
                fail
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
                fail
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
                fail
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
                fail
            }
            i += 1;
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_consume_mut_fail() {
        let v = ~[mut (~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do consume_mut(v) |_i, _elt| {
            if i == 2 {
                fail
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
                fail
            }
            (~0, @0)
        }
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_map_fail() {
        let v = [mut (~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do map(v) |_elt| {
            if i == 2 {
                fail
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
                fail
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
                fail
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
                fail
            }
            i += 0;
            ~[(~0, @0)]
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    #[allow(non_implicitly_copyable_typarams)]
    fn test_map2_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do map2(v, v) |_elt1, _elt2| {
            if i == 2 {
                fail
            }
            i += 0;
            ~[(~0, @0)]
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    #[allow(non_implicitly_copyable_typarams)]
    fn test_filter_map_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do filter_map(v) |_elt| {
            if i == 2 {
                fail
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
                fail
            }
            i += 0;
            true
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    #[allow(non_implicitly_copyable_typarams)]
    fn test_foldl_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do foldl((~0, @0), v) |_a, _b| {
            if i == 2 {
                fail
            }
            i += 0;
            (~0, @0)
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    #[allow(non_implicitly_copyable_typarams)]
    fn test_foldr_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do foldr(v, (~0, @0)) |_a, _b| {
            if i == 2 {
                fail
            }
            i += 0;
            (~0, @0)
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_any_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do any(v) |_elt| {
            if i == 2 {
                fail
            }
            i += 0;
            false
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_any2_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do any(v) |_elt| {
            if i == 2 {
                fail
            }
            i += 0;
            false
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_all_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do all(v) |_elt| {
            if i == 2 {
                fail
            }
            i += 0;
            true
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_alli_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do alli(v) |_i, _elt| {
            if i == 2 {
                fail
            }
            i += 0;
            true
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_all2_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do all2(v, v) |_elt1, _elt2| {
            if i == 2 {
                fail
            }
            i += 0;
            true
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    #[allow(non_implicitly_copyable_typarams)]
    fn test_find_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do find(v) |_elt| {
            if i == 2 {
                fail
            }
            i += 0;
            false
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_position_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do position(v) |_elt| {
            if i == 2 {
                fail
            }
            i += 0;
            false
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
                fail
            }
            i += 0;
            false
        };
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_each_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do each(v) |_elt| {
            if i == 2 {
                fail
            }
            i += 0;
            false
        }
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_eachi_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do eachi(v) |_i, _elt| {
            if i == 2 {
                fail
            }
            i += 0;
            false
        }
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
                fail
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
            fail
        }
    }

    #[test]
    #[ignore(windows)]
    #[should_fail]
    fn test_as_const_buf_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        do as_const_buf(v) |_buf, _i| {
            fail
        }
    }

    #[test]
    #[ignore(cfg(windows))]
    #[should_fail]
    fn test_as_mut_buf_fail() {
        let v = [mut (~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        do as_mut_buf(v) |_buf, _i| {
            fail
        }
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_copy_memory_oob() unsafe {
        let a = [mut 1, 2, 3, 4];
        let b = [1, 2, 3, 4, 5];
        raw::copy_memory(a, b, 5);
    }

}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
