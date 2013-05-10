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
use cmp::{Eq, Ord, TotalEq, TotalOrd, Ordering, Less, Equal, Greater};
use clone::Clone;
use old_iter::BaseIter;
use old_iter;
use iterator::Iterator;
use kinds::Copy;
use libc;
use option::{None, Option, Some};
use ptr::to_unsafe_ptr;
use ptr;
use sys;
use uint;
use unstable::intrinsics;
use vec;

#[cfg(not(test))] use cmp::Equiv;

pub mod rustrt {
    use libc;
    use sys;
    use vec::raw;

    #[abi = "cdecl"]
    pub extern {
        // These names are terrible. reserve_shared applies
        // to ~[] and reserve_shared_actual applies to @[].
        #[fast_ffi]
        unsafe fn vec_reserve_shared(t: *sys::TypeDesc,
                                     v: **raw::VecRepr,
                                     n: libc::size_t);
        #[fast_ffi]
        unsafe fn vec_reserve_shared_actual(t: *sys::TypeDesc,
                                            v: **raw::VecRepr,
                                            n: libc::size_t);
    }
}

/// Returns true if a vector contains no elements
pub fn is_empty<T>(v: &const [T]) -> bool {
    as_const_buf(v, |_p, len| len == 0u)
}

/// Returns true if two vectors have the same length
pub fn same_length<T, U>(xs: &const [T], ys: &const [U]) -> bool {
    xs.len() == ys.len()
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
#[inline]
pub fn reserve<T>(v: &mut ~[T], n: uint) {
    // Only make the (slow) call into the runtime if we have to
    use managed;
    if capacity(v) < n {
        unsafe {
            let ptr: **raw::VecRepr = cast::transmute(v);
            let td = sys::get_type_desc::<T>();
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
pub fn capacity<T>(v: &const ~[T]) -> uint {
    unsafe {
        let repr: **raw::VecRepr = transmute(v);
        (**repr).unboxed.alloc / sys::nonzero_size_of::<T>()
    }
}

/// Returns the length of a vector
#[inline(always)]
pub fn len<T>(v: &const [T]) -> uint {
    as_const_buf(v, |_p, len| len)
}

// A botch to tide us over until core and std are fully demuted.
pub fn uniq_len<T>(v: &const ~[T]) -> uint {
    unsafe {
        let v: &~[T] = transmute(v);
        as_const_buf(*v, |_p, len| len)
    }
}

/**
 * Creates and initializes an owned vector.
 *
 * Creates an owned vector of size `n_elts` and initializes the elements
 * to the value returned by the function `op`.
 */
pub fn from_fn<T>(n_elts: uint, op: old_iter::InitOp<T>) -> ~[T] {
    unsafe {
        let mut v = with_capacity(n_elts);
        do as_mut_buf(v) |p, _len| {
            let mut i: uint = 0u;
            while i < n_elts {
                intrinsics::move_val_init(&mut(*ptr::mut_offset(p, i)),
                                          op(i));
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
    from_fn(n_elts, |_i| copy t)
}

/// Creates a new unique vector with the same contents as the slice
pub fn from_slice<T:Copy>(t: &[T]) -> ~[T] {
    from_fn(t.len(), |i| t[i])
}

/// Creates a new vector with a capacity of `capacity`
pub fn with_capacity<T>(capacity: uint) -> ~[T] {
    let mut vec = ~[];
    reserve(&mut vec, capacity);
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
#[inline(always)]
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
#[inline(always)]
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
#[inline(always)]
pub fn build_sized_opt<A>(size: Option<uint>,
                          builder: &fn(push: &fn(v: A)))
                       -> ~[A] {
    build_sized(size.get_or_default(4), builder)
}

// Accessors

/// Returns the first element of a vector
pub fn head<'r,T>(v: &'r [T]) -> &'r T {
    if v.len() == 0 { fail!(~"head: empty vector") }
    &v[0]
}

/// Returns `Some(x)` where `x` is the first element of the slice `v`,
/// or `None` if the vector is empty.
pub fn head_opt<'r,T>(v: &'r [T]) -> Option<&'r T> {
    if v.len() == 0 { None } else { Some(&v[0]) }
}

/// Returns a vector containing all but the first element of a slice
pub fn tail<'r,T>(v: &'r [T]) -> &'r [T] { slice(v, 1, v.len()) }

/// Returns a vector containing all but the first `n` elements of a slice
pub fn tailn<'r,T>(v: &'r [T], n: uint) -> &'r [T] { slice(v, n, v.len()) }

/// Returns a vector containing all but the last element of a slice
pub fn init<'r,T>(v: &'r [T]) -> &'r [T] { slice(v, 0, v.len() - 1) }

/// Returns a vector containing all but the last `n' elements of a slice
pub fn initn<'r,T>(v: &'r [T], n: uint) -> &'r [T] {
    slice(v, 0, v.len() - n)
}

/// Returns the last element of the slice `v`, failing if the slice is empty.
pub fn last<'r,T>(v: &'r [T]) -> &'r T {
    if v.len() == 0 { fail!(~"last: empty vector") }
    &v[v.len() - 1]
}

/// Returns `Some(x)` where `x` is the last element of the slice `v`, or
/// `None` if the vector is empty.
pub fn last_opt<'r,T>(v: &'r [T]) -> Option<&'r T> {
    if v.len() == 0 { None } else { Some(&v[v.len() - 1]) }
}

/// Return a slice that points into another slice.
#[inline(always)]
pub fn slice<'r,T>(v: &'r [T], start: uint, end: uint) -> &'r [T] {
    assert!(start <= end);
    assert!(end <= len(v));
    do as_imm_buf(v) |p, _len| {
        unsafe {
            transmute((ptr::offset(p, start),
                       (end - start) * sys::nonzero_size_of::<T>()))
        }
    }
}

/// Return a slice that points into another slice.
#[inline(always)]
pub fn mut_slice<'r,T>(v: &'r mut [T], start: uint, end: uint)
                    -> &'r mut [T] {
    assert!(start <= end);
    assert!(end <= v.len());
    do as_mut_buf(v) |p, _len| {
        unsafe {
            transmute((ptr::mut_offset(p, start),
                       (end - start) * sys::nonzero_size_of::<T>()))
        }
    }
}

/// Return a slice that points into another slice.
#[inline(always)]
pub fn const_slice<'r,T>(v: &'r const [T], start: uint, end: uint)
                      -> &'r const [T] {
    assert!(start <= end);
    assert!(end <= len(v));
    do as_const_buf(v) |p, _len| {
        unsafe {
            transmute((ptr::const_offset(p, start),
                       (end - start) * sys::nonzero_size_of::<T>()))
        }
    }
}

/// Copies

/// Split the vector `v` by applying each element against the predicate `f`.
pub fn split<T:Copy>(v: &[T], f: &fn(t: &T) -> bool) -> ~[~[T]] {
    let ln = len(v);
    if (ln == 0u) { return ~[] }

    let mut start = 0u;
    let mut result = ~[];
    while start < ln {
        match position_between(v, start, ln, f) {
            None => break,
            Some(i) => {
                result.push(slice(v, start, i).to_vec());
                start = i + 1u;
            }
        }
    }
    result.push(slice(v, start, ln).to_vec());
    result
}

/**
 * Split the vector `v` by applying each element against the predicate `f` up
 * to `n` times.
 */
pub fn splitn<T:Copy>(v: &[T], n: uint, f: &fn(t: &T) -> bool) -> ~[~[T]] {
    let ln = len(v);
    if (ln == 0u) { return ~[] }

    let mut start = 0u;
    let mut count = n;
    let mut result = ~[];
    while start < ln && count > 0u {
        match position_between(v, start, ln, f) {
            None => break,
            Some(i) => {
                result.push(slice(v, start, i).to_vec());
                // Make sure to skip the separator.
                start = i + 1u;
                count -= 1u;
            }
        }
    }
    result.push(slice(v, start, ln).to_vec());
    result
}

/**
 * Reverse split the vector `v` by applying each element against the predicate
 * `f`.
 */
pub fn rsplit<T:Copy>(v: &[T], f: &fn(t: &T) -> bool) -> ~[~[T]] {
    let ln = len(v);
    if (ln == 0) { return ~[] }

    let mut end = ln;
    let mut result = ~[];
    while end > 0 {
        match rposition_between(v, 0, end, f) {
            None => break,
            Some(i) => {
                result.push(slice(v, i + 1, end).to_vec());
                end = i;
            }
        }
    }
    result.push(slice(v, 0u, end).to_vec());
    reverse(result);
    result
}

/**
 * Reverse split the vector `v` by applying each element against the predicate
 * `f` up to `n times.
 */
pub fn rsplitn<T:Copy>(v: &[T], n: uint, f: &fn(t: &T) -> bool) -> ~[~[T]] {
    let ln = len(v);
    if (ln == 0u) { return ~[] }

    let mut end = ln;
    let mut count = n;
    let mut result = ~[];
    while end > 0u && count > 0u {
        match rposition_between(v, 0u, end, f) {
            None => break,
            Some(i) => {
                result.push(slice(v, i + 1u, end).to_vec());
                // Make sure to skip the separator.
                end = i;
                count -= 1u;
            }
        }
    }
    result.push(slice(v, 0u, end).to_vec());
    reverse(result);
    result
}

/**
 * Partitions a vector into two new vectors: those that satisfies the
 * predicate, and those that do not.
 */
pub fn partition<T>(v: ~[T], f: &fn(&T) -> bool) -> (~[T], ~[T]) {
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
pub fn partitioned<T:Copy>(v: &[T], f: &fn(&T) -> bool) -> (~[T], ~[T]) {
    let mut lefts  = ~[];
    let mut rights = ~[];

    for each(v) |elt| {
        if f(elt) {
            lefts.push(*elt);
        } else {
            rights.push(*elt);
        }
    }

    (lefts, rights)
}

// Mutators

/// Removes the first element from a vector and return it
pub fn shift<T>(v: &mut ~[T]) -> T {
    unsafe {
        assert!(!v.is_empty());

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
        assert!(capacity(v) >= ln);
        // Pretend like we have the original length so we can use
        // the vector copy_memory to overwrite the hole we just made
        raw::set_len(&mut *v, ln);

        // Memcopy the head element (the one we want) to the location we just
        // popped. For the moment it unsafely exists at both the head and last
        // positions
        {
            let first_slice = slice(*v, 0, 1);
            let last_slice = slice(*v, next_ln, ln);
            raw::copy_memory(transmute(last_slice), first_slice, 1);
        }

        // Memcopy everything to the left one element
        {
            let init_slice = slice(*v, 0, next_ln);
            let tail_slice = slice(*v, 1, ln);
            raw::copy_memory(transmute(init_slice),
                             tail_slice,
                             next_ln);
        }

        // Set the new length. Now the vector is back to normal
        raw::set_len(&mut *v, next_ln);

        // Swap out the element we want from the end
        let vp = raw::to_mut_ptr(*v);
        let vp = ptr::mut_offset(vp, next_ln - 1);
        *vp <-> work_elt;

        work_elt
    }
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
    assert!(i <= len);

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
    assert!(i < len);

    let mut j = i;
    while j < len - 1 {
        v[j] <-> v[j + 1];
        j += 1;
    }
    v.pop()
}

pub fn consume<T>(mut v: ~[T], f: &fn(uint, v: T)) {
    unsafe {
        do as_mut_buf(v) |p, ln| {
            for uint::range(0, ln) |i| {
                // NB: This unsafe operation counts on init writing 0s to the
                // holes we create in the vector. That ensures that, if the
                // iterator fails then we won't try to clean up the consumed
                // elements during unwinding
                let mut x = intrinsics::init();
                let p = ptr::mut_offset(p, i);
                x <-> *p;
                f(i, x);
            }
        }

        raw::set_len(&mut v, 0);
    }
}

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
                let mut x = intrinsics::init();
                let p = ptr::mut_offset(p, i);
                x <-> *p;
                f(i, x);
            }
        }

        raw::set_len(&mut v, 0);
    }
}

/// Remove the last element from a vector and return it
#[cfg(not(stage0))]
pub fn pop<T>(v: &mut ~[T]) -> T {
    let ln = v.len();
    if ln == 0 {
        fail!(~"sorry, cannot vec::pop an empty vector")
    }
    let valptr = ptr::to_mut_unsafe_ptr(&mut v[ln - 1u]);
    unsafe {
        let mut val = intrinsics::uninit();
        val <-> *valptr;
        raw::set_len(v, ln - 1u);
        val
    }
}

#[cfg(stage0)]
pub fn pop<T>(v: &mut ~[T]) -> T {
    let ln = v.len();
    if ln == 0 {
        fail!(~"sorry, cannot vec::pop an empty vector")
    }
    let valptr = ptr::to_mut_unsafe_ptr(&mut v[ln - 1u]);
    unsafe {
        let mut val = intrinsics::init();
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
        fail!(fmt!("vec::swap_remove - index %u >= length %u", index, ln));
    }
    if index < ln - 1 {
        v[index] <-> v[ln - 1];
    }
    v.pop()
}

/// Append an element to a vector
#[inline(always)]
pub fn push<T>(v: &mut ~[T], initval: T) {
    unsafe {
        let repr: **raw::VecRepr = transmute(&mut *v);
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
    let repr: **mut raw::VecRepr = transmute(v);
    let fill = (**repr).unboxed.fill;
    (**repr).unboxed.fill += sys::nonzero_size_of::<T>();
    let p = to_unsafe_ptr(&((**repr).unboxed.data));
    let p = ptr::offset(p, fill) as *mut T;
    intrinsics::move_val_init(&mut(*p), initval);
}

#[inline(never)]
fn push_slow<T>(v: &mut ~[T], initval: T) {
    let new_len = v.len() + 1;
    reserve_at_least(&mut *v, new_len);
    unsafe { push_fast(v, initval) }
}

#[inline(always)]
pub fn push_all<T:Copy>(v: &mut ~[T], rhs: &const [T]) {
    let new_len = v.len() + rhs.len();
    reserve(&mut *v, new_len);

    for uint::range(0u, rhs.len()) |i| {
        push(&mut *v, unsafe { raw::get(rhs, i) })
    }
}

#[inline(always)]
#[cfg(not(stage0))]
pub fn push_all_move<T>(v: &mut ~[T], mut rhs: ~[T]) {
    let new_len = v.len() + rhs.len();
    reserve(&mut *v, new_len);
    unsafe {
        do as_mut_buf(rhs) |p, len| {
            for uint::range(0, len) |i| {
                let mut x = intrinsics::uninit();
                x <-> *ptr::mut_offset(p, i);
                push(&mut *v, x);
            }
        }
        raw::set_len(&mut rhs, 0);
    }
}

#[inline(always)]
#[cfg(stage0)]
pub fn push_all_move<T>(v: &mut ~[T], mut rhs: ~[T]) {
    let new_len = v.len() + rhs.len();
    reserve(&mut *v, new_len);
    unsafe {
        do as_mut_buf(rhs) |p, len| {
            for uint::range(0, len) |i| {
                let mut x = intrinsics::init();
                x <-> *ptr::mut_offset(p, i);
                push(&mut *v, x);
            }
        }
        raw::set_len(&mut rhs, 0);
    }
}

/// Shorten a vector, dropping excess elements.
#[cfg(not(stage0))]
pub fn truncate<T>(v: &mut ~[T], newlen: uint) {
    do as_mut_buf(*v) |p, oldlen| {
        assert!(newlen <= oldlen);
        unsafe {
            // This loop is optimized out for non-drop types.
            for uint::range(newlen, oldlen) |i| {
                let mut dropped = intrinsics::uninit();
                dropped <-> *ptr::mut_offset(p, i);
            }
        }
    }
    unsafe { raw::set_len(&mut *v, newlen); }
}

/// Shorten a vector, dropping excess elements.
#[cfg(stage0)]
pub fn truncate<T>(v: &mut ~[T], newlen: uint) {
    do as_mut_buf(*v) |p, oldlen| {
        assert!(newlen <= oldlen);
        unsafe {
            // This loop is optimized out for non-drop types.
            for uint::range(newlen, oldlen) |i| {
                let mut dropped = intrinsics::init();
                dropped <-> *ptr::mut_offset(p, i);
            }
        }
    }
    unsafe { raw::set_len(&mut *v, newlen); }
}

/**
 * Remove consecutive repeated elements from a vector; if the vector is
 * sorted, this removes all duplicates.
 */
#[cfg(not(stage0))]
pub fn dedup<T:Eq>(v: &mut ~[T]) {
    unsafe {
        if v.len() < 1 { return; }
        let mut last_written = 0, next_to_read = 1;
        do as_const_buf(*v) |p, ln| {
            // We have a mutable reference to v, so we can make arbitrary
            // changes. (cf. push and pop)
            let p = p as *mut T;
            // last_written < next_to_read <= ln
            while next_to_read < ln {
                // last_written < next_to_read < ln
                if *ptr::mut_offset(p, next_to_read) ==
                    *ptr::mut_offset(p, last_written) {
                    let mut dropped = intrinsics::uninit();
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
}

/**
 * Remove consecutive repeated elements from a vector; if the vector is
 * sorted, this removes all duplicates.
 */
#[cfg(stage0)]
pub fn dedup<T:Eq>(v: &mut ~[T]) {
    unsafe {
        if v.len() < 1 { return; }
        let mut last_written = 0, next_to_read = 1;
        do as_const_buf(*v) |p, ln| {
            // We have a mutable reference to v, so we can make arbitrary
            // changes. (cf. push and pop)
            let p = p as *mut T;
            // last_written < next_to_read <= ln
            while next_to_read < ln {
                // last_written < next_to_read < ln
                if *ptr::mut_offset(p, next_to_read) ==
                    *ptr::mut_offset(p, last_written) {
                    let mut dropped = intrinsics::init();
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
}


// Appending
#[inline(always)]
pub fn append<T:Copy>(lhs: ~[T], rhs: &const [T]) -> ~[T] {
    let mut v = lhs;
    v.push_all(rhs);
    v
}

#[inline(always)]
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
    reserve_at_least(&mut *v, new_len);
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
pub fn grow_fn<T>(v: &mut ~[T], n: uint, op: old_iter::InitOp<T>) {
    let new_len = v.len() + n;
    reserve_at_least(&mut *v, new_len);
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
    let mut result = with_capacity(len(v));
    for each(v) |elem| {
        result.push(f(elem));
    }
    result
}

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
    for each(v) |elem| { result.push_all_move(f(elem)); }
    result
}

/**
 * Apply a function to each pair of elements and return the results.
 * Equivalent to `map(zip(v0, v1), f)`.
 */
pub fn map_zip<T:Copy,U:Copy,V>(v0: &[T], v1: &[U],
                                  f: &fn(t: &T, v: &U) -> V) -> ~[V] {
    let v0_len = len(v0);
    if v0_len != len(v1) { fail!(); }
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
    for each(v) |elem| {
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
    for each(v) |elem| {
        if f(elem) { result.push(*elem); }
    }
    result
}

/**
 * Like `filter()`, but in place.  Preserves order of `v`.  Linear time.
 */
pub fn retain<T>(v: &mut ~[T], f: &fn(t: &T) -> bool) {
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
pub fn concat<T:Copy>(v: &[~[T]]) -> ~[T] {
    let mut r = ~[];
    for each(v) |inner| { r.push_all(*inner); }
    r
}

/// Concatenate a vector of vectors, placing a given separator between each
pub fn connect<T:Copy>(v: &[~[T]], sep: &T) -> ~[T] {
    let mut r: ~[T] = ~[];
    let mut first = true;
    for each(v) |inner| {
        if first { first = false; } else { r.push(*sep); }
        r.push_all(*inner);
    }
    r
}

/**
 * Reduces a vector from left to right.
 *
 * # Arguments
 * * `z` - initial accumulator value
 * * `v` - vector to iterate over
 * * `p` - a closure to operate on vector elements
 *
 * # Examples
 *
 * Sum all values in the vector [1, 2, 3]:
 *
 * ~~~
 * vec::foldl(0, [1, 2, 3], |a, b| a + *b);
 * ~~~
 *
 */
pub fn foldl<'a, T, U>(z: T, v: &'a [U], p: &fn(t: T, u: &'a U) -> T) -> T {
    let mut accum = z;
    let mut i = 0;
    let l = v.len();
    while i < l {
        // Use a while loop so that liveness analysis can handle moving
        // the accumulator.
        accum = p(accum, &v[i]);
        i += 1;
    }
    accum
}

/**
 * Reduces a vector from right to left. Note that the argument order is
 * reversed compared to `foldl` to reflect the order they are provided to
 * the closure.
 *
 * # Arguments
 * * `v` - vector to iterate over
 * * `z` - initial accumulator value
 * * `p` - a closure to do operate on vector elements
 *
 * # Examples
 *
 * Sum all values in the vector [1, 2, 3]:
 *
 * ~~~
 * vec::foldr([1, 2, 3], 0, |a, b| a + *b);
 * ~~~
 *
 */
pub fn foldr<'a, T, U>(v: &'a [T], mut z: U, p: &fn(t: &'a T, u: U) -> U) -> U {
    let mut i = v.len();
    while i > 0 {
        i -= 1;
        z = p(&v[i], z);
    }
    return z;
}

/**
 * Return true if a predicate matches any elements
 *
 * If the vector contains no elements then false is returned.
 */
pub fn any<T>(v: &[T], f: &fn(t: &T) -> bool) -> bool {
    for each(v) |elem| { if f(elem) { return true; } }
    false
}

/**
 * Return true if a predicate matches any elements in both vectors.
 *
 * If the vectors contains no elements then false is returned.
 */
pub fn any2<T, U>(v0: &[T], v1: &[U],
                   f: &fn(a: &T, b: &U) -> bool) -> bool {
    let v0_len = len(v0);
    let v1_len = len(v1);
    let mut i = 0u;
    while i < v0_len && i < v1_len {
        if f(&v0[i], &v1[i]) { return true; };
        i += 1u;
    }
    false
}

/**
 * Return true if a predicate matches all elements
 *
 * If the vector contains no elements then true is returned.
 */
pub fn all<T>(v: &[T], f: &fn(t: &T) -> bool) -> bool {
    for each(v) |elem| { if !f(elem) { return false; } }
    true
}

/**
 * Return true if a predicate matches all elements
 *
 * If the vector contains no elements then true is returned.
 */
pub fn alli<T>(v: &[T], f: &fn(uint, t: &T) -> bool) -> bool {
    for eachi(v) |i, elem| { if !f(i, elem) { return false; } }
    true
}

/**
 * Return true if a predicate matches all elements in both vectors.
 *
 * If the vectors are not the same size then false is returned.
 */
pub fn all2<T, U>(v0: &[T], v1: &[U],
                   f: &fn(t: &T, u: &U) -> bool) -> bool {
    let v0_len = len(v0);
    if v0_len != len(v1) { return false; }
    let mut i = 0u;
    while i < v0_len { if !f(&v0[i], &v1[i]) { return false; }; i += 1u; }
    true
}

/// Return true if a vector contains an element with the given value
pub fn contains<T:Eq>(v: &[T], x: &T) -> bool {
    for each(v) |elt| { if *x == *elt { return true; } }
    false
}

/// Returns the number of elements that are equal to a given value
pub fn count<T:Eq>(v: &[T], x: &T) -> uint {
    let mut cnt = 0u;
    for each(v) |elt| { if *x == *elt { cnt += 1u; } }
    cnt
}

/**
 * Search for the first element that matches a given predicate
 *
 * Apply function `f` to each element of `v`, starting from the first.
 * When function `f` returns true then an option containing the element
 * is returned. If `f` matches no elements then none is returned.
 */
pub fn find<T:Copy>(v: &[T], f: &fn(t: &T) -> bool) -> Option<T> {
    find_between(v, 0u, len(v), f)
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
    position_between(v, start, end, f).map(|i| v[*i])
}

/**
 * Search for the last element that matches a given predicate
 *
 * Apply function `f` to each element of `v` in reverse order. When function
 * `f` returns true then an option containing the element is returned. If `f`
 * matches no elements then none is returned.
 */
pub fn rfind<T:Copy>(v: &[T], f: &fn(t: &T) -> bool) -> Option<T> {
    rfind_between(v, 0u, len(v), f)
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
    rposition_between(v, start, end, f).map(|i| v[*i])
}

/// Find the first index containing a matching value
pub fn position_elem<T:Eq>(v: &[T], x: &T) -> Option<uint> {
    position(v, |y| *x == *y)
}

/**
 * Find the first index matching some predicate
 *
 * Apply function `f` to each element of `v`.  When function `f` returns true
 * then an option containing the index is returned. If `f` matches no elements
 * then none is returned.
 */
pub fn position<T>(v: &[T], f: &fn(t: &T) -> bool) -> Option<uint> {
    position_between(v, 0u, len(v), f)
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
    assert!(end <= len(v));
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
pub fn rposition_between<T>(v: &[T], start: uint, end: uint,
                             f: &fn(t: &T) -> bool) -> Option<uint> {
    assert!(start <= end);
    assert!(end <= len(v));
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
    let mut ts = ~[], us = ~[];
    for each(v) |p| {
        let (t, u) = *p;
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
    let mut ts = ~[], us = ~[];
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
pub fn zip_slice<T:Copy,U:Copy>(v: &const [T], u: &const [U])
        -> ~[(T, U)] {
    let mut zipped = ~[];
    let sz = len(v);
    let mut i = 0u;
    assert!(sz == len(u));
    while i < sz {
        zipped.push((v[i], u[i]));
        i += 1u;
    }
    zipped
}

/**
 * Convert two vectors to a vector of pairs.
 *
 * Returns a vector of tuples, where the i-th tuple contains contains the
 * i-th elements from each of the input vectors.
 */
pub fn zip<T, U>(mut v: ~[T], mut u: ~[U]) -> ~[(T, U)] {
    let mut i = len(v);
    assert!(i == len(u));
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
pub fn swap<T>(v: &mut [T], a: uint, b: uint) {
    v[a] <-> v[b];
}

/// Reverse the order of elements in a vector, in place
pub fn reverse<T>(v: &mut [T]) {
    let mut i: uint = 0;
    let ln = len::<T>(v);
    while i < ln / 2 { v[i] <-> v[ln - i - 1]; i += 1; }
}

/// Returns a vector with the order of elements reversed
pub fn reversed<T:Copy>(v: &const [T]) -> ~[T] {
    let mut rs: ~[T] = ~[];
    let mut i = len::<T>(v);
    if i == 0 { return (rs); } else { i -= 1; }
    while i != 0 { rs.push(v[i]); i -= 1; }
    rs.push(v[0]);
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
pub fn each<'r,T>(v: &'r [T], f: &fn(&'r T) -> bool) {
    //             ^^^^
    // NB---this CANNOT be &const [T]!  The reason
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
pub fn each_mut<'r,T>(v: &'r mut [T], f: &fn(elem: &'r mut T) -> bool) {
    do vec::as_mut_buf(v) |p, n| {
        let mut n = n;
        let mut p = p;
        while n > 0 {
            unsafe {
                let q: &'r mut T = cast::transmute_mut_region(&mut *p);
                if !f(q) {
                    break;
                }
                p = p.offset(1);
            }
            n -= 1;
        }
    }
}

/// Like `each()`, but for the case where you have a vector that *may or may
/// not* have mutable contents.
#[inline(always)]
pub fn each_const<T>(v: &const [T], f: &fn(elem: &const T) -> bool) {
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
pub fn eachi<'r,T>(v: &'r [T], f: &fn(uint, v: &'r T) -> bool) {
    let mut i = 0;
    for each(v) |p| {
        if !f(i, p) { return; }
        i += 1;
    }
}

/**
 * Iterates over a mutable vector's elements and indices
 *
 * Return true to continue, false to break.
 */
#[inline(always)]
pub fn eachi_mut<'r,T>(v: &'r mut [T], f: &fn(uint, v: &'r mut T) -> bool) {
    let mut i = 0;
    for each_mut(v) |p| {
        if !f(i, p) {
            return;
        }
        i += 1;
    }
}

/**
 * Iterates over a vector's elements in reverse
 *
 * Return true to continue, false to break.
 */
#[inline(always)]
pub fn each_reverse<'r,T>(v: &'r [T], blk: &fn(v: &'r T) -> bool) {
    eachi_reverse(v, |_i, v| blk(v))
}

/**
 * Iterates over a vector's elements and indices in reverse
 *
 * Return true to continue, false to break.
 */
#[inline(always)]
pub fn eachi_reverse<'r,T>(v: &'r [T], blk: &fn(i: uint, v: &'r T) -> bool) {
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
pub fn each2<U, T>(v1: &[U], v2: &[T], f: &fn(u: &U, t: &T) -> bool) {
    assert!(len(v1) == len(v2));
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
pub fn each_permutation<T:Copy>(v: &[T], put: &fn(ts: &[T]) -> bool) {
    let ln = len(v);
    if ln <= 1 {
        put(v);
    } else {
        // This does not seem like the most efficient implementation.  You
        // could make far fewer copies if you put your mind to it.
        let mut i = 0u;
        while i < ln {
            let elt = v[i];
            let mut rest = slice(v, 0u, i).to_vec();
            rest.push_all(const_slice(v, i+1u, ln));
            for each_permutation(rest) |permutation| {
                if !put(append(~[elt], permutation)) {
                    return;
                }
            }
            i += 1u;
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
 * ~~~
 * for windowed(2, &[1,2,3,4]) |v| {
 *     io::println(fmt!("%?", v));
 * }
 * ~~~
 *
 */
pub fn windowed<'r, T>(n: uint, v: &'r [T], it: &fn(&'r [T]) -> bool) {
    assert!(1u <= n);
    if n > v.len() { return; }
    for uint::range(0, v.len() - n + 1) |i| {
        if !it(v.slice(i, i + n)) { return }
    }
}

/**
 * Work with the buffer of a vector.
 *
 * Allows for unsafe manipulation of vector contents, which is useful for
 * foreign interop.
 */
#[inline(always)]
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

/// Similar to `as_imm_buf` but passing a `*const T`
#[inline(always)]
pub fn as_const_buf<T,U>(s: &const [T], f: &fn(*const T, uint) -> U) -> U {
    unsafe {
        let v : *(*const T,uint) = transmute(&s);
        let (buf,len) = *v;
        f(buf, len / sys::nonzero_size_of::<T>())
    }
}

/// Similar to `as_imm_buf` but passing a `*mut T`
#[inline(always)]
pub fn as_mut_buf<T,U>(s: &mut [T], f: &fn(*mut T, uint) -> U) -> U {
    unsafe {
        let v : *(*mut T,uint) = transmute(&s);
        let (buf,len) = *v;
        f(buf, len / sys::nonzero_size_of::<T>())
    }
}

// Equality

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
    #[inline(always)]
    fn eq(&self, other: & &'self [T]) -> bool { eq(*self, *other) }
    #[inline(always)]
    fn ne(&self, other: & &'self [T]) -> bool { !self.eq(other) }
}

#[cfg(not(test))]
impl<T:Eq> Eq for ~[T] {
    #[inline(always)]
    fn eq(&self, other: &~[T]) -> bool { eq(*self, *other) }
    #[inline(always)]
    fn ne(&self, other: &~[T]) -> bool { !self.eq(other) }
}

#[cfg(not(test))]
impl<T:Eq> Eq for @[T] {
    #[inline(always)]
    fn eq(&self, other: &@[T]) -> bool { eq(*self, *other) }
    #[inline(always)]
    fn ne(&self, other: &@[T]) -> bool { !self.eq(other) }
}

#[cfg(not(test))]
impl<'self,T:TotalEq> TotalEq for &'self [T] {
    #[inline(always)]
    fn equals(&self, other: & &'self [T]) -> bool { equals(*self, *other) }
}

#[cfg(not(test))]
impl<T:TotalEq> TotalEq for ~[T] {
    #[inline(always)]
    fn equals(&self, other: &~[T]) -> bool { equals(*self, *other) }
}

#[cfg(not(test))]
impl<T:TotalEq> TotalEq for @[T] {
    #[inline(always)]
    fn equals(&self, other: &@[T]) -> bool { equals(*self, *other) }
}

#[cfg(not(test))]
impl<'self,T:Eq> Equiv<~[T]> for &'self [T] {
    #[inline(always)]
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
    #[inline(always)]
    fn cmp(&self, other: & &'self [T]) -> Ordering { cmp(*self, *other) }
}

#[cfg(not(test))]
impl<T: TotalOrd> TotalOrd for ~[T] {
    #[inline(always)]
    fn cmp(&self, other: &~[T]) -> Ordering { cmp(*self, *other) }
}

#[cfg(not(test))]
impl<T: TotalOrd> TotalOrd for @[T] {
    #[inline(always)]
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
    #[inline(always)]
    fn lt(&self, other: & &'self [T]) -> bool { lt((*self), (*other)) }
    #[inline(always)]
    fn le(&self, other: & &'self [T]) -> bool { le((*self), (*other)) }
    #[inline(always)]
    fn ge(&self, other: & &'self [T]) -> bool { ge((*self), (*other)) }
    #[inline(always)]
    fn gt(&self, other: & &'self [T]) -> bool { gt((*self), (*other)) }
}

#[cfg(not(test))]
impl<T:Ord> Ord for ~[T] {
    #[inline(always)]
    fn lt(&self, other: &~[T]) -> bool { lt((*self), (*other)) }
    #[inline(always)]
    fn le(&self, other: &~[T]) -> bool { le((*self), (*other)) }
    #[inline(always)]
    fn ge(&self, other: &~[T]) -> bool { ge((*self), (*other)) }
    #[inline(always)]
    fn gt(&self, other: &~[T]) -> bool { gt((*self), (*other)) }
}

#[cfg(not(test))]
impl<T:Ord> Ord for @[T] {
    #[inline(always)]
    fn lt(&self, other: &@[T]) -> bool { lt((*self), (*other)) }
    #[inline(always)]
    fn le(&self, other: &@[T]) -> bool { le((*self), (*other)) }
    #[inline(always)]
    fn ge(&self, other: &@[T]) -> bool { ge((*self), (*other)) }
    #[inline(always)]
    fn gt(&self, other: &@[T]) -> bool { gt((*self), (*other)) }
}

#[cfg(not(test))]
pub mod traits {
    use kinds::Copy;
    use ops::Add;
    use vec::append;

    impl<'self,T:Copy> Add<&'self const [T],~[T]> for ~[T] {
        #[inline(always)]
        fn add(&self, rhs: & &'self const [T]) -> ~[T] {
            append(copy *self, (*rhs))
        }
    }
}

impl<'self,T> Container for &'self const [T] {
    /// Returns true if a vector contains no elements
    #[inline]
    fn is_empty(&const self) -> bool { is_empty(*self) }

    /// Returns the length of a vector
    #[inline]
    fn len(&const self) -> uint { len(*self) }
}

pub trait CopyableVector<T> {
    fn to_owned(&self) -> ~[T];
}

/// Extension methods for vectors
impl<'self,T:Copy> CopyableVector<T> for &'self [T] {
    /// Returns a copy of `v`.
    #[inline]
    fn to_owned(&self) -> ~[T] {
        let mut result = ~[];
        reserve(&mut result, self.len());
        for self.each |e| {
            result.push(copy *e);
        }
        result

    }
}

pub trait ImmutableVector<'self, T> {
    fn slice(&self, start: uint, end: uint) -> &'self [T];
    fn iter(self) -> VecIterator<'self, T>;
    fn head(&self) -> &'self T;
    fn head_opt(&self) -> Option<&'self T>;
    fn tail(&self) -> &'self [T];
    fn tailn(&self, n: uint) -> &'self [T];
    fn init(&self) -> &'self [T];
    fn initn(&self, n: uint) -> &'self [T];
    fn last(&self) -> &'self T;
    fn last_opt(&self) -> Option<&'self T>;
    fn each_reverse(&self, blk: &fn(&T) -> bool);
    fn eachi_reverse(&self, blk: &fn(uint, &T) -> bool);
    fn foldr<'a, U>(&'a self, z: U, p: &fn(t: &'a T, u: U) -> U) -> U;
    fn map<U>(&self, f: &fn(t: &T) -> U) -> ~[U];
    fn mapi<U>(&self, f: &fn(uint, t: &T) -> U) -> ~[U];
    fn map_r<U>(&self, f: &fn(x: &T) -> U) -> ~[U];
    fn alli(&self, f: &fn(uint, t: &T) -> bool) -> bool;
    fn flat_map<U>(&self, f: &fn(t: &T) -> ~[U]) -> ~[U];
    fn filter_mapped<U:Copy>(&self, f: &fn(t: &T) -> Option<U>) -> ~[U];
    unsafe fn unsafe_ref(&self, index: uint) -> *T;
}

/// Extension methods for vectors
impl<'self,T> ImmutableVector<'self, T> for &'self [T] {
    /// Return a slice that points into another slice.
    #[inline]
    fn slice(&self, start: uint, end: uint) -> &'self [T] {
        slice(*self, start, end)
    }

    #[inline]
    fn iter(self) -> VecIterator<'self, T> {
        unsafe {
            let p = vec::raw::to_ptr(self);
            VecIterator{ptr: p, end: p.offset(self.len()),
                        lifetime: cast::transmute(p)}
        }
    }

    /// Returns the first element of a vector, failing if the vector is empty.
    #[inline]
    fn head(&self) -> &'self T { head(*self) }

    /// Returns the first element of a vector
    #[inline]
    fn head_opt(&self) -> Option<&'self T> { head_opt(*self) }

    /// Returns all but the first element of a vector
    #[inline]
    fn tail(&self) -> &'self [T] { tail(*self) }

    /// Returns all but the first `n' elements of a vector
    #[inline]
    fn tailn(&self, n: uint) -> &'self [T] { tailn(*self, n) }

    /// Returns all but the last elemnt of a vector
    #[inline]
    fn init(&self) -> &'self [T] { init(*self) }

    /// Returns all but the last `n' elemnts of a vector
    #[inline]
    fn initn(&self, n: uint) -> &'self [T] { initn(*self, n) }

    /// Returns the last element of a `v`, failing if the vector is empty.
    #[inline]
    fn last(&self) -> &'self T { last(*self) }

    /// Returns the last element of a `v`, failing if the vector is empty.
    #[inline]
    fn last_opt(&self) -> Option<&'self T> { last_opt(*self) }

    /// Iterates over a vector's elements in reverse.
    #[inline]
    fn each_reverse(&self, blk: &fn(&T) -> bool) {
        each_reverse(*self, blk)
    }

    /// Iterates over a vector's elements and indices in reverse.
    #[inline]
    fn eachi_reverse(&self, blk: &fn(uint, &T) -> bool) {
        eachi_reverse(*self, blk)
    }

    /// Reduce a vector from right to left
    #[inline]
    fn foldr<'a, U>(&'a self, z: U, p: &fn(t: &'a T, u: U) -> U) -> U {
        foldr(*self, z, p)
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
     * Returns true if the function returns true for all elements.
     *
     *     If the vector is empty, true is returned.
     */
    fn alli(&self, f: &fn(uint, t: &T) -> bool) -> bool {
        alli(*self, f)
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
    #[inline(always)]
    unsafe fn unsafe_ref(&self, index: uint) -> *T {
        let (ptr, _): (*T, uint) = transmute(*self);
        ptr.offset(index)
    }
}

pub trait ImmutableEqVector<T:Eq> {
    fn position(&self, f: &fn(t: &T) -> bool) -> Option<uint>;
    fn position_elem(&self, t: &T) -> Option<uint>;
    fn rposition(&self, f: &fn(t: &T) -> bool) -> Option<uint>;
    fn rposition_elem(&self, t: &T) -> Option<uint>;
}

impl<'self,T:Eq> ImmutableEqVector<T> for &'self [T] {
    /**
     * Find the first index matching some predicate
     *
     * Apply function `f` to each element of `v`.  When function `f` returns
     * true then an option containing the index is returned. If `f` matches no
     * elements then none is returned.
     */
    #[inline]
    fn position(&self, f: &fn(t: &T) -> bool) -> Option<uint> {
        position(*self, f)
    }

    /// Find the first index containing a matching value
    #[inline]
    fn position_elem(&self, x: &T) -> Option<uint> {
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
    fn rposition(&self, f: &fn(t: &T) -> bool) -> Option<uint> {
        rposition(*self, f)
    }

    /// Find the last index containing a matching value
    #[inline]
    fn rposition_elem(&self, t: &T) -> Option<uint> {
        rposition_elem(*self, t)
    }
}

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
        partitioned(*self, f)
    }

    /// Returns the element at the given index, without doing bounds checking.
    #[inline(always)]
    unsafe fn unsafe_get(&self, index: uint) -> T {
        *self.unsafe_ref(index)
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
    fn retain(&mut self, f: &fn(t: &T) -> bool);
    fn consume(self, f: &fn(uint, v: T));
    fn consume_reverse(self, f: &fn(uint, v: T));
    fn filter(self, f: &fn(t: &T) -> bool) -> ~[T];
    fn partition(self, f: &fn(&T) -> bool) -> (~[T], ~[T]);
    fn grow_fn(&mut self, n: uint, op: old_iter::InitOp<T>);
}

impl<T> OwnedVector<T> for ~[T] {
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
    fn retain(&mut self, f: &fn(t: &T) -> bool) {
        retain(self, f);
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
        partition(self, f)
    }

    #[inline]
    fn grow_fn(&mut self, n: uint, op: old_iter::InitOp<T>) {
        grow_fn(self, n, op);
    }
}

impl<T> Mutable for ~[T] {
    /// Clear the vector, removing all values.
    fn clear(&mut self) { self.truncate(0) }
}

pub trait OwnedCopyableVector<T:Copy> {
    fn push_all(&mut self, rhs: &const [T]);
    fn grow(&mut self, n: uint, initval: &T);
    fn grow_set(&mut self, index: uint, initval: &T, val: T);
}

impl<T:Copy> OwnedCopyableVector<T> for ~[T] {
    #[inline]
    fn push_all(&mut self, rhs: &const [T]) {
        push_all(self, rhs);
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

trait OwnedEqVector<T:Eq> {
    fn dedup(&mut self);
}

impl<T:Eq> OwnedEqVector<T> for ~[T] {
    #[inline]
    fn dedup(&mut self) {
        dedup(self)
    }
}

pub trait MutableVector<T> {
    unsafe fn unsafe_mut_ref(&self, index: uint) -> *mut T;
    unsafe fn unsafe_set(&self, index: uint, val: T);
}

impl<'self,T> MutableVector<T> for &'self mut [T] {
    #[inline(always)]
    unsafe fn unsafe_mut_ref(&self, index: uint) -> *mut T {
        let pair_ptr: &(*mut T, uint) = transmute(self);
        let (ptr, _) = *pair_ptr;
        ptr.offset(index)
    }

    #[inline(always)]
    unsafe fn unsafe_set(&self, index: uint, val: T) {
        *self.unsafe_mut_ref(index) = val;
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
    use vec::{UnboxedVecRepr, as_const_buf, as_mut_buf, len, with_capacity};

    /// The internal representation of a (boxed) vector
    pub struct VecRepr {
        box_header: managed::raw::BoxHeaderRepr,
        unboxed: UnboxedVecRepr
    }

    pub struct SliceRepr {
        data: *u8,
        len: uint
    }

    /**
     * Sets the length of a vector
     *
     * This will explicitly set the size of the vector, without actually
     * modifing its buffers, so it is up to the caller to ensure that
     * the vector is actually the specified size.
     */
    #[inline(always)]
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
    #[inline(always)]
    pub unsafe fn to_ptr<T>(v: &[T]) -> *T {
        let repr: **SliceRepr = transmute(&v);
        transmute(&((**repr).data))
    }

    /** see `to_ptr()` */
    #[inline(always)]
    pub unsafe fn to_const_ptr<T>(v: &const [T]) -> *const T {
        let repr: **SliceRepr = transmute(&v);
        transmute(&((**repr).data))
    }

    /** see `to_ptr()` */
    #[inline(always)]
    pub unsafe fn to_mut_ptr<T>(v: &mut [T]) -> *mut T {
        let repr: **SliceRepr = transmute(&v);
        transmute(&((**repr).data))
    }

    /**
     * Form a slice from a pointer and length (as a number of units,
     * not bytes).
     */
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    pub unsafe fn get<T:Copy>(v: &const [T], i: uint) -> T {
        as_const_buf(v, |p, _len| *ptr::const_offset(p, i))
    }

    /**
     * Unchecked vector index assignment.  Does not drop the
     * old value and hence is only suitable when the vector
     * is newly allocated.
     */
    #[inline(always)]
    pub unsafe fn init_elem<T>(v: &mut [T], i: uint, val: T) {
        let mut box = Some(val);
        do as_mut_buf(v) |p, _len| {
            let mut box2 = None;
            box2 <-> box;
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
    #[inline(always)]
    pub unsafe fn copy_memory<T>(dst: &mut [T], src: &const [T],
                                 count: uint) {
        assert!(dst.len() >= count);
        assert!(src.len() >= count);

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
    use vec::raw;
    use vec;

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
    #[inline(always)]
    pub fn copy_memory(dst: &mut [u8], src: &const [u8], count: uint) {
        // Bound checks are done at vec::raw::copy_memory.
        unsafe { vec::raw::copy_memory(dst, src, count) }
    }
}

// ___________________________________________________________________________
// ITERATION TRAIT METHODS

impl<'self,A> old_iter::BaseIter<A> for &'self [A] {
    #[inline(always)]
    fn each<'a>(&'a self, blk: &fn(v: &'a A) -> bool) { each(*self, blk) }
    #[inline(always)]
    fn size_hint(&self) -> Option<uint> { Some(self.len()) }
}

// FIXME(#4148): This should be redundant
impl<A> old_iter::BaseIter<A> for ~[A] {
    #[inline(always)]
    fn each<'a>(&'a self, blk: &fn(v: &'a A) -> bool) { each(*self, blk) }
    #[inline(always)]
    fn size_hint(&self) -> Option<uint> { Some(self.len()) }
}

// FIXME(#4148): This should be redundant
impl<A> old_iter::BaseIter<A> for @[A] {
    #[inline(always)]
    fn each<'a>(&'a self, blk: &fn(v: &'a A) -> bool) { each(*self, blk) }
    #[inline(always)]
    fn size_hint(&self) -> Option<uint> { Some(self.len()) }
}

impl<'self,A> old_iter::MutableIter<A> for &'self mut [A] {
    #[inline(always)]
    fn each_mut<'a>(&'a mut self, blk: &fn(v: &'a mut A) -> bool) {
        each_mut(*self, blk)
    }
}

// FIXME(#4148): This should be redundant
impl<A> old_iter::MutableIter<A> for ~[A] {
    #[inline(always)]
    fn each_mut<'a>(&'a mut self, blk: &fn(v: &'a mut A) -> bool) {
        each_mut(*self, blk)
    }
}

// FIXME(#4148): This should be redundant
impl<A> old_iter::MutableIter<A> for @mut [A] {
    #[inline(always)]
    fn each_mut(&mut self, blk: &fn(v: &mut A) -> bool) {
        each_mut(*self, blk)
    }
}

impl<'self,A> old_iter::ExtendedIter<A> for &'self [A] {
    pub fn eachi(&self, blk: &fn(uint, v: &A) -> bool) {
        old_iter::eachi(self, blk)
    }
    pub fn all(&self, blk: &fn(&A) -> bool) -> bool {
        old_iter::all(self, blk)
    }
    pub fn any(&self, blk: &fn(&A) -> bool) -> bool {
        old_iter::any(self, blk)
    }
    pub fn foldl<B>(&self, b0: B, blk: &fn(&B, &A) -> B) -> B {
        old_iter::foldl(self, b0, blk)
    }
    pub fn position(&self, f: &fn(&A) -> bool) -> Option<uint> {
        old_iter::position(self, f)
    }
    fn map_to_vec<B>(&self, op: &fn(&A) -> B) -> ~[B] {
        old_iter::map_to_vec(self, op)
    }
    fn flat_map_to_vec<B,IB:BaseIter<B>>(&self, op: &fn(&A) -> IB)
        -> ~[B] {
        old_iter::flat_map_to_vec(self, op)
    }
}

impl<'self,A> old_iter::ExtendedMutableIter<A> for &'self mut [A] {
    #[inline(always)]
    pub fn eachi_mut(&mut self, blk: &fn(uint, v: &mut A) -> bool) {
        eachi_mut(*self, blk)
    }
}

// FIXME(#4148): This should be redundant
impl<A> old_iter::ExtendedIter<A> for ~[A] {
    pub fn eachi(&self, blk: &fn(uint, v: &A) -> bool) {
        old_iter::eachi(self, blk)
    }
    pub fn all(&self, blk: &fn(&A) -> bool) -> bool {
        old_iter::all(self, blk)
    }
    pub fn any(&self, blk: &fn(&A) -> bool) -> bool {
        old_iter::any(self, blk)
    }
    pub fn foldl<B>(&self, b0: B, blk: &fn(&B, &A) -> B) -> B {
        old_iter::foldl(self, b0, blk)
    }
    pub fn position(&self, f: &fn(&A) -> bool) -> Option<uint> {
        old_iter::position(self, f)
    }
    fn map_to_vec<B>(&self, op: &fn(&A) -> B) -> ~[B] {
        old_iter::map_to_vec(self, op)
    }
    fn flat_map_to_vec<B,IB:BaseIter<B>>(&self, op: &fn(&A) -> IB)
        -> ~[B] {
        old_iter::flat_map_to_vec(self, op)
    }
}

// FIXME(#4148): This should be redundant
impl<A> old_iter::ExtendedIter<A> for @[A] {
    pub fn eachi(&self, blk: &fn(uint, v: &A) -> bool) {
        old_iter::eachi(self, blk)
    }
    pub fn all(&self, blk: &fn(&A) -> bool) -> bool {
        old_iter::all(self, blk)
    }
    pub fn any(&self, blk: &fn(&A) -> bool) -> bool {
        old_iter::any(self, blk)
    }
    pub fn foldl<B>(&self, b0: B, blk: &fn(&B, &A) -> B) -> B {
        old_iter::foldl(self, b0, blk)
    }
    pub fn position(&self, f: &fn(&A) -> bool) -> Option<uint> {
        old_iter::position(self, f)
    }
    fn map_to_vec<B>(&self, op: &fn(&A) -> B) -> ~[B] {
        old_iter::map_to_vec(self, op)
    }
    fn flat_map_to_vec<B,IB:BaseIter<B>>(&self, op: &fn(&A) -> IB)
        -> ~[B] {
        old_iter::flat_map_to_vec(self, op)
    }
}

impl<'self,A:Eq> old_iter::EqIter<A> for &'self [A] {
    pub fn contains(&self, x: &A) -> bool { old_iter::contains(self, x) }
    pub fn count(&self, x: &A) -> uint { old_iter::count(self, x) }
}

// FIXME(#4148): This should be redundant
impl<A:Eq> old_iter::EqIter<A> for ~[A] {
    pub fn contains(&self, x: &A) -> bool { old_iter::contains(self, x) }
    pub fn count(&self, x: &A) -> uint { old_iter::count(self, x) }
}

// FIXME(#4148): This should be redundant
impl<A:Eq> old_iter::EqIter<A> for @[A] {
    pub fn contains(&self, x: &A) -> bool { old_iter::contains(self, x) }
    pub fn count(&self, x: &A) -> uint { old_iter::count(self, x) }
}

impl<'self,A:Copy> old_iter::CopyableIter<A> for &'self [A] {
    fn filter_to_vec(&self, pred: &fn(&A) -> bool) -> ~[A] {
        old_iter::filter_to_vec(self, pred)
    }
    fn to_vec(&self) -> ~[A] { old_iter::to_vec(self) }
    pub fn find(&self, f: &fn(&A) -> bool) -> Option<A> {
        old_iter::find(self, f)
    }
}

// FIXME(#4148): This should be redundant
impl<A:Copy> old_iter::CopyableIter<A> for ~[A] {
    fn filter_to_vec(&self, pred: &fn(&A) -> bool) -> ~[A] {
        old_iter::filter_to_vec(self, pred)
    }
    fn to_vec(&self) -> ~[A] { old_iter::to_vec(self) }
    pub fn find(&self, f: &fn(&A) -> bool) -> Option<A> {
        old_iter::find(self, f)
    }
}

// FIXME(#4148): This should be redundant
impl<A:Copy> old_iter::CopyableIter<A> for @[A] {
    fn filter_to_vec(&self, pred: &fn(&A) -> bool) -> ~[A] {
        old_iter::filter_to_vec(self, pred)
    }
    fn to_vec(&self) -> ~[A] { old_iter::to_vec(self) }
    pub fn find(&self, f: &fn(&A) -> bool) -> Option<A> {
        old_iter::find(self, f)
    }
}

impl<'self,A:Copy + Ord> old_iter::CopyableOrderedIter<A> for &'self [A] {
    fn min(&self) -> A { old_iter::min(self) }
    fn max(&self) -> A { old_iter::max(self) }
}

// FIXME(#4148): This should be redundant
impl<A:Copy + Ord> old_iter::CopyableOrderedIter<A> for ~[A] {
    fn min(&self) -> A { old_iter::min(self) }
    fn max(&self) -> A { old_iter::max(self) }
}

// FIXME(#4148): This should be redundant
impl<A:Copy + Ord> old_iter::CopyableOrderedIter<A> for @[A] {
    fn min(&self) -> A { old_iter::min(self) }
    fn max(&self) -> A { old_iter::max(self) }
}

impl<'self,A:Copy> old_iter::CopyableNonstrictIter<A> for &'self [A] {
    fn each_val(&const self, f: &fn(A) -> bool) {
        let mut i = 0;
        while i < self.len() {
            if !f(copy self[i]) { break; }
            i += 1;
        }
    }
}

// FIXME(#4148): This should be redundant
impl<A:Copy> old_iter::CopyableNonstrictIter<A> for ~[A] {
    fn each_val(&const self, f: &fn(A) -> bool) {
        let mut i = 0;
        while i < uniq_len(self) {
            if !f(copy self[i]) { break; }
            i += 1;
        }
    }
}

// FIXME(#4148): This should be redundant
impl<A:Copy> old_iter::CopyableNonstrictIter<A> for @[A] {
    fn each_val(&const self, f: &fn(A) -> bool) {
        let mut i = 0;
        while i < self.len() {
            if !f(copy self[i]) { break; }
            i += 1;
        }
    }
}

impl<A:Clone> Clone for ~[A] {
    #[inline]
    fn clone(&self) -> ~[A] {
        self.map(|item| item.clone())
    }
}

// could be implemented with &[T] with .slice(), but this avoids bounds checks
pub struct VecIterator<'self, T> {
    priv ptr: *T,
    priv end: *T,
    priv lifetime: &'self T // FIXME: #5922
}

impl<'self, T> Iterator<&'self T> for VecIterator<'self, T> {
    #[inline]
    fn next(&mut self) -> Option<&'self T> {
        unsafe {
            if self.ptr == self.end {
                None
            } else {
                let old = self.ptr;
                self.ptr = self.ptr.offset(1);
                Some(cast::transmute(old))
            }
        }
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
            assert!(b.len() == 3u);
            assert!(b[0] == 1);
            assert!(b[1] == 2);
            assert!(b[2] == 3);

            // Test on-heap copy-from-buf.
            let c = ~[1, 2, 3, 4, 5];
            ptr = raw::to_ptr(c);
            let d = from_buf(ptr, 5u);
            assert!(d.len() == 5u);
            assert!(d[0] == 1);
            assert!(d[1] == 2);
            assert!(d[2] == 3);
            assert!(d[3] == 4);
            assert!(d[4] == 5);
        }
    }

    #[test]
    fn test_from_fn() {
        // Test on-stack from_fn.
        let mut v = from_fn(3u, square);
        assert!(v.len() == 3u);
        assert!(v[0] == 0u);
        assert!(v[1] == 1u);
        assert!(v[2] == 4u);

        // Test on-heap from_fn.
        v = from_fn(5u, square);
        assert!(v.len() == 5u);
        assert!(v[0] == 0u);
        assert!(v[1] == 1u);
        assert!(v[2] == 4u);
        assert!(v[3] == 9u);
        assert!(v[4] == 16u);
    }

    #[test]
    fn test_from_elem() {
        // Test on-stack from_elem.
        let mut v = from_elem(2u, 10u);
        assert!(v.len() == 2u);
        assert!(v[0] == 10u);
        assert!(v[1] == 10u);

        // Test on-heap from_elem.
        v = from_elem(6u, 20u);
        assert!(v[0] == 20u);
        assert!(v[1] == 20u);
        assert!(v[2] == 20u);
        assert!(v[3] == 20u);
        assert!(v[4] == 20u);
        assert!(v[5] == 20u);
    }

    #[test]
    fn test_is_empty() {
        assert!(is_empty::<int>(~[]));
        assert!(!is_empty(~[0]));
    }

    #[test]
    fn test_len_divzero() {
        type Z = [i8, ..0];
        let v0 : &[Z] = &[];
        let v1 : &[Z] = &[[]];
        let v2 : &[Z] = &[[], []];
        assert!(sys::size_of::<Z>() == 0);
        assert!(v0.len() == 0);
        assert!(v1.len() == 1);
        assert!(v2.len() == 2);
    }

    #[test]
    fn test_head() {
        let mut a = ~[11];
        assert!(a.head() == &11);
        a = ~[11, 12];
        assert!(a.head() == &11);
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
        assert!(a.head_opt() == None);
        a = ~[11];
        assert!(a.head_opt().unwrap() == &11);
        a = ~[11, 12];
        assert!(a.head_opt().unwrap() == &11);
    }

    #[test]
    fn test_tail() {
        let mut a = ~[11];
        assert!(a.tail() == &[]);
        a = ~[11, 12];
        assert!(a.tail() == &[12]);
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
        assert!(a.tailn(0) == &[11, 12, 13]);
        a = ~[11, 12, 13];
        assert!(a.tailn(2) == &[13]);
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
        assert!(a.init() == &[]);
        a = ~[11, 12];
        assert!(a.init() == &[11]);
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
        assert!(a.initn(0) == &[11, 12, 13]);
        a = ~[11, 12, 13];
        assert!(a.initn(2) == &[11]);
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
        assert!(a.last() == &11);
        a = ~[11, 12];
        assert!(a.last() == &12);
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
        assert!(a.last_opt() == None);
        a = ~[11];
        assert!(a.last_opt().unwrap() == &11);
        a = ~[11, 12];
        assert!(a.last_opt().unwrap() == &12);
    }

    #[test]
    fn test_slice() {
        // Test fixed length vector.
        let vec_fixed = [1, 2, 3, 4];
        let v_a = slice(vec_fixed, 1u, vec_fixed.len()).to_vec();
        assert!(v_a.len() == 3u);
        assert!(v_a[0] == 2);
        assert!(v_a[1] == 3);
        assert!(v_a[2] == 4);

        // Test on stack.
        let vec_stack = &[1, 2, 3];
        let v_b = slice(vec_stack, 1u, 3u).to_vec();
        assert!(v_b.len() == 2u);
        assert!(v_b[0] == 2);
        assert!(v_b[1] == 3);

        // Test on managed heap.
        let vec_managed = @[1, 2, 3, 4, 5];
        let v_c = slice(vec_managed, 0u, 3u).to_vec();
        assert!(v_c.len() == 3u);
        assert!(v_c[0] == 1);
        assert!(v_c[1] == 2);
        assert!(v_c[2] == 3);

        // Test on exchange heap.
        let vec_unique = ~[1, 2, 3, 4, 5, 6];
        let v_d = slice(vec_unique, 1u, 6u).to_vec();
        assert!(v_d.len() == 5u);
        assert!(v_d[0] == 2);
        assert!(v_d[1] == 3);
        assert!(v_d[2] == 4);
        assert!(v_d[3] == 5);
        assert!(v_d[4] == 6);
    }

    #[test]
    fn test_pop() {
        // Test on-heap pop.
        let mut v = ~[1, 2, 3, 4, 5];
        let e = v.pop();
        assert!(v.len() == 4u);
        assert!(v[0] == 1);
        assert!(v[1] == 2);
        assert!(v[2] == 3);
        assert!(v[3] == 4);
        assert!(e == 5);
    }

    #[test]
    fn test_swap_remove() {
        let mut v = ~[1, 2, 3, 4, 5];
        let mut e = v.swap_remove(0);
        assert!(v.len() == 4);
        assert!(e == 1);
        assert!(v[0] == 5);
        e = v.swap_remove(3);
        assert!(v.len() == 3);
        assert!(e == 4);
        assert!(v[0] == 5);
        assert!(v[1] == 2);
        assert!(v[2] == 3);
    }

    #[test]
    fn test_swap_remove_noncopyable() {
        // Tests that we don't accidentally run destructors twice.
        let mut v = ~[::unstable::exclusive(()), ::unstable::exclusive(()),
                      ::unstable::exclusive(())];
        let mut _e = v.swap_remove(0);
        assert!(v.len() == 2);
        _e = v.swap_remove(1);
        assert!(v.len() == 1);
        _e = v.swap_remove(0);
        assert!(v.len() == 0);
    }

    #[test]
    fn test_push() {
        // Test on-stack push().
        let mut v = ~[];
        v.push(1);
        assert!(v.len() == 1u);
        assert!(v[0] == 1);

        // Test on-heap push().
        v.push(2);
        assert!(v.len() == 2u);
        assert!(v[0] == 1);
        assert!(v[1] == 2);
    }

    #[test]
    fn test_grow() {
        // Test on-stack grow().
        let mut v = ~[];
        v.grow(2u, &1);
        assert!(v.len() == 2u);
        assert!(v[0] == 1);
        assert!(v[1] == 1);

        // Test on-heap grow().
        v.grow(3u, &2);
        assert!(v.len() == 5u);
        assert!(v[0] == 1);
        assert!(v[1] == 1);
        assert!(v[2] == 2);
        assert!(v[3] == 2);
        assert!(v[4] == 2);
    }

    #[test]
    fn test_grow_fn() {
        let mut v = ~[];
        v.grow_fn(3u, square);
        assert!(v.len() == 3u);
        assert!(v[0] == 0u);
        assert!(v[1] == 1u);
        assert!(v[2] == 4u);
    }

    #[test]
    fn test_grow_set() {
        let mut v = ~[1, 2, 3];
        v.grow_set(4u, &4, 5);
        assert!(v.len() == 5u);
        assert!(v[0] == 1);
        assert!(v[1] == 2);
        assert!(v[2] == 3);
        assert!(v[3] == 4);
        assert!(v[4] == 5);
    }

    #[test]
    fn test_truncate() {
        let mut v = ~[@6,@5,@4];
        v.truncate(1);
        assert!(v.len() == 1);
        assert!(*(v[0]) == 6);
        // If the unsafe block didn't drop things properly, we blow up here.
    }

    #[test]
    fn test_clear() {
        let mut v = ~[@6,@5,@4];
        v.clear();
        assert!(v.len() == 0);
        // If the unsafe block didn't drop things properly, we blow up here.
    }

    #[test]
    fn test_dedup() {
        fn case(a: ~[uint], b: ~[uint]) {
            let mut v = a;
            v.dedup();
            assert!(v == b);
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
        assert!(w.len() == 3u);
        assert!(w[0] == 1u);
        assert!(w[1] == 4u);
        assert!(w[2] == 9u);

        // Test on-heap map.
        v = ~[1u, 2u, 3u, 4u, 5u];
        w = map(v, square_ref);
        assert!(w.len() == 5u);
        assert!(w[0] == 1u);
        assert!(w[1] == 4u);
        assert!(w[2] == 9u);
        assert!(w[3] == 16u);
        assert!(w[4] == 25u);
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
        assert!(w.len() == 2u);
        assert!(w[0] == 1u);
        assert!(w[1] == 9u);

        // Test on-heap filter-map.
        v = ~[1u, 2u, 3u, 4u, 5u];
        w = filter_mapped(v, square_if_odd_r);
        assert!(w.len() == 3u);
        assert!(w[0] == 1u);
        assert!(w[1] == 9u);
        assert!(w[2] == 25u);

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
        assert!(filter_mapped(all_odd1, halve) == ~[]);
        assert!(filter_mapped(all_odd2, halve) == ~[]);
        assert!(filter_mapped(mix, halve) == mix_dest);
    }

    #[test]
    fn test_filter_map() {
        // Test on-stack filter-map.
        let mut v = ~[1u, 2u, 3u];
        let mut w = filter_map(v, square_if_odd_v);
        assert!(w.len() == 2u);
        assert!(w[0] == 1u);
        assert!(w[1] == 9u);

        // Test on-heap filter-map.
        v = ~[1u, 2u, 3u, 4u, 5u];
        w = filter_map(v, square_if_odd_v);
        assert!(w.len() == 3u);
        assert!(w[0] == 1u);
        assert!(w[1] == 9u);
        assert!(w[2] == 25u);

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
        assert!(filter_map(all_odd1, halve) == ~[]);
        assert!(filter_map(all_odd2, halve) == ~[]);
        assert!(filter_map(mix, halve) == mix_dest);
    }

    #[test]
    fn test_filter() {
        assert!(filter(~[1u, 2u, 3u], is_odd) == ~[1u, 3u]);
        assert!(filter(~[1u, 2u, 4u, 8u, 16u], is_three) == ~[]);
    }

    #[test]
    fn test_retain() {
        let mut v = ~[1, 2, 3, 4, 5];
        v.retain(is_odd);
        assert!(v == ~[1, 3, 5]);
    }

    #[test]
    fn test_foldl() {
        // Test on-stack fold.
        let mut v = ~[1u, 2u, 3u];
        let mut sum = foldl(0u, v, add);
        assert!(sum == 6u);

        // Test on-heap fold.
        v = ~[1u, 2u, 3u, 4u, 5u];
        sum = foldl(0u, v, add);
        assert!(sum == 15u);
    }

    #[test]
    fn test_foldl2() {
        fn sub(a: int, b: &int) -> int {
            a - *b
        }
        let mut v = ~[1, 2, 3, 4];
        let sum = foldl(0, v, sub);
        assert!(sum == -10);
    }

    #[test]
    fn test_foldr() {
        fn sub(a: &int, b: int) -> int {
            *a - b
        }
        let mut v = ~[1, 2, 3, 4];
        let sum = foldr(v, 0, sub);
        assert!(sum == -2);
    }

    #[test]
    fn test_each_empty() {
        for each::<int>(~[]) |_v| {
            fail!(); // should never be executed
        }
    }

    #[test]
    fn test_each_nonempty() {
        let mut i = 0;
        for each(~[1, 2, 3]) |v| {
            i += *v;
        }
        assert!(i == 6);
    }

    #[test]
    fn test_eachi() {
        let mut i = 0;
        for eachi(~[1, 2, 3]) |j, v| {
            if i == 0 { assert!(*v == 1); }
            assert!(j + 1u == *v as uint);
            i += *v;
        }
        assert!(i == 6);
    }

    #[test]
    fn test_each_reverse_empty() {
        let v: ~[int] = ~[];
        for v.each_reverse |_v| {
            fail!(); // should never execute
        }
    }

    #[test]
    fn test_each_reverse_nonempty() {
        let mut i = 0;
        for each_reverse(~[1, 2, 3]) |v| {
            if i == 0 { assert!(*v == 3); }
            i += *v
        }
        assert!(i == 6);
    }

    #[test]
    fn test_eachi_reverse() {
        let mut i = 0;
        for eachi_reverse(~[0, 1, 2]) |j, v| {
            if i == 0 { assert!(*v == 2); }
            assert!(j == *v as uint);
            i += *v;
        }
        assert!(i == 3);
    }

    #[test]
    fn test_eachi_reverse_empty() {
        let v: ~[int] = ~[];
        for v.eachi_reverse |_i, _v| {
            fail!(); // should never execute
        }
    }

    #[test]
    fn test_each_permutation() {
        let mut results: ~[~[int]];

        results = ~[];
        for each_permutation(~[]) |v| { results.push(from_slice(v)); }
        assert!(results == ~[~[]]);

        results = ~[];
        for each_permutation(~[7]) |v| { results.push(from_slice(v)); }
        assert!(results == ~[~[7]]);

        results = ~[];
        for each_permutation(~[1,1]) |v| { results.push(from_slice(v)); }
        assert!(results == ~[~[1,1],~[1,1]]);

        results = ~[];
        for each_permutation(~[5,2,0]) |v| { results.push(from_slice(v)); }
        assert!(results ==
            ~[~[5,2,0],~[5,0,2],~[2,5,0],~[2,0,5],~[0,5,2],~[0,2,5]]);
    }

    #[test]
    fn test_any_and_all() {
        assert!(any(~[1u, 2u, 3u], is_three));
        assert!(!any(~[0u, 1u, 2u], is_three));
        assert!(any(~[1u, 2u, 3u, 4u, 5u], is_three));
        assert!(!any(~[1u, 2u, 4u, 5u, 6u], is_three));

        assert!(all(~[3u, 3u, 3u], is_three));
        assert!(!all(~[3u, 3u, 2u], is_three));
        assert!(all(~[3u, 3u, 3u, 3u, 3u], is_three));
        assert!(!all(~[3u, 3u, 0u, 1u, 2u], is_three));
    }

    #[test]
    fn test_any2_and_all2() {

        assert!(any2(~[2u, 4u, 6u], ~[2u, 4u, 6u], is_equal));
        assert!(any2(~[1u, 2u, 3u], ~[4u, 5u, 3u], is_equal));
        assert!(!any2(~[1u, 2u, 3u], ~[4u, 5u, 6u], is_equal));
        assert!(any2(~[2u, 4u, 6u], ~[2u, 4u], is_equal));

        assert!(all2(~[2u, 4u, 6u], ~[2u, 4u, 6u], is_equal));
        assert!(!all2(~[1u, 2u, 3u], ~[4u, 5u, 3u], is_equal));
        assert!(!all2(~[1u, 2u, 3u], ~[4u, 5u, 6u], is_equal));
        assert!(!all2(~[2u, 4u, 6u], ~[2u, 4u], is_equal));
    }

    #[test]
    fn test_zip_unzip() {
        let v1 = ~[1, 2, 3];
        let v2 = ~[4, 5, 6];

        let z1 = zip(v1, v2);

        assert!((1, 4) == z1[0]);
        assert!((2, 5) == z1[1]);
        assert!((3, 6) == z1[2]);

        let (left, right) = unzip(z1);

        assert!((1, 4) == (left[0], right[0]));
        assert!((2, 5) == (left[1], right[1]));
        assert!((3, 6) == (left[2], right[2]));
    }

    #[test]
    fn test_position_elem() {
        assert!(position_elem(~[], &1).is_none());

        let v1 = ~[1, 2, 3, 3, 2, 5];
        assert!(position_elem(v1, &1) == Some(0u));
        assert!(position_elem(v1, &2) == Some(1u));
        assert!(position_elem(v1, &5) == Some(5u));
        assert!(position_elem(v1, &4).is_none());
    }

    #[test]
    fn test_position() {
        fn less_than_three(i: &int) -> bool { *i < 3 }
        fn is_eighteen(i: &int) -> bool { *i == 18 }

        assert!(position(~[], less_than_three).is_none());

        let v1 = ~[5, 4, 3, 2, 1];
        assert!(position(v1, less_than_three) == Some(3u));
        assert!(position(v1, is_eighteen).is_none());
    }

    #[test]
    fn test_position_between() {
        assert!(position_between(~[], 0u, 0u, f).is_none());

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert!(position_between(v, 0u, 0u, f).is_none());
        assert!(position_between(v, 0u, 1u, f).is_none());
        assert!(position_between(v, 0u, 2u, f) == Some(1u));
        assert!(position_between(v, 0u, 3u, f) == Some(1u));
        assert!(position_between(v, 0u, 4u, f) == Some(1u));

        assert!(position_between(v, 1u, 1u, f).is_none());
        assert!(position_between(v, 1u, 2u, f) == Some(1u));
        assert!(position_between(v, 1u, 3u, f) == Some(1u));
        assert!(position_between(v, 1u, 4u, f) == Some(1u));

        assert!(position_between(v, 2u, 2u, f).is_none());
        assert!(position_between(v, 2u, 3u, f).is_none());
        assert!(position_between(v, 2u, 4u, f) == Some(3u));

        assert!(position_between(v, 3u, 3u, f).is_none());
        assert!(position_between(v, 3u, 4u, f) == Some(3u));

        assert!(position_between(v, 4u, 4u, f).is_none());
    }

    #[test]
    fn test_find() {
        assert!(find(~[], f).is_none());

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        fn g(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'd' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert!(find(v, f) == Some((1, 'b')));
        assert!(find(v, g).is_none());
    }

    #[test]
    fn test_find_between() {
        assert!(find_between(~[], 0u, 0u, f).is_none());

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert!(find_between(v, 0u, 0u, f).is_none());
        assert!(find_between(v, 0u, 1u, f).is_none());
        assert!(find_between(v, 0u, 2u, f) == Some((1, 'b')));
        assert!(find_between(v, 0u, 3u, f) == Some((1, 'b')));
        assert!(find_between(v, 0u, 4u, f) == Some((1, 'b')));

        assert!(find_between(v, 1u, 1u, f).is_none());
        assert!(find_between(v, 1u, 2u, f) == Some((1, 'b')));
        assert!(find_between(v, 1u, 3u, f) == Some((1, 'b')));
        assert!(find_between(v, 1u, 4u, f) == Some((1, 'b')));

        assert!(find_between(v, 2u, 2u, f).is_none());
        assert!(find_between(v, 2u, 3u, f).is_none());
        assert!(find_between(v, 2u, 4u, f) == Some((3, 'b')));

        assert!(find_between(v, 3u, 3u, f).is_none());
        assert!(find_between(v, 3u, 4u, f) == Some((3, 'b')));

        assert!(find_between(v, 4u, 4u, f).is_none());
    }

    #[test]
    fn test_rposition() {
        assert!(find(~[], f).is_none());

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        fn g(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'd' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert!(position(v, f) == Some(1u));
        assert!(position(v, g).is_none());
    }

    #[test]
    fn test_rposition_between() {
        assert!(rposition_between(~[], 0u, 0u, f).is_none());

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert!(rposition_between(v, 0u, 0u, f).is_none());
        assert!(rposition_between(v, 0u, 1u, f).is_none());
        assert!(rposition_between(v, 0u, 2u, f) == Some(1u));
        assert!(rposition_between(v, 0u, 3u, f) == Some(1u));
        assert!(rposition_between(v, 0u, 4u, f) == Some(3u));

        assert!(rposition_between(v, 1u, 1u, f).is_none());
        assert!(rposition_between(v, 1u, 2u, f) == Some(1u));
        assert!(rposition_between(v, 1u, 3u, f) == Some(1u));
        assert!(rposition_between(v, 1u, 4u, f) == Some(3u));

        assert!(rposition_between(v, 2u, 2u, f).is_none());
        assert!(rposition_between(v, 2u, 3u, f).is_none());
        assert!(rposition_between(v, 2u, 4u, f) == Some(3u));

        assert!(rposition_between(v, 3u, 3u, f).is_none());
        assert!(rposition_between(v, 3u, 4u, f) == Some(3u));

        assert!(rposition_between(v, 4u, 4u, f).is_none());
    }

    #[test]
    fn test_rfind() {
        assert!(rfind(~[], f).is_none());

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        fn g(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'd' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert!(rfind(v, f) == Some((3, 'b')));
        assert!(rfind(v, g).is_none());
    }

    #[test]
    fn test_rfind_between() {
        assert!(rfind_between(~[], 0u, 0u, f).is_none());

        fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert!(rfind_between(v, 0u, 0u, f).is_none());
        assert!(rfind_between(v, 0u, 1u, f).is_none());
        assert!(rfind_between(v, 0u, 2u, f) == Some((1, 'b')));
        assert!(rfind_between(v, 0u, 3u, f) == Some((1, 'b')));
        assert!(rfind_between(v, 0u, 4u, f) == Some((3, 'b')));

        assert!(rfind_between(v, 1u, 1u, f).is_none());
        assert!(rfind_between(v, 1u, 2u, f) == Some((1, 'b')));
        assert!(rfind_between(v, 1u, 3u, f) == Some((1, 'b')));
        assert!(rfind_between(v, 1u, 4u, f) == Some((3, 'b')));

        assert!(rfind_between(v, 2u, 2u, f).is_none());
        assert!(rfind_between(v, 2u, 3u, f).is_none());
        assert!(rfind_between(v, 2u, 4u, f) == Some((3, 'b')));

        assert!(rfind_between(v, 3u, 3u, f).is_none());
        assert!(rfind_between(v, 3u, 4u, f) == Some((3, 'b')));

        assert!(rfind_between(v, 4u, 4u, f).is_none());
    }

    #[test]
    fn test_bsearch_elem() {
        assert!(bsearch_elem([1,2,3,4,5], &5) == Some(4));
        assert!(bsearch_elem([1,2,3,4,5], &4) == Some(3));
        assert!(bsearch_elem([1,2,3,4,5], &3) == Some(2));
        assert!(bsearch_elem([1,2,3,4,5], &2) == Some(1));
        assert!(bsearch_elem([1,2,3,4,5], &1) == Some(0));

        assert!(bsearch_elem([2,4,6,8,10], &1) == None);
        assert!(bsearch_elem([2,4,6,8,10], &5) == None);
        assert!(bsearch_elem([2,4,6,8,10], &4) == Some(1));
        assert!(bsearch_elem([2,4,6,8,10], &10) == Some(4));

        assert!(bsearch_elem([2,4,6,8], &1) == None);
        assert!(bsearch_elem([2,4,6,8], &5) == None);
        assert!(bsearch_elem([2,4,6,8], &4) == Some(1));
        assert!(bsearch_elem([2,4,6,8], &8) == Some(3));

        assert!(bsearch_elem([2,4,6], &1) == None);
        assert!(bsearch_elem([2,4,6], &5) == None);
        assert!(bsearch_elem([2,4,6], &4) == Some(1));
        assert!(bsearch_elem([2,4,6], &6) == Some(2));

        assert!(bsearch_elem([2,4], &1) == None);
        assert!(bsearch_elem([2,4], &5) == None);
        assert!(bsearch_elem([2,4], &2) == Some(0));
        assert!(bsearch_elem([2,4], &4) == Some(1));

        assert!(bsearch_elem([2], &1) == None);
        assert!(bsearch_elem([2], &5) == None);
        assert!(bsearch_elem([2], &2) == Some(0));

        assert!(bsearch_elem([], &1) == None);
        assert!(bsearch_elem([], &5) == None);

        assert!(bsearch_elem([1,1,1,1,1], &1) != None);
        assert!(bsearch_elem([1,1,1,1,2], &1) != None);
        assert!(bsearch_elem([1,1,1,2,2], &1) != None);
        assert!(bsearch_elem([1,1,2,2,2], &1) != None);
        assert!(bsearch_elem([1,2,2,2,2], &1) == Some(0));

        assert!(bsearch_elem([1,2,3,4,5], &6) == None);
        assert!(bsearch_elem([1,2,3,4,5], &0) == None);
    }

    #[test]
    fn reverse_and_reversed() {
        let mut v: ~[int] = ~[10, 20];
        assert!(v[0] == 10);
        assert!(v[1] == 20);
        reverse(v);
        assert!(v[0] == 20);
        assert!(v[1] == 10);
        let v2 = reversed::<int>(~[10, 20]);
        assert!(v2[0] == 20);
        assert!(v2[1] == 10);
        v[0] = 30;
        assert!(v2[0] == 20);
        // Make sure they work with 0-length vectors too.

        let v4 = reversed::<int>(~[]);
        assert!(v4 == ~[]);
        let mut v3: ~[int] = ~[];
        reverse::<int>(v3);
    }

    #[test]
    fn reversed_mut() {
        let v2 = reversed::<int>(~[10, 20]);
        assert!(v2[0] == 20);
        assert!(v2[1] == 10);
    }

    #[test]
    fn test_split() {
        fn f(x: &int) -> bool { *x == 3 }

        assert!(split(~[], f) == ~[]);
        assert!(split(~[1, 2], f) == ~[~[1, 2]]);
        assert!(split(~[3, 1, 2], f) == ~[~[], ~[1, 2]]);
        assert!(split(~[1, 2, 3], f) == ~[~[1, 2], ~[]]);
        assert!(split(~[1, 2, 3, 4, 3, 5], f) == ~[~[1, 2], ~[4], ~[5]]);
    }

    #[test]
    fn test_splitn() {
        fn f(x: &int) -> bool { *x == 3 }

        assert!(splitn(~[], 1u, f) == ~[]);
        assert!(splitn(~[1, 2], 1u, f) == ~[~[1, 2]]);
        assert!(splitn(~[3, 1, 2], 1u, f) == ~[~[], ~[1, 2]]);
        assert!(splitn(~[1, 2, 3], 1u, f) == ~[~[1, 2], ~[]]);
        assert!(splitn(~[1, 2, 3, 4, 3, 5], 1u, f) ==
                      ~[~[1, 2], ~[4, 3, 5]]);
    }

    #[test]
    fn test_rsplit() {
        fn f(x: &int) -> bool { *x == 3 }

        assert!(rsplit(~[], f) == ~[]);
        assert!(rsplit(~[1, 2], f) == ~[~[1, 2]]);
        assert!(rsplit(~[1, 2, 3], f) == ~[~[1, 2], ~[]]);
        assert!(rsplit(~[1, 2, 3, 4, 3, 5], f) ==
            ~[~[1, 2], ~[4], ~[5]]);
    }

    #[test]
    fn test_rsplitn() {
        fn f(x: &int) -> bool { *x == 3 }

        assert!(rsplitn(~[], 1u, f) == ~[]);
        assert!(rsplitn(~[1, 2], 1u, f) == ~[~[1, 2]]);
        assert!(rsplitn(~[1, 2, 3], 1u, f) == ~[~[1, 2], ~[]]);
        assert!(rsplitn(~[1, 2, 3, 4, 3, 5], 1u, f) ==
                       ~[~[1, 2, 3, 4], ~[5]]);
    }

    #[test]
    fn test_partition() {
        // FIXME (#4355 maybe): using v.partition here crashes
        assert!(partition(~[], |x: &int| *x < 3) == (~[], ~[]));
        assert!(partition(~[1, 2, 3], |x: &int| *x < 4) ==
            (~[1, 2, 3], ~[]));
        assert!(partition(~[1, 2, 3], |x: &int| *x < 2) ==
            (~[1], ~[2, 3]));
        assert!(partition(~[1, 2, 3], |x: &int| *x < 0) ==
            (~[], ~[1, 2, 3]));
    }

    #[test]
    fn test_partitioned() {
        assert!((~[]).partitioned(|x: &int| *x < 3) == (~[], ~[]));
        assert!((~[1, 2, 3]).partitioned(|x: &int| *x < 4) ==
                     (~[1, 2, 3], ~[]));
        assert!((~[1, 2, 3]).partitioned(|x: &int| *x < 2) ==
                     (~[1], ~[2, 3]));
        assert!((~[1, 2, 3]).partitioned(|x: &int| *x < 0) ==
                     (~[], ~[1, 2, 3]));
    }

    #[test]
    fn test_concat() {
        assert!(concat(~[~[1], ~[2,3]]) == ~[1, 2, 3]);
    }

    #[test]
    fn test_connect() {
        assert!(connect(~[], &0) == ~[]);
        assert!(connect(~[~[1], ~[2, 3]], &0) == ~[1, 0, 2, 3]);
        assert!(connect(~[~[1], ~[2], ~[3]], &0) == ~[1, 0, 2, 0, 3]);
    }

    #[test]
    fn test_windowed () {
        fn t(n: uint, expected: &[&[int]]) {
            let mut i = 0;
            for windowed(n, ~[1,2,3,4,5,6]) |v| {
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
        for windowed (0u, ~[1u,2u,3u,4u,5u,6u]) |_v| {}
    }

    #[test]
    fn test_unshift() {
        let mut x = ~[1, 2, 3];
        x.unshift(0);
        assert!(x == ~[0, 1, 2, 3]);
    }

    #[test]
    fn test_insert() {
        let mut a = ~[1, 2, 4];
        a.insert(2, 3);
        assert!(a == ~[1, 2, 3, 4]);

        let mut a = ~[1, 2, 3];
        a.insert(0, 0);
        assert!(a == ~[0, 1, 2, 3]);

        let mut a = ~[1, 2, 3];
        a.insert(3, 4);
        assert!(a == ~[1, 2, 3, 4]);

        let mut a = ~[];
        a.insert(0, 1);
        assert!(a == ~[1]);
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
        assert!(a == ~[1, 2, 4]);

        let mut a = ~[1, 2, 3];
        a.remove(0);
        assert!(a == ~[2, 3]);

        let mut a = ~[1];
        a.remove(0);
        assert!(a == ~[]);
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
        assert!(capacity(&v) == 10u);
        let mut v = ~[0u32];
        reserve(&mut v, 10u);
        assert!(capacity(&v) == 10u);
    }

    #[test]
    fn test_slice_2() {
        let v = ~[1, 2, 3, 4, 5];
        let v = v.slice(1u, 3u);
        assert!(v.len() == 2u);
        assert!(v[0] == 2);
        assert!(v[1] == 3);
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
        let mut v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
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
    #[allow(non_implicitly_copyable_typarams)]
    fn test_foldl_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        let mut i = 0;
        do foldl((~0, @0), v) |_a, _b| {
            if i == 2 {
                fail!()
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
                fail!()
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
                fail!()
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
                fail!()
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
                fail!()
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
                fail!()
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
                fail!()
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
                fail!()
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
                fail!()
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
                fail!()
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
                fail!()
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
                fail!()
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
    #[ignore(windows)]
    #[should_fail]
    fn test_as_const_buf_fail() {
        let v = [(~0, @0), (~0, @0), (~0, @0), (~0, @0)];
        do as_const_buf(v) |_buf, _i| {
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
        let ys = [1, 2, 5, 10, 11, 19];
        let mut it = xs.iter();
        let mut i = 0;
        for it.advance |&x| {
            assert_eq!(x, ys[i]);
            i += 1;
        }
    }
}
