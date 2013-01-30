// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Dynamic vector

A growable vector that makes use of unique pointers so that the
result can be sent between tasks and so forth.

Note that recursive use is not permitted.

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cast;
use cast::reinterpret_cast;
use prelude::*;
use ptr::null;
use vec;

/**
 * A growable, modifiable vector type that accumulates elements into a
 * unique vector.
 *
 * # Limitations on recursive use
 *
 * This class works by swapping the unique vector out of the data
 * structure whenever it is to be used.  Therefore, recursive use is not
 * permitted.  That is, while iterating through a vector, you cannot
 * access the vector in any other way or else the program will fail.  If
 * you wish, you can use the `swap()` method to gain access to the raw
 * vector and transform it or use it any way you like.  Eventually, we
 * may permit read-only access during iteration or other use.
 *
 * # WARNING
 *
 * For maximum performance, this type is implemented using some rather
 * unsafe code.  In particular, this innocent looking `~[mut A]` pointer
 * *may be null!*  Therefore, it is important you not reach into the
 * data structure manually but instead use the provided extensions.
 *
 * The reason that I did not use an unsafe pointer in the structure
 * itself is that I wanted to ensure that the vector would be freed when
 * the dvec is dropped.  The reason that I did not use an `Option<T>`
 * instead of a nullable pointer is that I found experimentally that it
 * becomes approximately 50% slower. This can probably be improved
 * through optimization.  You can run your own experiments using
 * `src/test/bench/vec-append.rs`. My own tests found that using null
 * pointers achieved about 103 million pushes/second.  Using an option
 * type could only produce 47 million pushes/second.
 */
pub struct DVec<A> {
    mut data: ~[A]
}

/// Creates a new, empty dvec
pub pure fn DVec<A>() -> DVec<A> {
    DVec {mut data: ~[]}
}

/// Creates a new dvec with a single element
pub pure fn from_elem<A>(e: A) -> DVec<A> {
    DVec {mut data: ~[move e]}
}

/// Creates a new dvec with the contents of a vector
pub pure fn from_vec<A>(v: ~[A]) -> DVec<A> {
    DVec {mut data: move v}
}

/// Consumes the vector and returns its contents
pub pure fn unwrap<A>(d: DVec<A>) -> ~[A] {
    let DVec {data: v} = move d;
    move v
}

priv impl<A> DVec<A> {
    #[inline(always)]
    pure fn check_not_borrowed() {
        unsafe {
            let data: *() = cast::reinterpret_cast(&self.data);
            if data.is_null() {
                fail ~"Recursive use of dvec";
            }
        }
    }

    #[inline(always)]
    fn check_out<B>(f: &fn(v: ~[A]) -> B) -> B {
        unsafe {
            let mut data = cast::reinterpret_cast(&null::<()>());
            data <-> self.data;
            let data_ptr: *() = cast::reinterpret_cast(&data);
            if data_ptr.is_null() { fail ~"Recursive use of dvec"; }
            return f(move data);
        }
    }

    #[inline(always)]
    fn give_back(data: ~[A]) {
        unsafe {
            self.data = move data;
        }
    }

    #[inline(always)]
    fn unwrap(self) -> ~[A] { unwrap(self) }
}

// In theory, most everything should work with any A, but in practice
// almost nothing works without the copy bound due to limitations
// around closures.
impl<A> DVec<A> {
    /// Reserves space for N elements
    fn reserve(count: uint) {
        vec::reserve(&mut self.data, count)
    }

    /**
     * Swaps out the current vector and hands it off to a user-provided
     * function `f`.  The function should transform it however is desired
     * and return a new vector to replace it with.
     */
    #[inline(always)]
    fn swap(f: &fn(v: ~[A]) -> ~[A]) {
        self.check_out(|v| self.give_back(f(move v)))
    }

    /**
     * Swaps out the current vector and hands it off to a user-provided
     * function `f`.  The function should transform it however is desired
     * and return a new vector to replace it with.
     */
    #[inline(always)]
    fn swap_mut(f: &fn(v: ~[mut A]) -> ~[mut A]) {
        do self.swap |v| {
            vec::cast_from_mut(f(vec::cast_to_mut(move v)))
        }
    }

    /// Returns the number of elements currently in the dvec
    #[inline(always)]
    pure fn len() -> uint {
        self.check_not_borrowed();
        return self.data.len();
    }

    /// Overwrite the current contents
    #[inline(always)]
    fn set(w: ~[A]) {
        self.check_not_borrowed();
        self.data = move w;
    }

    /// Remove and return the last element
    fn pop() -> A {
        do self.check_out |v| {
            let mut v = move v;
            let result = v.pop();
            self.give_back(move v);
            move result
        }
    }

    /// Insert a single item at the front of the list
    fn unshift(t: A) {
        unsafe {
            let mut data = cast::reinterpret_cast(&null::<()>());
            data <-> self.data;
            let data_ptr: *() = cast::reinterpret_cast(&data);
            if data_ptr.is_null() { fail ~"Recursive use of dvec"; }
            log(error, ~"a");
            self.data = move ~[move t];
            self.data.push_all_move(move data);
            log(error, ~"b");
        }
    }

    /// Append a single item to the end of the list
    #[inline(always)]
    fn push(t: A) {
        self.check_not_borrowed();
        self.data.push(move t);
    }

    /// Remove and return the first element
    fn shift() -> A {
        do self.check_out |v| {
            let mut v = move v;
            let result = v.shift();
            self.give_back(move v);
            move result
        }
    }

    /// Reverse the elements in the list, in place
    fn reverse() {
        do self.check_out |v| {
            let mut v = move v;
            vec::reverse(v);
            self.give_back(move v);
        }
    }

    /// Gives access to the vector as a slice with immutable contents
    fn borrow<R>(op: fn(x: &[A]) -> R) -> R {
        do self.check_out |v| {
            let result = op(v);
            self.give_back(move v);
            move result
        }
    }

    /// Gives access to the vector as a slice with mutable contents
    fn borrow_mut<R>(op: fn(x: &mut [A]) -> R) -> R {
        do self.check_out |v| {
            let mut v = move v;
            let result = op(v);
            self.give_back(move v);
            move result
        }
    }
}

impl<A: Copy> DVec<A> {
    /**
     * Append all elements of a vector to the end of the list
     *
     * Equivalent to `append_iter()` but potentially more efficient.
     */
    fn push_all(ts: &[const A]) {
        self.push_slice(ts, 0u, vec::len(ts));
    }

    /// Appends elements from `from_idx` to `to_idx` (exclusive)
    fn push_slice(ts: &[const A], from_idx: uint, to_idx: uint) {
        do self.swap |v| {
            let mut v = move v;
            let new_len = vec::len(v) + to_idx - from_idx;
            vec::reserve(&mut v, new_len);
            let mut i = from_idx;
            while i < to_idx {
                v.push(ts[i]);
                i += 1u;
            }
            move v
        }
    }

    /**
     * Append all elements of an iterable.
     *
     * Failure will occur if the iterable's `each()` method
     * attempts to access this vector.
     */
    /*
    fn append_iter<A, I:iter::base_iter<A>>(ts: I) {
        do self.swap |v| {
           let mut v = match ts.size_hint() {
             none { v }
             Some(h) {
               let len = v.len() + h;
               let mut v = move v;
               vec::reserve(v, len);
               v
            }
           };

        for ts.each |t| { v.push(*t) };
           v
        }
    }
    */

    /**
     * Gets a copy of the current contents.
     *
     * See `unwrap()` if you do not wish to copy the contents.
     */
    pure fn get() -> ~[A] {
        unsafe {
            do self.check_out |v| {
                let w = copy v;
                self.give_back(move v);
                move w
            }
        }
    }

    /// Copy out an individual element
    #[inline(always)]
    pure fn get_elt(idx: uint) -> A {
        self.check_not_borrowed();
        return self.data[idx];
    }

    /// Overwrites the contents of the element at `idx` with `a`
    fn set_elt(idx: uint, a: A) {
        self.check_not_borrowed();
        self.data[idx] = a;
    }

    /**
     * Overwrites the contents of the element at `idx` with `a`,
     * growing the vector if necessary.  New elements will be initialized
     * with `initval`
     */
    fn grow_set_elt(idx: uint, initval: &A, val: A) {
        do self.swap |v| {
            let mut v = move v;
            v.grow_set(idx, initval, val);
            move v
        }
    }

    /// Returns the last element, failing if the vector is empty
    #[inline(always)]
    pure fn last() -> A {
        self.check_not_borrowed();

        let length = self.len();
        if length == 0 {
            fail ~"attempt to retrieve the last element of an empty vector";
        }

        return self.data[length - 1];
    }

    /// Iterates over the elements in reverse order
    #[inline(always)]
    fn rev_each(f: fn(v: &A) -> bool) {
        do self.swap |v| {
            // FIXME(#2263)---we should be able to write
            // `vec::rev_each(v, f);` but we cannot write now
            for vec::rev_each(v) |e| {
                if !f(e) { break; }
            }
            move v
        }
    }

    /// Iterates over the elements and indices in reverse order
    #[inline(always)]
    fn rev_eachi(f: fn(uint, v: &A) -> bool) {
        do self.swap |v| {
            // FIXME(#2263)---we should be able to write
            // `vec::rev_eachi(v, f);` but we cannot write now
            for vec::rev_eachi(v) |i, e| {
                if !f(i, e) { break; }
            }
            move v
        }
    }
}

impl<A:Copy> DVec<A>: Index<uint,A> {
    #[inline(always)]
    pure fn index(&self, idx: uint) -> A {
        self.get_elt(idx)
    }
}

