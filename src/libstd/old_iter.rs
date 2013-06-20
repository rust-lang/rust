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

**Deprecated** iteration traits and common implementations.

*/

#[allow(missing_doc)];

use cmp::{Eq};
use kinds::Copy;
use option::{None, Option, Some};
use vec;

/// A function used to initialize the elements of a sequence
pub type InitOp<'self,T> = &'self fn(uint) -> T;

pub trait BaseIter<A> {
    fn each(&self, blk: &fn(v: &A) -> bool) -> bool;
    fn size_hint(&self) -> Option<uint>;
}

pub trait ReverseIter<A>: BaseIter<A> {
    fn each_reverse(&self, blk: &fn(&A) -> bool) -> bool;
}

pub trait ExtendedIter<A> {
    fn eachi(&self, blk: &fn(uint, v: &A) -> bool) -> bool;
    fn all(&self, blk: &fn(&A) -> bool) -> bool;
    fn any(&self, blk: &fn(&A) -> bool) -> bool;
    fn foldl<B>(&self, b0: B, blk: &fn(&B, &A) -> B) -> B;
    fn position(&self, f: &fn(&A) -> bool) -> Option<uint>;
    fn map_to_vec<B>(&self, op: &fn(&A) -> B) -> ~[B];
    fn flat_map_to_vec<B,IB: BaseIter<B>>(&self, op: &fn(&A) -> IB) -> ~[B];
}

pub trait EqIter<A:Eq> {
    fn contains(&self, x: &A) -> bool;
    fn count(&self, x: &A) -> uint;
}

pub trait CopyableIter<A:Copy> {
    fn filter_to_vec(&self, pred: &fn(&A) -> bool) -> ~[A];
    fn to_vec(&self) -> ~[A];
    fn find(&self, p: &fn(&A) -> bool) -> Option<A>;
}

// A trait for sequences that can be built by imperatively pushing elements
// onto them.
pub trait Buildable<A> {
    /**
     * Builds a buildable sequence by calling a provided function with
     * an argument function that pushes an element onto the back of
     * the sequence.
     * This version takes an initial size for the sequence.
     *
     * # Arguments
     *
     * * size - A hint for an initial size of the sequence
     * * builder - A function that will construct the sequence. It receives
     *             as an argument a function that will push an element
     *             onto the sequence being constructed.
     */
     fn build_sized(size: uint, builder: &fn(push: &fn(A))) -> Self;
}

#[inline]
pub fn _eachi<A,IA:BaseIter<A>>(this: &IA, blk: &fn(uint, &A) -> bool) -> bool {
    let mut i = 0;
    for this.each |a| {
        if !blk(i, a) {
            return false;
        }
        i += 1;
    }
    return true;
}

pub fn eachi<A,IA:BaseIter<A>>(this: &IA, blk: &fn(uint, &A) -> bool) -> bool {
    _eachi(this, blk)
}

#[inline]
pub fn all<A,IA:BaseIter<A>>(this: &IA, blk: &fn(&A) -> bool) -> bool {
    for this.each |a| {
        if !blk(a) {
            return false;
        }
    }
    return true;
}

#[inline]
pub fn any<A,IA:BaseIter<A>>(this: &IA, blk: &fn(&A) -> bool) -> bool {
    for this.each |a| {
        if blk(a) {
            return true;
        }
    }
    return false;
}

#[inline]
pub fn filter_to_vec<A:Copy,IA:BaseIter<A>>(this: &IA,
                                            prd: &fn(&A) -> bool)
                                         -> ~[A] {
    do vec::build_sized_opt(this.size_hint()) |push| {
        for this.each |a| {
            if prd(a) { push(copy *a); }
        }
    }
}

#[inline]
pub fn map_to_vec<A,B,IA:BaseIter<A>>(this: &IA, op: &fn(&A) -> B) -> ~[B] {
    do vec::build_sized_opt(this.size_hint()) |push| {
        for this.each |a| {
            push(op(a));
        }
    }
}

#[inline]
pub fn flat_map_to_vec<A,B,IA:BaseIter<A>,IB:BaseIter<B>>(this: &IA,
                                                          op: &fn(&A) -> IB)
                                                       -> ~[B] {
    do vec::build |push| {
        for this.each |a| {
            for op(a).each |&b| {
                push(b);
            }
        }
    }
}

#[inline]
pub fn foldl<A,B,IA:BaseIter<A>>(this: &IA, b0: B, blk: &fn(&B, &A) -> B)
                              -> B {
    let mut b = b0;
    for this.each |a| {
        b = blk(&b, a);
    }
    b
}

#[inline]
pub fn to_vec<A:Copy,IA:BaseIter<A>>(this: &IA) -> ~[A] {
    map_to_vec(this, |&x| x)
}

#[inline]
pub fn contains<A:Eq,IA:BaseIter<A>>(this: &IA, x: &A) -> bool {
    for this.each |a| {
        if *a == *x { return true; }
    }
    return false;
}

#[inline]
pub fn count<A:Eq,IA:BaseIter<A>>(this: &IA, x: &A) -> uint {
    do foldl(this, 0) |count, value| {
        if *value == *x {
            *count + 1
        } else {
            *count
        }
    }
}

#[inline]
pub fn position<A,IA:BaseIter<A>>(this: &IA, f: &fn(&A) -> bool)
                               -> Option<uint> {
    let mut i = 0;
    for this.each |a| {
        if f(a) { return Some(i); }
        i += 1;
    }
    return None;
}

#[inline]
pub fn find<A:Copy,IA:BaseIter<A>>(this: &IA, f: &fn(&A) -> bool)
                                -> Option<A> {
    for this.each |i| {
        if f(i) { return Some(copy *i) }
    }
    return None;
}

// Some functions for just building

/**
 * Builds a sequence by calling a provided function with an argument
 * function that pushes an element to the back of a sequence.
 *
 * # Arguments
 *
 * * builder - A function that will construct the sequence. It receives
 *             as an argument a function that will push an element
 *             onto the sequence being constructed.
 */
#[inline]
pub fn build<A,B: Buildable<A>>(builder: &fn(push: &fn(A))) -> B {
    Buildable::build_sized(4, builder)
}

/**
 * Builds a sequence by calling a provided function with an argument
 * function that pushes an element to the back of the sequence.
 * This version takes an initial size for the sequence.
 *
 * # Arguments
 *
 * * size - An option, maybe containing initial size of the sequence
 *          to reserve.
 * * builder - A function that will construct the sequence. It receives
 *             as an argument a function that will push an element
 *             onto the sequence being constructed.
 */
#[inline]
pub fn build_sized_opt<A,B: Buildable<A>>(size: Option<uint>,
                                          builder: &fn(push: &fn(A))) -> B {
    Buildable::build_sized(size.get_or_default(4), builder)
}

// Functions that combine iteration and building

/// Applies a function to each element of an iterable and returns the results
/// in a sequence built via `BU`.  See also `map_to_vec`.
#[inline]
pub fn map<T,IT: BaseIter<T>,U,BU: Buildable<U>>(v: &IT, f: &fn(&T) -> U)
    -> BU {
    do build_sized_opt(v.size_hint()) |push| {
        for v.each() |elem| {
            push(f(elem));
        }
    }
}

/**
 * Creates and initializes a generic sequence from a function.
 *
 * Creates a generic sequence of size `n_elts` and initializes the elements
 * to the value returned by the function `op`.
 */
#[inline]
pub fn from_fn<T,BT: Buildable<T>>(n_elts: uint, op: InitOp<T>) -> BT {
    do Buildable::build_sized(n_elts) |push| {
        let mut i: uint = 0u;
        while i < n_elts { push(op(i)); i += 1u; }
    }
}

/**
 * Creates and initializes a generic sequence with some elements.
 *
 * Creates an immutable vector of size `n_elts` and initializes the elements
 * to the value `t`.
 */
#[inline]
pub fn from_elem<T:Copy,BT:Buildable<T>>(n_elts: uint, t: T) -> BT {
    do Buildable::build_sized(n_elts) |push| {
        let mut i: uint = 0;
        while i < n_elts { push(copy t); i += 1; }
    }
}

/// Appends two generic sequences.
#[inline]
pub fn append<T:Copy,IT:BaseIter<T>,BT:Buildable<T>>(lhs: &IT, rhs: &IT)
                                                  -> BT {
    let size_opt = lhs.size_hint().chain_ref(
        |sz1| rhs.size_hint().map(|sz2| *sz1+*sz2));
    do build_sized_opt(size_opt) |push| {
        for lhs.each |x| { push(copy *x); }
        for rhs.each |x| { push(copy *x); }
    }
}

/// Copies a generic sequence, possibly converting it to a different
/// type of sequence.
#[inline]
pub fn copy_seq<T:Copy,IT:BaseIter<T>,BT:Buildable<T>>(v: &IT) -> BT {
    do build_sized_opt(v.size_hint()) |push| {
        for v.each |x| { push(copy *x); }
    }
}
