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

use cmp::{Eq, Ord};
use kinds::Copy;
use option::{None, Option, Some};
use vec;

/// A function used to initialize the elements of a sequence
pub type InitOp<'self,T> = &'self fn(uint) -> T;

#[cfg(stage0)]
pub trait BaseIter<A> {
    fn each(&self, blk: &fn(v: &A) -> bool);
    fn size_hint(&self) -> Option<uint>;
}
#[cfg(not(stage0))]
pub trait BaseIter<A> {
    fn each(&self, blk: &fn(v: &A) -> bool) -> bool;
    fn size_hint(&self) -> Option<uint>;
}

#[cfg(stage0)]
pub trait ReverseIter<A>: BaseIter<A> {
    fn each_reverse(&self, blk: &fn(&A) -> bool);
}
#[cfg(not(stage0))]
pub trait ReverseIter<A>: BaseIter<A> {
    fn each_reverse(&self, blk: &fn(&A) -> bool) -> bool;
}

#[cfg(stage0)]
pub trait MutableIter<A>: BaseIter<A> {
    fn each_mut(&mut self, blk: &fn(&mut A) -> bool);
}
#[cfg(not(stage0))]
pub trait MutableIter<A>: BaseIter<A> {
    fn each_mut(&mut self, blk: &fn(&mut A) -> bool) -> bool;
}

pub trait ExtendedIter<A> {
    #[cfg(stage0)]
    fn eachi(&self, blk: &fn(uint, v: &A) -> bool);
    #[cfg(not(stage0))]
    fn eachi(&self, blk: &fn(uint, v: &A) -> bool) -> bool;
    fn all(&self, blk: &fn(&A) -> bool) -> bool;
    fn any(&self, blk: &fn(&A) -> bool) -> bool;
    fn foldl<B>(&self, b0: B, blk: &fn(&B, &A) -> B) -> B;
    fn position(&self, f: &fn(&A) -> bool) -> Option<uint>;
    fn map_to_vec<B>(&self, op: &fn(&A) -> B) -> ~[B];
    fn flat_map_to_vec<B,IB: BaseIter<B>>(&self, op: &fn(&A) -> IB) -> ~[B];
}

#[cfg(stage0)]
pub trait ExtendedMutableIter<A> {
    fn eachi_mut(&mut self, blk: &fn(uint, &mut A) -> bool);
}
#[cfg(not(stage0))]
pub trait ExtendedMutableIter<A> {
    fn eachi_mut(&mut self, blk: &fn(uint, &mut A) -> bool) -> bool;
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

pub trait CopyableOrderedIter<A:Copy + Ord> {
    fn min(&self) -> A;
    fn max(&self) -> A;
}

pub trait CopyableNonstrictIter<A:Copy> {
    // Like "each", but copies out the value. If the receiver is mutated while
    // iterating over it, the semantics must not be memory-unsafe but are
    // otherwise undefined.
    fn each_val(&const self, f: &fn(A) -> bool);
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

#[inline(always)]
pub fn _eachi<A,IA:BaseIter<A>>(self: &IA, blk: &fn(uint, &A) -> bool) -> bool {
    let mut i = 0;
    for self.each |a| {
        if !blk(i, a) { return false; }
        i += 1;
    }
    return true;
}

#[cfg(stage0)]
pub fn eachi<A,IA:BaseIter<A>>(self: &IA, blk: &fn(uint, &A) -> bool) {
    _eachi(self, blk);
}
#[cfg(not(stage0))]
pub fn eachi<A,IA:BaseIter<A>>(self: &IA, blk: &fn(uint, &A) -> bool) -> bool {
    _eachi(self, blk)
}

#[inline(always)]
pub fn all<A,IA:BaseIter<A>>(self: &IA, blk: &fn(&A) -> bool) -> bool {
    for self.each |a| {
        if !blk(a) { return false; }
    }
    return true;
}

#[inline(always)]
pub fn any<A,IA:BaseIter<A>>(self: &IA, blk: &fn(&A) -> bool) -> bool {
    for self.each |a| {
        if blk(a) { return true; }
    }
    return false;
}

#[inline(always)]
pub fn filter_to_vec<A:Copy,IA:BaseIter<A>>(self: &IA,
                                            prd: &fn(&A) -> bool)
                                         -> ~[A] {
    do vec::build_sized_opt(self.size_hint()) |push| {
        for self.each |a| {
            if prd(a) { push(*a); }
        }
    }
}

#[inline(always)]
pub fn map_to_vec<A,B,IA:BaseIter<A>>(self: &IA, op: &fn(&A) -> B) -> ~[B] {
    do vec::build_sized_opt(self.size_hint()) |push| {
        for self.each |a| {
            push(op(a));
        }
    }
}

#[inline(always)]
pub fn flat_map_to_vec<A,B,IA:BaseIter<A>,IB:BaseIter<B>>(self: &IA,
                                                          op: &fn(&A) -> IB)
                                                       -> ~[B] {
    do vec::build |push| {
        for self.each |a| {
            for op(a).each |&b| {
                push(b);
            }
        }
    }
}

#[inline(always)]
pub fn foldl<A,B,IA:BaseIter<A>>(self: &IA, b0: B, blk: &fn(&B, &A) -> B)
                              -> B {
    let mut b = b0;
    for self.each |a| {
        b = blk(&b, a);
    }
    b
}

#[inline(always)]
pub fn to_vec<A:Copy,IA:BaseIter<A>>(self: &IA) -> ~[A] {
    map_to_vec(self, |&x| x)
}

#[inline(always)]
pub fn contains<A:Eq,IA:BaseIter<A>>(self: &IA, x: &A) -> bool {
    for self.each |a| {
        if *a == *x { return true; }
    }
    return false;
}

#[inline(always)]
pub fn count<A:Eq,IA:BaseIter<A>>(self: &IA, x: &A) -> uint {
    do foldl(self, 0) |count, value| {
        if *value == *x {
            *count + 1
        } else {
            *count
        }
    }
}

#[inline(always)]
pub fn position<A,IA:BaseIter<A>>(self: &IA, f: &fn(&A) -> bool)
                               -> Option<uint> {
    let mut i = 0;
    for self.each |a| {
        if f(a) { return Some(i); }
        i += 1;
    }
    return None;
}

// note: 'rposition' would only make sense to provide with a bidirectional
// iter interface, such as would provide "reach" in addition to "each". As is,
// it would have to be implemented with foldr, which is too inefficient.

#[inline(always)]
#[cfg(stage0)]
pub fn repeat(times: uint, blk: &fn() -> bool) {
    let mut i = 0;
    while i < times {
        if !blk() { break }
        i += 1;
    }
}
#[inline(always)]
#[cfg(not(stage0))]
pub fn repeat(times: uint, blk: &fn() -> bool) -> bool {
    let mut i = 0;
    while i < times {
        if !blk() { return false; }
        i += 1;
    }
    return true;
}

#[inline(always)]
pub fn min<A:Copy + Ord,IA:BaseIter<A>>(self: &IA) -> A {
    match do foldl::<A,Option<A>,IA>(self, None) |a, b| {
        match a {
          &Some(ref a_) if *a_ < *b => {
             *(a)
          }
          _ => Some(*b)
        }
    } {
        Some(val) => val,
        None => fail!(~"min called on empty iterator")
    }
}

#[inline(always)]
pub fn max<A:Copy + Ord,IA:BaseIter<A>>(self: &IA) -> A {
    match do foldl::<A,Option<A>,IA>(self, None) |a, b| {
        match a {
          &Some(ref a_) if *a_ > *b => {
              *(a)
          }
          _ => Some(*b)
        }
    } {
        Some(val) => val,
        None => fail!(~"max called on empty iterator")
    }
}

#[inline(always)]
pub fn find<A:Copy,IA:BaseIter<A>>(self: &IA, f: &fn(&A) -> bool)
                                -> Option<A> {
    for self.each |i| {
        if f(i) { return Some(*i) }
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
#[inline(always)]
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
#[inline(always)]
pub fn build_sized_opt<A,B: Buildable<A>>(size: Option<uint>,
                                          builder: &fn(push: &fn(A))) -> B {
    Buildable::build_sized(size.get_or_default(4), builder)
}

// Functions that combine iteration and building

/// Applies a function to each element of an iterable and returns the results
/// in a sequence built via `BU`.  See also `map_to_vec`.
#[inline(always)]
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
#[inline(always)]
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
#[inline(always)]
pub fn from_elem<T:Copy,BT:Buildable<T>>(n_elts: uint, t: T) -> BT {
    do Buildable::build_sized(n_elts) |push| {
        let mut i: uint = 0;
        while i < n_elts { push(t); i += 1; }
    }
}

/// Appends two generic sequences.
#[inline(always)]
pub fn append<T:Copy,IT:BaseIter<T>,BT:Buildable<T>>(lhs: &IT, rhs: &IT)
                                                  -> BT {
    let size_opt = lhs.size_hint().chain_ref(
        |sz1| rhs.size_hint().map(|sz2| *sz1+*sz2));
    do build_sized_opt(size_opt) |push| {
        for lhs.each |x| { push(*x); }
        for rhs.each |x| { push(*x); }
    }
}

/// Copies a generic sequence, possibly converting it to a different
/// type of sequence.
#[inline(always)]
pub fn copy_seq<T:Copy,IT:BaseIter<T>,BT:Buildable<T>>(v: &IT) -> BT {
    do build_sized_opt(v.size_hint()) |push| {
        for v.each |x| { push(*x); }
    }
}
