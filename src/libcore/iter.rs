/*!

The iteration traits and common implementation

*/

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cmp::{Eq, Ord};

/// A function used to initialize the elements of a sequence
pub type InitOp<T> = &fn(uint) -> T;

pub trait BaseIter<A> {
    pure fn each(blk: fn(v: &A) -> bool);
    pure fn size_hint() -> Option<uint>;
}

pub trait ExtendedIter<A> {
    pure fn eachi(blk: fn(uint, v: &A) -> bool);
    pure fn all(blk: fn(&A) -> bool) -> bool;
    pure fn any(blk: fn(&A) -> bool) -> bool;
    pure fn foldl<B>(b0: B, blk: fn(&B, &A) -> B) -> B;
    pure fn position(f: fn(&A) -> bool) -> Option<uint>;
}

pub trait EqIter<A:Eq> {
    pure fn contains(x: &A) -> bool;
    pure fn count(x: &A) -> uint;
}

pub trait Times {
    pure fn times(it: fn() -> bool);
}

pub trait CopyableIter<A:Copy> {
    pure fn filter_to_vec(pred: fn(a: A) -> bool) -> ~[A];
    pure fn map_to_vec<B>(op: fn(v: A) -> B) -> ~[B];
    pure fn flat_map_to_vec<B:Copy,IB: BaseIter<B>>(op: fn(A) -> IB) -> ~[B];
    pure fn to_vec() -> ~[A];
    pure fn find(p: fn(a: A) -> bool) -> Option<A>;
}

pub trait CopyableOrderedIter<A:Copy Ord> {
    pure fn min() -> A;
    pure fn max() -> A;
}

pub trait CopyableNonstrictIter<A:Copy> {
    // Like "each", but copies out the value. If the receiver is mutated while
    // iterating over it, the semantics must not be memory-unsafe but are
    // otherwise undefined.
    pure fn each_val(&const self, f: &fn(A) -> bool);
}

// A trait for sequences that can be by imperatively pushing elements
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
     * * builder - A function that will construct the sequence. It recieves
     *             as an argument a function that will push an element
     *             onto the sequence being constructed.
     */
     static pure fn build_sized(size: uint,
                                builder: fn(push: pure fn(v: A))) -> self;
}

pub pure fn eachi<A,IA:BaseIter<A>>(self: &IA,
                                    blk: fn(uint, v: &A) -> bool) {
    let mut i = 0;
    for self.each |a| {
        if !blk(i, a) { break; }
        i += 1;
    }
}

pub pure fn all<A,IA:BaseIter<A>>(self: &IA,
                                  blk: fn(&A) -> bool) -> bool {
    for self.each |a| {
        if !blk(a) { return false; }
    }
    return true;
}

pub pure fn any<A,IA:BaseIter<A>>(self: &IA,
                                  blk: fn(&A) -> bool) -> bool {
    for self.each |a| {
        if blk(a) { return true; }
    }
    return false;
}

pub pure fn filter_to_vec<A:Copy,IA:BaseIter<A>>(
    self: &IA, prd: fn(a: A) -> bool) -> ~[A] {
    do vec::build_sized_opt(self.size_hint()) |push| {
        for self.each |a| {
            if prd(*a) { push(*a); }
        }
    }
}

pub pure fn map_to_vec<A:Copy,B,IA:BaseIter<A>>(self: &IA,
                                                op: fn(v: A) -> B)
    -> ~[B] {
    do vec::build_sized_opt(self.size_hint()) |push| {
        for self.each |a| {
            push(op(*a));
        }
    }
}

pub pure fn flat_map_to_vec<A:Copy,B:Copy,IA:BaseIter<A>,IB:BaseIter<B>>(
    self: &IA, op: fn(a: A) -> IB) -> ~[B] {
    do vec::build |push| {
        for self.each |a| {
            for op(*a).each |b| {
                push(*b);
            }
        }
    }
}

pub pure fn foldl<A,B,IA:BaseIter<A>>(self: &IA, b0: B,
                                      blk: fn(&B, &A) -> B)
    -> B {
    let mut b = move b0;
    for self.each |a| {
        b = blk(&b, a);
    }
    move b
}

pub pure fn to_vec<A:Copy,IA:BaseIter<A>>(self: &IA) -> ~[A] {
    foldl::<A,~[A],IA>(self, ~[], |r, a| vec::append(copy (*r), ~[*a]))
}

pub pure fn contains<A:Eq,IA:BaseIter<A>>(self: &IA, x: &A) -> bool {
    for self.each |a| {
        if *a == *x { return true; }
    }
    return false;
}

pub pure fn count<A:Eq,IA:BaseIter<A>>(self: &IA, x: &A) -> uint {
    do foldl(self, 0) |count, value| {
        if *value == *x {
            *count + 1
        } else {
            *count
        }
    }
}

pub pure fn position<A,IA:BaseIter<A>>(self: &IA, f: fn(&A) -> bool)
    -> Option<uint>
{
    let mut i = 0;
    for self.each |a| {
        if f(a) { return Some(i); }
        i += 1;
    }
    return None;
}

// note: 'rposition' would only make sense to provide with a bidirectional
// iter interface, such as would provide "reach" in addition to "each". as is,
// it would have to be implemented with foldr, which is too inefficient.

pub pure fn repeat(times: uint, blk: fn() -> bool) {
    let mut i = 0;
    while i < times {
        if !blk() { break }
        i += 1;
    }
}

pub pure fn min<A:Copy Ord,IA:BaseIter<A>>(self: &IA) -> A {
    match do foldl::<A,Option<A>,IA>(self, None) |a, b| {
        match a {
          &Some(ref a_) if *a_ < *b => {
             *(move a)
          }
          _ => Some(*b)
        }
    } {
        Some(move val) => val,
        None => fail ~"min called on empty iterator"
    }
}

pub pure fn max<A:Copy Ord,IA:BaseIter<A>>(self: &IA) -> A {
    match do foldl::<A,Option<A>,IA>(self, None) |a, b| {
        match a {
          &Some(ref a_) if *a_ > *b => {
              *(move a)
          }
          _ => Some(*b)
        }
    } {
        Some(move val) => val,
        None => fail ~"max called on empty iterator"
    }
}

pub pure fn find<A: Copy,IA:BaseIter<A>>(self: &IA,
                                     p: fn(a: A) -> bool) -> Option<A> {
    for self.each |i| {
        if p(*i) { return Some(*i) }
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
 * * builder - A function that will construct the sequence. It recieves
 *             as an argument a function that will push an element
 *             onto the sequence being constructed.
 */
#[inline(always)]
pub pure fn build<A,B: Buildable<A>>(builder: fn(push: pure fn(v: A)))
    -> B {
    build_sized(4, builder)
}

/**
 * Builds a sequence by calling a provided function with an argument
 * function that pushes an element to the back of a sequence.
 * This version takes an initial size for the sequence.
 *
 * # Arguments
 *
 * * size - An option, maybe containing initial size of the sequence
 *          to reserve
 * * builder - A function that will construct the sequence. It recieves
 *             as an argument a function that will push an element
 *             onto the sequence being constructed.
 */
#[inline(always)]
pub pure fn build_sized_opt<A,B: Buildable<A>>(
    size: Option<uint>,
    builder: fn(push: pure fn(v: A))) -> B {

    build_sized(size.get_default(4), builder)
}

// Functions that combine iteration and building

/// Apply a function to each element of an iterable and return the results
pub fn map<T,IT: BaseIter<T>,U,BU: Buildable<U>>(v: &IT, f: fn(&T) -> U)
    -> BU {
    do build_sized_opt(v.size_hint()) |push| {
        for v.each() |elem| {
            push(f(elem));
        }
    }
}

/**
 * Creates and initializes a generic sequence from a function
 *
 * Creates a generic sequence of size `n_elts` and initializes the elements
 * to the value returned by the function `op`.
 */
pub pure fn from_fn<T,BT: Buildable<T>>(n_elts: uint,
                                        op: InitOp<T>) -> BT {
    do build_sized(n_elts) |push| {
        let mut i: uint = 0u;
        while i < n_elts { push(op(i)); i += 1u; }
    }
}

/**
 * Creates and initializes a generic sequence with some element
 *
 * Creates an immutable vector of size `n_elts` and initializes the elements
 * to the value `t`.
 */
pub pure fn from_elem<T: Copy,BT: Buildable<T>>(n_elts: uint,
                                                t: T) -> BT {
    do build_sized(n_elts) |push| {
        let mut i: uint = 0;
        while i < n_elts { push(t); i += 1; }
    }
}

/// Appending two generic sequences
#[inline(always)]
pub pure fn append<T: Copy,IT: BaseIter<T>,BT: Buildable<T>>(
    lhs: &IT, rhs: &IT) -> BT {
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
pub pure fn copy_seq<T: Copy,IT: BaseIter<T>,BT: Buildable<T>>(
    v: &IT) -> BT {
    do build_sized_opt(v.size_hint()) |push| {
        for v.each |x| { push(*x); }
    }
}
