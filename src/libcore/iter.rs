/*!

The iteration traits and common implementation

*/

use cmp::{Eq, Ord};

/// A function used to initialize the elements of a sequence
type InitOp<T> = &fn(uint) -> T;

trait BaseIter<A> {
    pure fn each(blk: fn(v: &A) -> bool);
    pure fn size_hint() -> Option<uint>;
}

trait ExtendedIter<A> {
    pure fn eachi(blk: fn(uint, v: &A) -> bool);
    pure fn all(blk: fn(A) -> bool) -> bool;
    pure fn any(blk: fn(A) -> bool) -> bool;
    pure fn foldl<B>(+b0: B, blk: fn(B, A) -> B) -> B;
    pure fn position(f: fn(A) -> bool) -> Option<uint>;
}

trait EqIter<A:Eq> {
    pure fn contains(x: &A) -> bool;
    pure fn count(x: &A) -> uint;
}

trait Times {
    pure fn times(it: fn() -> bool);
}
trait TimesIx{
    pure fn timesi(it: fn(uint) -> bool);
}

trait CopyableIter<A:Copy> {
    pure fn filter_to_vec(pred: fn(A) -> bool) -> ~[A];
    pure fn map_to_vec<B>(op: fn(v: &A) -> B) -> ~[B];
    pure fn to_vec() -> ~[A];
    pure fn find(p: fn(A) -> bool) -> Option<A>;
}

trait CopyableOrderedIter<A:Copy Ord> {
    pure fn min() -> A;
    pure fn max() -> A;
}

// A trait for sequences that can be by imperatively pushing elements
// onto them.
trait Buildable<A> {
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
                                builder: fn(push: pure fn(+v: A))) -> self;
}

pure fn eachi<A,IA:BaseIter<A>>(self: &IA, blk: fn(uint, v: &A) -> bool) {
    let mut i = 0;
    for self.each |a| {
        if !blk(i, a) { break; }
        i += 1;
    }
}

pure fn all<A,IA:BaseIter<A>>(self: IA, blk: fn(A) -> bool) -> bool {
    for self.each |a| {
        if !blk(*a) { return false; }
    }
    return true;
}

pure fn any<A,IA:BaseIter<A>>(self: IA, blk: fn(A) -> bool) -> bool {
    for self.each |a| {
        if blk(*a) { return true; }
    }
    return false;
}

pure fn filter_to_vec<A:Copy,IA:BaseIter<A>>(self: IA,
                                         prd: fn(A) -> bool) -> ~[A] {
    do vec::build_sized_opt(self.size_hint()) |push| {
        for self.each |a| {
            if prd(*a) { push(*a); }
        }
    }
}

pure fn map_to_vec<A:Copy,B,IA:BaseIter<A>>(self: IA, op: fn(v: &A) -> B)
    -> ~[B] {
    do vec::build_sized_opt(self.size_hint()) |push| {
        for self.each |a| {
            push(op(a));
        }
    }
}

pure fn flat_map_to_vec<A:Copy,B:Copy,IA:BaseIter<A>,IB:BaseIter<B>>(
    self: IA, op: fn(A) -> IB) -> ~[B] {

    do vec::build |push| {
        for self.each |a| {
            for op(*a).each |b| {
                push(*b);
            }
        }
    }
}

pure fn foldl<A,B,IA:BaseIter<A>>(self: IA, +b0: B, blk: fn(B, A) -> B) -> B {
    let mut b <- b0;
    for self.each |a| {
        b = blk(b, *a);
    }
    move b
}

pure fn to_vec<A:Copy,IA:BaseIter<A>>(self: IA) -> ~[A] {
    foldl::<A,~[A],IA>(self, ~[], |r, a| vec::append(copy r, ~[a]))
}

pure fn contains<A:Eq,IA:BaseIter<A>>(self: IA, x: &A) -> bool {
    for self.each |a| {
        if *a == *x { return true; }
    }
    return false;
}

pure fn count<A:Eq,IA:BaseIter<A>>(self: IA, x: &A) -> uint {
    do foldl(self, 0) |count, value| {
        if value == *x {
            count + 1
        } else {
            count
        }
    }
}

pure fn position<A,IA:BaseIter<A>>(self: IA, f: fn(A) -> bool)
    -> Option<uint>
{
    let mut i = 0;
    for self.each |a| {
        if f(*a) { return Some(i); }
        i += 1;
    }
    return None;
}

// note: 'rposition' would only make sense to provide with a bidirectional
// iter interface, such as would provide "reach" in addition to "each". as is,
// it would have to be implemented with foldr, which is too inefficient.

pure fn repeat(times: uint, blk: fn() -> bool) {
    let mut i = 0u;
    while i < times {
        if !blk() { break }
        i += 1u;
    }
}

// XXX bad copies
pure fn min<A:Copy Ord,IA:BaseIter<A>>(self: IA) -> A {
    match do foldl::<A,Option<A>,IA>(self, None) |a, b| {
        match a {
          Some(copy a_) if a_ < b => {
            // FIXME (#2005): Not sure if this is successfully optimized to
            // a move
            a
          }
          _ => Some(b)
        }
    } {
        Some(move val) => val,
        None => fail ~"min called on empty iterator"
    }
}

// XXX bad copies
pure fn max<A:Copy Ord,IA:BaseIter<A>>(self: IA) -> A {
    match do foldl::<A,Option<A>,IA>(self, None) |a, b| {
        match a {
          Some(copy a_) if a_ > b => {
            // FIXME (#2005): Not sure if this is successfully optimized to
            // a move.
            a
          }
          _ => Some(b)
        }
    } {
        Some(move val) => val,
        None => fail ~"max called on empty iterator"
    }
}

pure fn find<A: Copy,IA:BaseIter<A>>(self: IA,
                                     p: fn(A) -> bool) -> Option<A> {
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
pure fn build<A,B: Buildable<A>>(builder: fn(push: pure fn(+v: A))) -> B {
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
pure fn build_sized_opt<A,B: Buildable<A>>(
    size: Option<uint>,
    builder: fn(push: pure fn(+v: A))) -> B {

    build_sized(size.get_default(4), builder)
}

// Functions that combine iteration and building

/// Apply a function to each element of an iterable and return the results
fn map<T,IT: BaseIter<T>,U,BU: Buildable<U>>(v: IT, f: fn(T) -> U) -> BU {
    do build_sized_opt(v.size_hint()) |push| {
        for v.each() |elem| {
            push(f(*elem));
        }
    }
}

/**
 * Creates and initializes a generic sequence from a function
 *
 * Creates a generic sequence of size `n_elts` and initializes the elements
 * to the value returned by the function `op`.
 */
pure fn from_fn<T,BT: Buildable<T>>(n_elts: uint, op: InitOp<T>) -> BT {
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
pure fn from_elem<T: Copy,BT: Buildable<T>>(n_elts: uint, t: T) -> BT {
    do build_sized(n_elts) |push| {
        let mut i: uint = 0u;
        while i < n_elts { push(t); i += 1u; }
    }
}

/// Appending two generic sequences
#[inline(always)]
pure fn append<T: Copy,IT: BaseIter<T>,BT: Buildable<T>>(
    lhs: IT, rhs: IT) -> BT {
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
pure fn copy_seq<T: Copy,IT: BaseIter<T>,BT: Buildable<T>>(
    v: IT) -> BT {
    do build_sized_opt(v.size_hint()) |push| {
        for v.each |x| { push(*x); }
    }
}
