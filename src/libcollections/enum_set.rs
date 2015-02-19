// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A structure for holding a set of enum variants.
//!
//! This module defines a container which uses an efficient bit mask
//! representation to hold C-like enum variants.

use core::prelude::*;
use core::marker;
use core::fmt;
use core::num::Int;
use core::iter::{FromIterator, IntoIterator};
use core::ops::{Sub, BitOr, BitAnd, BitXor};

// FIXME(contentions): implement union family of methods? (general design may be wrong here)

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// A specialized set implementation to use enum types.
pub struct EnumSet<E> {
    // We must maintain the invariant that no bits are set
    // for which no variant exists
    bits: usize,
    marker: marker::PhantomData<E>,
}

impl<E> Copy for EnumSet<E> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<E:CLike + fmt::Debug> fmt::Debug for EnumSet<E> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(fmt, "EnumSet {{"));
        let mut first = true;
        for e in self {
            if !first {
                try!(write!(fmt, ", "));
            }
            try!(write!(fmt, "{:?}", e));
            first = false;
        }
        write!(fmt, "}}")
    }
}

/// An interface for casting C-like enum to usize and back.
/// A typically implementation is as below.
///
/// ```{rust,ignore}
/// #[repr(usize)]
/// enum Foo {
///     A, B, C
/// }
///
/// impl CLike for Foo {
///     fn to_usize(&self) -> usize {
///         *self as usize
///     }
///
///     fn from_usize(v: usize) -> Foo {
///         unsafe { mem::transmute(v) }
///     }
/// }
/// ```
pub trait CLike {
    /// Converts a C-like enum to a `usize`.
    fn to_usize(&self) -> usize;
    /// Converts a `usize` to a C-like enum.
    fn from_usize(usize) -> Self;
}

fn bit<E:CLike>(e: &E) -> usize {
    use core::usize;
    let value = e.to_usize();
    assert!(value < usize::BITS,
            "EnumSet only supports up to {} variants.", usize::BITS - 1);
    1 << value
}

impl<E:CLike> EnumSet<E> {
    /// Returns an empty `EnumSet`.
    #[unstable(feature = "collections",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn new() -> EnumSet<E> {
        EnumSet {bits: 0, marker: marker::PhantomData}
    }

    /// Returns the number of elements in the given `EnumSet`.
    #[unstable(feature = "collections",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn len(&self) -> usize {
        self.bits.count_ones()
    }

    /// Returns true if the `EnumSet` is empty.
    #[unstable(feature = "collections",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }

    pub fn clear(&mut self) {
        self.bits = 0;
    }

    /// Returns `false` if the `EnumSet` contains any enum of the given `EnumSet`.
    #[unstable(feature = "collections",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn is_disjoint(&self, other: &EnumSet<E>) -> bool {
        (self.bits & other.bits) == 0
    }

    /// Returns `true` if a given `EnumSet` is included in this `EnumSet`.
    #[unstable(feature = "collections",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn is_superset(&self, other: &EnumSet<E>) -> bool {
        (self.bits & other.bits) == other.bits
    }

    /// Returns `true` if this `EnumSet` is included in the given `EnumSet`.
    #[unstable(feature = "collections",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn is_subset(&self, other: &EnumSet<E>) -> bool {
        other.is_superset(self)
    }

    /// Returns the union of both `EnumSets`.
    pub fn union(&self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits | e.bits,
                 marker: marker::PhantomData}
    }

    /// Returns the intersection of both `EnumSets`.
    pub fn intersection(&self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits & e.bits,
                 marker: marker::PhantomData}
    }

    /// Adds an enum to the `EnumSet`, and returns `true` if it wasn't there before
    #[unstable(feature = "collections",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn insert(&mut self, e: E) -> bool {
        let result = !self.contains(&e);
        self.bits |= bit(&e);
        result
    }

    /// Removes an enum from the EnumSet
    #[unstable(feature = "collections",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn remove(&mut self, e: &E) -> bool {
        let result = self.contains(e);
        self.bits &= !bit(e);
        result
    }

    /// Returns `true` if an `EnumSet` contains a given enum.
    #[unstable(feature = "collections",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn contains(&self, e: &E) -> bool {
        (self.bits & bit(e)) != 0
    }

    /// Returns an iterator over an `EnumSet`.
    #[unstable(feature = "collections",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn iter(&self) -> Iter<E> {
        Iter::new(self.bits)
    }
}

impl<E:CLike> Sub for EnumSet<E> {
    type Output = EnumSet<E>;

    fn sub(self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits & !e.bits, marker: marker::PhantomData}
    }
}

impl<E:CLike> BitOr for EnumSet<E> {
    type Output = EnumSet<E>;

    fn bitor(self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits | e.bits, marker: marker::PhantomData}
    }
}

impl<E:CLike> BitAnd for EnumSet<E> {
    type Output = EnumSet<E>;

    fn bitand(self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits & e.bits, marker: marker::PhantomData}
    }
}

impl<E:CLike> BitXor for EnumSet<E> {
    type Output = EnumSet<E>;

    fn bitxor(self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits ^ e.bits, marker: marker::PhantomData}
    }
}

/// An iterator over an EnumSet
pub struct Iter<E> {
    index: usize,
    bits: usize,
    marker: marker::PhantomData<E>,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
impl<E> Clone for Iter<E> {
    fn clone(&self) -> Iter<E> {
        Iter {
            index: self.index,
            bits: self.bits,
            marker: marker::PhantomData,
        }
    }
}

impl<E:CLike> Iter<E> {
    fn new(bits: usize) -> Iter<E> {
        Iter { index: 0, bits: bits, marker: marker::PhantomData }
    }
}

impl<E:CLike> Iterator for Iter<E> {
    type Item = E;

    fn next(&mut self) -> Option<E> {
        if self.bits == 0 {
            return None;
        }

        while (self.bits & 1) == 0 {
            self.index += 1;
            self.bits >>= 1;
        }
        let elem = CLike::from_usize(self.index);
        self.index += 1;
        self.bits >>= 1;
        Some(elem)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.bits.count_ones();
        (exact, Some(exact))
    }
}

impl<E:CLike> FromIterator<E> for EnumSet<E> {
    fn from_iter<I: IntoIterator<Item=E>>(iter: I) -> EnumSet<E> {
        let mut ret = EnumSet::new();
        ret.extend(iter);
        ret
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, E> IntoIterator for &'a EnumSet<E> where E: CLike {
    type Item = E;
    type IntoIter = Iter<E>;

    fn into_iter(self) -> Iter<E> {
        self.iter()
    }
}

impl<E:CLike> Extend<E> for EnumSet<E> {
    fn extend<I: IntoIterator<Item=E>>(&mut self, iter: I) {
        for element in iter {
            self.insert(element);
        }
    }
}

#[cfg(test)]
mod test {
    use self::Foo::*;
    use prelude::*;
    use core::mem;

    use super::{EnumSet, CLike};

    #[derive(Copy, PartialEq, Debug)]
    #[repr(usize)]
    enum Foo {
        A, B, C
    }

    impl CLike for Foo {
        fn to_usize(&self) -> usize {
            *self as usize
        }

        fn from_usize(v: usize) -> Foo {
            unsafe { mem::transmute(v) }
        }
    }

    #[test]
    fn test_new() {
        let e: EnumSet<Foo> = EnumSet::new();
        assert!(e.is_empty());
    }

    #[test]
    fn test_show() {
        let mut e = EnumSet::new();
        assert!(format!("{:?}", e) == "EnumSet {}");
        e.insert(A);
        assert!(format!("{:?}", e) == "EnumSet {A}");
        e.insert(C);
        assert!(format!("{:?}", e) == "EnumSet {A, C}");
    }

    #[test]
    fn test_len() {
        let mut e = EnumSet::new();
        assert_eq!(e.len(), 0);
        e.insert(A);
        e.insert(B);
        e.insert(C);
        assert_eq!(e.len(), 3);
        e.remove(&A);
        assert_eq!(e.len(), 2);
        e.clear();
        assert_eq!(e.len(), 0);
    }

    ///////////////////////////////////////////////////////////////////////////
    // intersect

    #[test]
    fn test_two_empties_do_not_intersect() {
        let e1: EnumSet<Foo> = EnumSet::new();
        let e2: EnumSet<Foo> = EnumSet::new();
        assert!(e1.is_disjoint(&e2));
    }

    #[test]
    fn test_empty_does_not_intersect_with_full() {
        let e1: EnumSet<Foo> = EnumSet::new();

        let mut e2: EnumSet<Foo> = EnumSet::new();
        e2.insert(A);
        e2.insert(B);
        e2.insert(C);

        assert!(e1.is_disjoint(&e2));
    }

    #[test]
    fn test_disjoint_intersects() {
        let mut e1: EnumSet<Foo> = EnumSet::new();
        e1.insert(A);

        let mut e2: EnumSet<Foo> = EnumSet::new();
        e2.insert(B);

        assert!(e1.is_disjoint(&e2));
    }

    #[test]
    fn test_overlapping_intersects() {
        let mut e1: EnumSet<Foo> = EnumSet::new();
        e1.insert(A);

        let mut e2: EnumSet<Foo> = EnumSet::new();
        e2.insert(A);
        e2.insert(B);

        assert!(!e1.is_disjoint(&e2));
    }

    ///////////////////////////////////////////////////////////////////////////
    // contains and contains_elem

    #[test]
    fn test_superset() {
        let mut e1: EnumSet<Foo> = EnumSet::new();
        e1.insert(A);

        let mut e2: EnumSet<Foo> = EnumSet::new();
        e2.insert(A);
        e2.insert(B);

        let mut e3: EnumSet<Foo> = EnumSet::new();
        e3.insert(C);

        assert!(e1.is_subset(&e2));
        assert!(e2.is_superset(&e1));
        assert!(!e3.is_superset(&e2));
        assert!(!e2.is_superset(&e3))
    }

    #[test]
    fn test_contains() {
        let mut e1: EnumSet<Foo> = EnumSet::new();
        e1.insert(A);
        assert!(e1.contains(&A));
        assert!(!e1.contains(&B));
        assert!(!e1.contains(&C));

        e1.insert(A);
        e1.insert(B);
        assert!(e1.contains(&A));
        assert!(e1.contains(&B));
        assert!(!e1.contains(&C));
    }

    ///////////////////////////////////////////////////////////////////////////
    // iter

    #[test]
    fn test_iterator() {
        let mut e1: EnumSet<Foo> = EnumSet::new();

        let elems: ::vec::Vec<Foo> = e1.iter().collect();
        assert!(elems.is_empty());

        e1.insert(A);
        let elems: ::vec::Vec<_> = e1.iter().collect();
        assert_eq!(vec![A], elems);

        e1.insert(C);
        let elems: ::vec::Vec<_> = e1.iter().collect();
        assert_eq!(vec![A,C], elems);

        e1.insert(C);
        let elems: ::vec::Vec<_> = e1.iter().collect();
        assert_eq!(vec![A,C], elems);

        e1.insert(B);
        let elems: ::vec::Vec<_> = e1.iter().collect();
        assert_eq!(vec![A,B,C], elems);
    }

    ///////////////////////////////////////////////////////////////////////////
    // operators

    #[test]
    fn test_operators() {
        let mut e1: EnumSet<Foo> = EnumSet::new();
        e1.insert(A);
        e1.insert(C);

        let mut e2: EnumSet<Foo> = EnumSet::new();
        e2.insert(B);
        e2.insert(C);

        let e_union = e1 | e2;
        let elems: ::vec::Vec<_> = e_union.iter().collect();
        assert_eq!(vec![A,B,C], elems);

        let e_intersection = e1 & e2;
        let elems: ::vec::Vec<_> = e_intersection.iter().collect();
        assert_eq!(vec![C], elems);

        // Another way to express intersection
        let e_intersection = e1 - (e1 - e2);
        let elems: ::vec::Vec<_> = e_intersection.iter().collect();
        assert_eq!(vec![C], elems);

        let e_subtract = e1 - e2;
        let elems: ::vec::Vec<_> = e_subtract.iter().collect();
        assert_eq!(vec![A], elems);

        // Bitwise XOR of two sets, aka symmetric difference
        let e_symmetric_diff = e1 ^ e2;
        let elems: ::vec::Vec<_> = e_symmetric_diff.iter().collect();
        assert_eq!(vec![A,B], elems);

        // Another way to express symmetric difference
        let e_symmetric_diff = (e1 - e2) | (e2 - e1);
        let elems: ::vec::Vec<_> = e_symmetric_diff.iter().collect();
        assert_eq!(vec![A,B], elems);

        // Yet another way to express symmetric difference
        let e_symmetric_diff = (e1 | e2) - (e1 & e2);
        let elems: ::vec::Vec<_> = e_symmetric_diff.iter().collect();
        assert_eq!(vec![A,B], elems);
    }

    #[test]
    #[should_fail]
    fn test_overflow() {
        #[allow(dead_code)]
        #[derive(Copy)]
        #[repr(usize)]
        enum Bar {
            V00, V01, V02, V03, V04, V05, V06, V07, V08, V09,
            V10, V11, V12, V13, V14, V15, V16, V17, V18, V19,
            V20, V21, V22, V23, V24, V25, V26, V27, V28, V29,
            V30, V31, V32, V33, V34, V35, V36, V37, V38, V39,
            V40, V41, V42, V43, V44, V45, V46, V47, V48, V49,
            V50, V51, V52, V53, V54, V55, V56, V57, V58, V59,
            V60, V61, V62, V63, V64, V65, V66, V67, V68, V69,
        }

        impl CLike for Bar {
            fn to_usize(&self) -> usize {
                *self as usize
            }

            fn from_usize(v: usize) -> Bar {
                unsafe { mem::transmute(v) }
            }
        }
        let mut set = EnumSet::new();
        set.insert(Bar::V64);
    }
}
