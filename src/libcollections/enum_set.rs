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
use core::fmt;

#[deriving(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// A specialized `Set` implementation to use enum types.
pub struct EnumSet<E> {
    // We must maintain the invariant that no bits are set
    // for which no variant exists
    bits: uint
}

impl<E:CLike+fmt::Show> fmt::Show for EnumSet<E> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(fmt, "{{"));
        let mut first = true;
        for e in self.iter() {
            if !first {
                try!(write!(fmt, ", "));
            }
            try!(write!(fmt, "{}", e));
            first = false;
        }
        write!(fmt, "}}")
    }
}

/// An interface for casting C-like enum to uint and back.
pub trait CLike {
    /// Converts a C-like enum to a `uint`.
    fn to_uint(&self) -> uint;
    /// Converts a `uint` to a C-like enum.
    fn from_uint(uint) -> Self;
}

fn bit<E:CLike>(e: E) -> uint {
    1 << e.to_uint()
}

impl<E:CLike> EnumSet<E> {
    /// Returns an empty `EnumSet`.
    pub fn empty() -> EnumSet<E> {
        EnumSet {bits: 0}
    }

    /// Returns true if the `EnumSet` is empty.
    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }

    /// Returns `true` if the `EnumSet` contains any enum of the given `EnumSet`.
    pub fn intersects(&self, e: EnumSet<E>) -> bool {
        (self.bits & e.bits) != 0
    }

    /// Returns the intersection of both `EnumSets`.
    pub fn intersection(&self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits & e.bits}
    }

    /// Returns `true` if a given `EnumSet` is included in an `EnumSet`.
    pub fn contains(&self, e: EnumSet<E>) -> bool {
        (self.bits & e.bits) == e.bits
    }

    /// Returns the union of both `EnumSets`.
    pub fn union(&self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits | e.bits}
    }

    /// Adds an enum to an `EnumSet`.
    pub fn add(&mut self, e: E) {
        self.bits |= bit(e);
    }

    /// Returns `true` if an `EnumSet` contains a given enum.
    pub fn contains_elem(&self, e: E) -> bool {
        (self.bits & bit(e)) != 0
    }

    /// Returns an iterator over an `EnumSet`.
    pub fn iter(&self) -> Items<E> {
        Items::new(self.bits)
    }
}

impl<E:CLike> Sub<EnumSet<E>, EnumSet<E>> for EnumSet<E> {
    fn sub(&self, e: &EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits & !e.bits}
    }
}

impl<E:CLike> BitOr<EnumSet<E>, EnumSet<E>> for EnumSet<E> {
    fn bitor(&self, e: &EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits | e.bits}
    }
}

impl<E:CLike> BitAnd<EnumSet<E>, EnumSet<E>> for EnumSet<E> {
    fn bitand(&self, e: &EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits & e.bits}
    }
}

/// An iterator over an EnumSet
pub struct Items<E> {
    index: uint,
    bits: uint,
}

impl<E:CLike> Items<E> {
    fn new(bits: uint) -> Items<E> {
        Items { index: 0, bits: bits }
    }
}

impl<E:CLike> Iterator<E> for Items<E> {
    fn next(&mut self) -> Option<E> {
        if self.bits == 0 {
            return None;
        }

        while (self.bits & 1) == 0 {
            self.index += 1;
            self.bits >>= 1;
        }
        let elem = CLike::from_uint(self.index);
        self.index += 1;
        self.bits >>= 1;
        Some(elem)
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        let exact = self.bits.count_ones();
        (exact, Some(exact))
    }
}

#[cfg(test)]
mod test {
    use std::prelude::*;
    use std::mem;

    use enum_set::{EnumSet, CLike};

    use MutableSeq;

    #[deriving(PartialEq, Show)]
    #[repr(uint)]
    enum Foo {
        A, B, C
    }

    impl CLike for Foo {
        fn to_uint(&self) -> uint {
            *self as uint
        }

        fn from_uint(v: uint) -> Foo {
            unsafe { mem::transmute(v) }
        }
    }

    #[test]
    fn test_empty() {
        let e: EnumSet<Foo> = EnumSet::empty();
        assert!(e.is_empty());
    }

    #[test]
    fn test_show() {
        let mut e = EnumSet::empty();
        assert_eq!("{}", e.to_string().as_slice());
        e.add(A);
        assert_eq!("{A}", e.to_string().as_slice());
        e.add(C);
        assert_eq!("{A, C}", e.to_string().as_slice());
    }

    ///////////////////////////////////////////////////////////////////////////
    // intersect

    #[test]
    fn test_two_empties_do_not_intersect() {
        let e1: EnumSet<Foo> = EnumSet::empty();
        let e2: EnumSet<Foo> = EnumSet::empty();
        assert!(!e1.intersects(e2));
    }

    #[test]
    fn test_empty_does_not_intersect_with_full() {
        let e1: EnumSet<Foo> = EnumSet::empty();

        let mut e2: EnumSet<Foo> = EnumSet::empty();
        e2.add(A);
        e2.add(B);
        e2.add(C);

        assert!(!e1.intersects(e2));
    }

    #[test]
    fn test_disjoint_intersects() {
        let mut e1: EnumSet<Foo> = EnumSet::empty();
        e1.add(A);

        let mut e2: EnumSet<Foo> = EnumSet::empty();
        e2.add(B);

        assert!(!e1.intersects(e2));
    }

    #[test]
    fn test_overlapping_intersects() {
        let mut e1: EnumSet<Foo> = EnumSet::empty();
        e1.add(A);

        let mut e2: EnumSet<Foo> = EnumSet::empty();
        e2.add(A);
        e2.add(B);

        assert!(e1.intersects(e2));
    }

    ///////////////////////////////////////////////////////////////////////////
    // contains and contains_elem

    #[test]
    fn test_contains() {
        let mut e1: EnumSet<Foo> = EnumSet::empty();
        e1.add(A);

        let mut e2: EnumSet<Foo> = EnumSet::empty();
        e2.add(A);
        e2.add(B);

        assert!(!e1.contains(e2));
        assert!(e2.contains(e1));
    }

    #[test]
    fn test_contains_elem() {
        let mut e1: EnumSet<Foo> = EnumSet::empty();
        e1.add(A);
        assert!(e1.contains_elem(A));
        assert!(!e1.contains_elem(B));
        assert!(!e1.contains_elem(C));

        e1.add(A);
        e1.add(B);
        assert!(e1.contains_elem(A));
        assert!(e1.contains_elem(B));
        assert!(!e1.contains_elem(C));
    }

    ///////////////////////////////////////////////////////////////////////////
    // iter

    #[test]
    fn test_iterator() {
        let mut e1: EnumSet<Foo> = EnumSet::empty();

        let elems: Vec<Foo> = e1.iter().collect();
        assert!(elems.is_empty())

        e1.add(A);
        let elems = e1.iter().collect();
        assert_eq!(vec![A], elems)

        e1.add(C);
        let elems = e1.iter().collect();
        assert_eq!(vec![A,C], elems)

        e1.add(C);
        let elems = e1.iter().collect();
        assert_eq!(vec![A,C], elems)

        e1.add(B);
        let elems = e1.iter().collect();
        assert_eq!(vec![A,B,C], elems)
    }

    ///////////////////////////////////////////////////////////////////////////
    // operators

    #[test]
    fn test_operators() {
        let mut e1: EnumSet<Foo> = EnumSet::empty();
        e1.add(A);
        e1.add(C);

        let mut e2: EnumSet<Foo> = EnumSet::empty();
        e2.add(B);
        e2.add(C);

        let e_union = e1 | e2;
        let elems = e_union.iter().collect();
        assert_eq!(vec![A,B,C], elems)

        let e_intersection = e1 & e2;
        let elems = e_intersection.iter().collect();
        assert_eq!(vec![C], elems)

        let e_subtract = e1 - e2;
        let elems = e_subtract.iter().collect();
        assert_eq!(vec![A], elems)
    }
}
