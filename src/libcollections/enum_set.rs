// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A structure for holding a set of enum variants
//!
//! This module defines a container which uses an efficient bit mask
//! representation to hold C-like enum variants.

use std::num::Bitwise;

#[deriving(Clone, Eq, TotalEq, Hash, Show)]
/// A specialized Set implementation to use enum types.
pub struct EnumSet<E> {
    // We must maintain the invariant that no bits are set
    // for which no variant exists
    bits: uint
}

/// An interface for casting C-like enum to uint and back.
pub trait CLike {
    /// Converts C-like enum to uint.
    fn to_uint(&self) -> uint;
    /// Converts uint to C-like enum.
    fn from_uint(uint) -> Self;
}

fn bit<E:CLike>(e: E) -> uint {
    1 << e.to_uint()
}

impl<E:CLike> EnumSet<E> {
    /// Returns an empty EnumSet.
    pub fn empty() -> EnumSet<E> {
        EnumSet {bits: 0}
    }

    /// Returns true if an EnumSet is empty.
    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }

    /// Returns true if an EnumSet contains any enum of a given EnumSet
    pub fn intersects(&self, e: EnumSet<E>) -> bool {
        (self.bits & e.bits) != 0
    }

    /// Returns an intersection of both EnumSets.
    pub fn intersection(&self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits & e.bits}
    }

    /// Returns true if a given EnumSet is included in an EnumSet.
    pub fn contains(&self, e: EnumSet<E>) -> bool {
        (self.bits & e.bits) == e.bits
    }

    /// Returns a union of both EnumSets.
    pub fn union(&self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits | e.bits}
    }

    /// Add an enum to an EnumSet
    pub fn add(&mut self, e: E) {
        self.bits |= bit(e);
    }

    /// Returns true if an EnumSet contains a given enum
    pub fn contains_elem(&self, e: E) -> bool {
        (self.bits & bit(e)) != 0
    }

    /// Returns an iterator over an EnumSet
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

    use std::cast;

    use enum_set::{EnumSet, CLike};

    #[deriving(Eq, Show)]
    #[repr(uint)]
    enum Foo {
        A, B, C
    }

    impl CLike for Foo {
        fn to_uint(&self) -> uint {
            *self as uint
        }

        fn from_uint(v: uint) -> Foo {
            unsafe { cast::transmute(v) }
        }
    }

    #[test]
    fn test_empty() {
        let e: EnumSet<Foo> = EnumSet::empty();
        assert!(e.is_empty());
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

        let elems: ~[Foo] = e1.iter().collect();
        assert_eq!(~[], elems)

        e1.add(A);
        let elems: ~[Foo] = e1.iter().collect();
        assert_eq!(~[A], elems)

        e1.add(C);
        let elems: ~[Foo] = e1.iter().collect();
        assert_eq!(~[A,C], elems)

        e1.add(C);
        let elems: ~[Foo] = e1.iter().collect();
        assert_eq!(~[A,C], elems)

        e1.add(B);
        let elems: ~[Foo] = e1.iter().collect();
        assert_eq!(~[A,B,C], elems)
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
        let elems: ~[Foo] = e_union.iter().collect();
        assert_eq!(~[A,B,C], elems)

        let e_intersection = e1 & e2;
        let elems: ~[Foo] = e_intersection.iter().collect();
        assert_eq!(~[C], elems)

        let e_subtract = e1 - e2;
        let elems: ~[Foo] = e_subtract.iter().collect();
        assert_eq!(~[A], elems)
    }
}
