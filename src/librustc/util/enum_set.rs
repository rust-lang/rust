// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use core;

#[deriving(Eq, IterBytes)]
pub struct EnumSet<E> {
    // We must maintain the invariant that no bits are set
    // for which no variant exists
    priv bits: uint
}

pub trait CLike {
    pub fn to_uint(&self) -> uint;
    pub fn from_uint(uint) -> Self;
}

fn bit<E:CLike>(e: E) -> uint {
    1 << e.to_uint()
}

pub impl<E:CLike> EnumSet<E> {
    fn empty() -> EnumSet<E> {
        EnumSet {bits: 0}
    }

    fn is_empty(&self) -> bool {
        self.bits == 0
    }

    fn intersects(&self, e: EnumSet<E>) -> bool {
        (self.bits & e.bits) != 0
    }

    fn intersection(&self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits & e.bits}
    }

    fn contains(&self, e: EnumSet<E>) -> bool {
        (self.bits & e.bits) == e.bits
    }

    fn union(&self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {bits: self.bits | e.bits}
    }

    fn add(&mut self, e: E) {
        self.bits |= bit(e);
    }

    fn contains_elem(&self, e: E) -> bool {
        (self.bits & bit(e)) != 0
    }

    fn each(&self, f: &fn(E) -> bool) -> bool {
        let mut bits = self.bits;
        let mut index = 0;
        while bits != 0 {
            if (bits & 1) != 0 {
                let e = CLike::from_uint(index);
                if !f(e) {
                    return false;
                }
            }
            index += 1;
            bits >>= 1;
        }
        return true;
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

#[cfg(test)]
mod test {
    use core::iter;
    use util::enum_set::*;

    #[deriving(Eq)]
    enum Foo {
        A, B, C
    }

    impl CLike for Foo {
        pub fn to_uint(&self) -> uint {
            *self as uint
        }

        pub fn from_uint(v: uint) -> Foo {
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
    // each

    #[test]
    fn test_each() {
        let mut e1: EnumSet<Foo> = EnumSet::empty();

        assert_eq!(~[], iter::to_vec(|f| e1.each(f)))

        e1.add(A);
        assert_eq!(~[A], iter::to_vec(|f| e1.each(f)))

        e1.add(C);
        assert_eq!(~[A,C], iter::to_vec(|f| e1.each(f)))

        e1.add(C);
        assert_eq!(~[A,C], iter::to_vec(|f| e1.each(f)))

        e1.add(B);
        assert_eq!(~[A,B,C], iter::to_vec(|f| e1.each(f)))
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
        assert_eq!(~[A,B,C], iter::to_vec(|f| e_union.each(f)))

        let e_intersection = e1 & e2;
        assert_eq!(~[C], iter::to_vec(|f| e_intersection.each(f)))

        let e_subtract = e1 - e2;
        assert_eq!(~[A], iter::to_vec(|f| e_subtract.each(f)))
    }
}
