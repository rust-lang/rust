// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is pretty much entirely stolen from TreeSet, since BTreeMap has an identical interface
// to TreeMap

use core::prelude::*;

use super::{BTreeMap, Keys, MoveEntries};
use std::hash::Hash;
use core::default::Default;
use core::{iter, fmt};
use core::iter::Peekable;
use core::fmt::Show;

use {Mutable, Set, MutableSet, MutableMap, Map};

/// A set based on a B-Tree.
///
/// See BTreeMap's documentation for a detailed discussion of this collection's performance
/// benefits and drawbacks.
#[deriving(Clone, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub struct BTreeSet<T>{
    map: BTreeMap<T, ()>,
}

/// An iterator over a BTreeSet's items.
pub type Items<'a, T> = Keys<'a, T, ()>;

/// An owning iterator over a BTreeSet's items.
pub type MoveItems<T> = iter::Map<'static, (T, ()), T, MoveEntries<T, ()>>;

/// A lazy iterator producing elements in the set difference (in-order).
pub struct DifferenceItems<'a, T:'a> {
    a: Peekable<&'a T, Items<'a, T>>,
    b: Peekable<&'a T, Items<'a, T>>,
}

/// A lazy iterator producing elements in the set symmetric difference (in-order).
pub struct SymDifferenceItems<'a, T:'a> {
    a: Peekable<&'a T, Items<'a, T>>,
    b: Peekable<&'a T, Items<'a, T>>,
}

/// A lazy iterator producing elements in the set intersection (in-order).
pub struct IntersectionItems<'a, T:'a> {
    a: Peekable<&'a T, Items<'a, T>>,
    b: Peekable<&'a T, Items<'a, T>>,
}

/// A lazy iterator producing elements in the set union (in-order).
pub struct UnionItems<'a, T:'a> {
    a: Peekable<&'a T, Items<'a, T>>,
    b: Peekable<&'a T, Items<'a, T>>,
}

impl<T: Ord> BTreeSet<T> {
    /// Makes a new BTreeSet with a reasonable choice of B.
    pub fn new() -> BTreeSet<T> {
        BTreeSet { map: BTreeMap::new() }
    }

    /// Makes a new BTreeSet with the given B.
    ///
    /// B cannot be less than 2.
    pub fn with_b(b: uint) -> BTreeSet<T> {
        BTreeSet { map: BTreeMap::with_b(b) }
    }
}

impl<T> BTreeSet<T> {
    /// Gets an iterator over the BTreeSet's contents.
    pub fn iter<'a>(&'a self) -> Items<'a, T> {
        self.map.keys()
    }

    /// Gets an iterator for moving out the BtreeSet's contents.
    pub fn into_iter(self) -> MoveItems<T> {
        self.map.into_iter().map(|(k, _)| k)
    }
}

impl<T: Ord> BTreeSet<T> {
    /// Visits the values representing the difference, in ascending order.
    pub fn difference<'a>(&'a self, other: &'a BTreeSet<T>) -> DifferenceItems<'a, T> {
        DifferenceItems{a: self.iter().peekable(), b: other.iter().peekable()}
    }

    /// Visits the values representing the symmetric difference, in ascending order.
    pub fn symmetric_difference<'a>(&'a self, other: &'a BTreeSet<T>)
        -> SymDifferenceItems<'a, T> {
        SymDifferenceItems{a: self.iter().peekable(), b: other.iter().peekable()}
    }

    /// Visits the values representing the intersection, in ascending order.
    pub fn intersection<'a>(&'a self, other: &'a BTreeSet<T>)
        -> IntersectionItems<'a, T> {
        IntersectionItems{a: self.iter().peekable(), b: other.iter().peekable()}
    }

    /// Visits the values representing the union, in ascending order.
    pub fn union<'a>(&'a self, other: &'a BTreeSet<T>) -> UnionItems<'a, T> {
        UnionItems{a: self.iter().peekable(), b: other.iter().peekable()}
    }
}

impl<T> Collection for BTreeSet<T> {
    fn len(&self) -> uint {
        self.map.len()
    }
}

impl<T: Ord> Mutable for BTreeSet<T> {
    fn clear(&mut self) {
        self.map.clear()
    }
}

impl<T: Ord> Set<T> for BTreeSet<T> {
    fn contains(&self, value: &T) -> bool {
        self.map.find(value).is_some()
    }

    fn is_disjoint(&self, other: &BTreeSet<T>) -> bool {
        self.intersection(other).next().is_none()
    }

    fn is_subset(&self, other: &BTreeSet<T>) -> bool {
        // Stolen from TreeMap
        let mut x = self.iter();
        let mut y = other.iter();
        let mut a = x.next();
        let mut b = y.next();
        while a.is_some() {
            if b.is_none() {
                return false;
            }

            let a1 = a.unwrap();
            let b1 = b.unwrap();

            match b1.cmp(a1) {
                Less => (),
                Greater => return false,
                Equal => a = x.next(),
            }

            b = y.next();
        }
        true
    }
}

impl<T: Ord> MutableSet<T> for BTreeSet<T>{
    fn insert(&mut self, value: T) -> bool {
        self.map.insert(value, ())
    }

    fn remove(&mut self, value: &T) -> bool {
        self.map.remove(value)
    }
}

impl<T: Ord> FromIterator<T> for BTreeSet<T> {
    fn from_iter<Iter: Iterator<T>>(iter: Iter) -> BTreeSet<T> {
        let mut set = BTreeSet::new();
        set.extend(iter);
        set
    }
}

impl<T: Ord> Extendable<T> for BTreeSet<T> {
    #[inline]
    fn extend<Iter: Iterator<T>>(&mut self, mut iter: Iter) {
        for elem in iter {
            self.insert(elem);
        }
    }
}

impl<T: Ord> Default for BTreeSet<T> {
    fn default() -> BTreeSet<T> {
        BTreeSet::new()
    }
}

impl<T: Show> Show for BTreeSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{{"));

        for (i, x) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{}", *x));
        }

        write!(f, "}}")
    }
}

/// Compare `x` and `y`, but return `short` if x is None and `long` if y is None
fn cmp_opt<T: Ord>(x: Option<&T>, y: Option<&T>,
                        short: Ordering, long: Ordering) -> Ordering {
    match (x, y) {
        (None    , _       ) => short,
        (_       , None    ) => long,
        (Some(x1), Some(y1)) => x1.cmp(y1),
    }
}

impl<'a, T: Ord> Iterator<&'a T> for DifferenceItems<'a, T> {
    fn next(&mut self) -> Option<&'a T> {
        loop {
            match cmp_opt(self.a.peek(), self.b.peek(), Less, Less) {
                Less    => return self.a.next(),
                Equal   => { self.a.next(); self.b.next(); }
                Greater => { self.b.next(); }
            }
        }
    }
}

impl<'a, T: Ord> Iterator<&'a T> for SymDifferenceItems<'a, T> {
    fn next(&mut self) -> Option<&'a T> {
        loop {
            match cmp_opt(self.a.peek(), self.b.peek(), Greater, Less) {
                Less    => return self.a.next(),
                Equal   => { self.a.next(); self.b.next(); }
                Greater => return self.b.next(),
            }
        }
    }
}

impl<'a, T: Ord> Iterator<&'a T> for IntersectionItems<'a, T> {
    fn next(&mut self) -> Option<&'a T> {
        loop {
            let o_cmp = match (self.a.peek(), self.b.peek()) {
                (None    , _       ) => None,
                (_       , None    ) => None,
                (Some(a1), Some(b1)) => Some(a1.cmp(b1)),
            };
            match o_cmp {
                None          => return None,
                Some(Less)    => { self.a.next(); }
                Some(Equal)   => { self.b.next(); return self.a.next() }
                Some(Greater) => { self.b.next(); }
            }
        }
    }
}

impl<'a, T: Ord> Iterator<&'a T> for UnionItems<'a, T> {
    fn next(&mut self) -> Option<&'a T> {
        loop {
            match cmp_opt(self.a.peek(), self.b.peek(), Greater, Less) {
                Less    => return self.a.next(),
                Equal   => { self.b.next(); return self.a.next() }
                Greater => return self.b.next(),
            }
        }
    }
}


#[cfg(test)]
mod test {
    use std::prelude::*;

    use {Set, MutableSet};
    use super::BTreeSet;
    use std::hash;

    #[test]
    fn test_clone_eq() {
      let mut m = BTreeSet::new();

      m.insert(1i);
      m.insert(2);

      assert!(m.clone() == m);
    }

    #[test]
    fn test_hash() {
      let mut x = BTreeSet::new();
      let mut y = BTreeSet::new();

      x.insert(1i);
      x.insert(2);
      x.insert(3);

      y.insert(3i);
      y.insert(2);
      y.insert(1);

      assert!(hash::hash(&x) == hash::hash(&y));
    }

    fn check(a: &[int],
             b: &[int],
             expected: &[int],
             f: |&BTreeSet<int>, &BTreeSet<int>, f: |&int| -> bool| -> bool) {
        let mut set_a = BTreeSet::new();
        let mut set_b = BTreeSet::new();

        for x in a.iter() { assert!(set_a.insert(*x)) }
        for y in b.iter() { assert!(set_b.insert(*y)) }

        let mut i = 0;
        f(&set_a, &set_b, |x| {
            assert_eq!(*x, expected[i]);
            i += 1;
            true
        });
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_intersection() {
        fn check_intersection(a: &[int], b: &[int], expected: &[int]) {
            check(a, b, expected, |x, y, f| x.intersection(y).all(f))
        }

        check_intersection([], [], []);
        check_intersection([1, 2, 3], [], []);
        check_intersection([], [1, 2, 3], []);
        check_intersection([2], [1, 2, 3], [2]);
        check_intersection([1, 2, 3], [2], [2]);
        check_intersection([11, 1, 3, 77, 103, 5, -5],
                           [2, 11, 77, -9, -42, 5, 3],
                           [3, 5, 11, 77]);
    }

    #[test]
    fn test_difference() {
        fn check_difference(a: &[int], b: &[int], expected: &[int]) {
            check(a, b, expected, |x, y, f| x.difference(y).all(f))
        }

        check_difference([], [], []);
        check_difference([1, 12], [], [1, 12]);
        check_difference([], [1, 2, 3, 9], []);
        check_difference([1, 3, 5, 9, 11],
                         [3, 9],
                         [1, 5, 11]);
        check_difference([-5, 11, 22, 33, 40, 42],
                         [-12, -5, 14, 23, 34, 38, 39, 50],
                         [11, 22, 33, 40, 42]);
    }

    #[test]
    fn test_symmetric_difference() {
        fn check_symmetric_difference(a: &[int], b: &[int],
                                      expected: &[int]) {
            check(a, b, expected, |x, y, f| x.symmetric_difference(y).all(f))
        }

        check_symmetric_difference([], [], []);
        check_symmetric_difference([1, 2, 3], [2], [1, 3]);
        check_symmetric_difference([2], [1, 2, 3], [1, 3]);
        check_symmetric_difference([1, 3, 5, 9, 11],
                                   [-2, 3, 9, 14, 22],
                                   [-2, 1, 5, 11, 14, 22]);
    }

    #[test]
    fn test_union() {
        fn check_union(a: &[int], b: &[int],
                                      expected: &[int]) {
            check(a, b, expected, |x, y, f| x.union(y).all(f))
        }

        check_union([], [], []);
        check_union([1, 2, 3], [2], [1, 2, 3]);
        check_union([2], [1, 2, 3], [1, 2, 3]);
        check_union([1, 3, 5, 9, 11, 16, 19, 24],
                    [-2, 1, 5, 9, 13, 19],
                    [-2, 1, 3, 5, 9, 11, 13, 16, 19, 24]);
    }

    #[test]
    fn test_zip() {
        let mut x = BTreeSet::new();
        x.insert(5u);
        x.insert(12u);
        x.insert(11u);

        let mut y = BTreeSet::new();
        y.insert("foo");
        y.insert("bar");

        let x = x;
        let y = y;
        let mut z = x.iter().zip(y.iter());

        // FIXME: #5801: this needs a type hint to compile...
        let result: Option<(&uint, & &'static str)> = z.next();
        assert_eq!(result.unwrap(), (&5u, &("bar")));

        let result: Option<(&uint, & &'static str)> = z.next();
        assert_eq!(result.unwrap(), (&11u, &("foo")));

        let result: Option<(&uint, & &'static str)> = z.next();
        assert!(result.is_none());
    }

    #[test]
    fn test_from_iter() {
        let xs = [1i, 2, 3, 4, 5, 6, 7, 8, 9];

        let set: BTreeSet<int> = xs.iter().map(|&x| x).collect();

        for x in xs.iter() {
            assert!(set.contains(x));
        }
    }

    #[test]
    fn test_show() {
        let mut set: BTreeSet<int> = BTreeSet::new();
        let empty: BTreeSet<int> = BTreeSet::new();

        set.insert(1);
        set.insert(2);

        let set_str = format!("{}", set);

        assert!(set_str == "{1, 2}".to_string());
        assert_eq!(format!("{}", empty), "{}".to_string());
    }
}
