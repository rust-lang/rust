// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME(conventions): implement bounded iterators
// FIXME(conventions): replace each_reverse by making iter DoubleEnded
// FIXME(conventions): implement iter_mut and into_iter

use core::prelude::*;

use core::default::Default;
use core::fmt;
use core::fmt::Show;
use core::iter::Peekable;
use std::hash::Hash;

use trie_map::{TrieMap, Entries};

/// A set implemented as a radix trie.
///
/// # Examples
///
/// ```
/// use std::collections::TrieSet;
///
/// let mut set = TrieSet::new();
/// set.insert(6);
/// set.insert(28);
/// set.insert(6);
///
/// assert_eq!(set.len(), 2);
///
/// if !set.contains(&3) {
///     println!("3 is not in the set");
/// }
///
/// // Print contents in order
/// for x in set.iter() {
///     println!("{}", x);
/// }
///
/// set.remove(&6);
/// assert_eq!(set.len(), 1);
///
/// set.clear();
/// assert!(set.is_empty());
/// ```
#[deriving(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct TrieSet {
    map: TrieMap<()>
}

impl Show for TrieSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{{"));

        for (i, x) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{}", x));
        }

        write!(f, "}}")
    }
}

impl Default for TrieSet {
    #[inline]
    fn default() -> TrieSet { TrieSet::new() }
}

impl TrieSet {
    /// Creates an empty TrieSet.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    /// let mut set = TrieSet::new();
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn new() -> TrieSet {
        TrieSet{map: TrieMap::new()}
    }

    /// Visits all values in reverse order. Aborts traversal when `f` returns `false`.
    /// Returns `true` if `f` returns `true` for all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let set: TrieSet = [1, 2, 3, 4, 5].iter().map(|&x| x).collect();
    ///
    /// let mut vec = Vec::new();
    /// assert_eq!(true, set.each_reverse(|&x| { vec.push(x); true }));
    /// assert_eq!(vec, vec![5, 4, 3, 2, 1]);
    ///
    /// // Stop when we reach 3
    /// let mut vec = Vec::new();
    /// assert_eq!(false, set.each_reverse(|&x| { vec.push(x); x != 3 }));
    /// assert_eq!(vec, vec![5, 4, 3]);
    /// ```
    #[inline]
    pub fn each_reverse(&self, f: |&uint| -> bool) -> bool {
        self.map.each_reverse(|k, _| f(k))
    }

    /// Gets an iterator over the values in the set, in sorted order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let mut set = TrieSet::new();
    /// set.insert(3);
    /// set.insert(2);
    /// set.insert(1);
    /// set.insert(2);
    ///
    /// // Print 1, 2, 3
    /// for x in set.iter() {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn iter<'a>(&'a self) -> SetItems<'a> {
        SetItems{iter: self.map.iter()}
    }

    /// Gets an iterator pointing to the first value that is not less than `val`.
    /// If all values in the set are less than `val` an empty iterator is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let set: TrieSet = [2, 4, 6, 8].iter().map(|&x| x).collect();
    /// assert_eq!(set.lower_bound(4).next(), Some(4));
    /// assert_eq!(set.lower_bound(5).next(), Some(6));
    /// assert_eq!(set.lower_bound(10).next(), None);
    /// ```
    pub fn lower_bound<'a>(&'a self, val: uint) -> SetItems<'a> {
        SetItems{iter: self.map.lower_bound(val)}
    }

    /// Gets an iterator pointing to the first value that key is greater than `val`.
    /// If all values in the set are less than or equal to `val` an empty iterator is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let set: TrieSet = [2, 4, 6, 8].iter().map(|&x| x).collect();
    /// assert_eq!(set.upper_bound(4).next(), Some(6));
    /// assert_eq!(set.upper_bound(5).next(), Some(6));
    /// assert_eq!(set.upper_bound(10).next(), None);
    /// ```
    pub fn upper_bound<'a>(&'a self, val: uint) -> SetItems<'a> {
        SetItems{iter: self.map.upper_bound(val)}
    }

    /// Visits the values representing the difference, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let a: TrieSet = [1, 2, 3].iter().map(|&x| x).collect();
    /// let b: TrieSet = [3, 4, 5].iter().map(|&x| x).collect();
    ///
    /// // Can be seen as `a - b`.
    /// for x in a.difference(&b) {
    ///     println!("{}", x); // Print 1 then 2
    /// }
    ///
    /// let diff1: TrieSet = a.difference(&b).collect();
    /// assert_eq!(diff1, [1, 2].iter().map(|&x| x).collect());
    ///
    /// // Note that difference is not symmetric,
    /// // and `b - a` means something else:
    /// let diff2: TrieSet = b.difference(&a).collect();
    /// assert_eq!(diff2, [4, 5].iter().map(|&x| x).collect());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn difference<'a>(&'a self, other: &'a TrieSet) -> DifferenceItems<'a> {
        DifferenceItems{a: self.iter().peekable(), b: other.iter().peekable()}
    }

    /// Visits the values representing the symmetric difference, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let a: TrieSet = [1, 2, 3].iter().map(|&x| x).collect();
    /// let b: TrieSet = [3, 4, 5].iter().map(|&x| x).collect();
    ///
    /// // Print 1, 2, 4, 5 in ascending order.
    /// for x in a.symmetric_difference(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let diff1: TrieSet = a.symmetric_difference(&b).collect();
    /// let diff2: TrieSet = b.symmetric_difference(&a).collect();
    ///
    /// assert_eq!(diff1, diff2);
    /// assert_eq!(diff1, [1, 2, 4, 5].iter().map(|&x| x).collect());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle."]
    pub fn symmetric_difference<'a>(&'a self, other: &'a TrieSet) -> SymDifferenceItems<'a> {
        SymDifferenceItems{a: self.iter().peekable(), b: other.iter().peekable()}
    }

    /// Visits the values representing the intersection, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let a: TrieSet = [1, 2, 3].iter().map(|&x| x).collect();
    /// let b: TrieSet = [2, 3, 4].iter().map(|&x| x).collect();
    ///
    /// // Print 2, 3 in ascending order.
    /// for x in a.intersection(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let diff: TrieSet = a.intersection(&b).collect();
    /// assert_eq!(diff, [2, 3].iter().map(|&x| x).collect());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn intersection<'a>(&'a self, other: &'a TrieSet) -> IntersectionItems<'a> {
        IntersectionItems{a: self.iter().peekable(), b: other.iter().peekable()}
    }

    /// Visits the values representing the union, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let a: TrieSet = [1, 2, 3].iter().map(|&x| x).collect();
    /// let b: TrieSet = [3, 4, 5].iter().map(|&x| x).collect();
    ///
    /// // Print 1, 2, 3, 4, 5 in ascending order.
    /// for x in a.union(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let diff: TrieSet = a.union(&b).collect();
    /// assert_eq!(diff, [1, 2, 3, 4, 5].iter().map(|&x| x).collect());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn union<'a>(&'a self, other: &'a TrieSet) -> UnionItems<'a> {
        UnionItems{a: self.iter().peekable(), b: other.iter().peekable()}
    }

    /// Return the number of elements in the set
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let mut v = TrieSet::new();
    /// assert_eq!(v.len(), 0);
    /// v.insert(1);
    /// assert_eq!(v.len(), 1);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn len(&self) -> uint { self.map.len() }

    /// Returns true if the set contains no elements
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let mut v = TrieSet::new();
    /// assert!(v.is_empty());
    /// v.insert(1);
    /// assert!(!v.is_empty());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Clears the set, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let mut v = TrieSet::new();
    /// v.insert(1);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn clear(&mut self) { self.map.clear() }

    /// Returns `true` if the set contains a value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let set: TrieSet = [1, 2, 3].iter().map(|&x| x).collect();
    /// assert_eq!(set.contains(&1), true);
    /// assert_eq!(set.contains(&4), false);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn contains(&self, value: &uint) -> bool {
        self.map.contains_key(value)
    }

    /// Returns `true` if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let a: TrieSet = [1, 2, 3].iter().map(|&x| x).collect();
    /// let mut b: TrieSet = TrieSet::new();
    ///
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(4);
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(1);
    /// assert_eq!(a.is_disjoint(&b), false);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn is_disjoint(&self, other: &TrieSet) -> bool {
        self.iter().all(|v| !other.contains(&v))
    }

    /// Returns `true` if the set is a subset of another.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let sup: TrieSet = [1, 2, 3].iter().map(|&x| x).collect();
    /// let mut set: TrieSet = TrieSet::new();
    ///
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(2);
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(4);
    /// assert_eq!(set.is_subset(&sup), false);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn is_subset(&self, other: &TrieSet) -> bool {
        self.iter().all(|v| other.contains(&v))
    }

    /// Returns `true` if the set is a superset of another.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let sub: TrieSet = [1, 2].iter().map(|&x| x).collect();
    /// let mut set: TrieSet = TrieSet::new();
    ///
    /// assert_eq!(set.is_superset(&sub), false);
    ///
    /// set.insert(0);
    /// set.insert(1);
    /// assert_eq!(set.is_superset(&sub), false);
    ///
    /// set.insert(2);
    /// assert_eq!(set.is_superset(&sub), true);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn is_superset(&self, other: &TrieSet) -> bool {
        other.is_subset(self)
    }

    /// Adds a value to the set. Returns `true` if the value was not already
    /// present in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let mut set = TrieSet::new();
    ///
    /// assert_eq!(set.insert(2), true);
    /// assert_eq!(set.insert(2), false);
    /// assert_eq!(set.len(), 1);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn insert(&mut self, value: uint) -> bool {
        self.map.insert(value, ()).is_none()
    }

    /// Removes a value from the set. Returns `true` if the value was
    /// present in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let mut set = TrieSet::new();
    ///
    /// set.insert(2);
    /// assert_eq!(set.remove(&2), true);
    /// assert_eq!(set.remove(&2), false);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn remove(&mut self, value: &uint) -> bool {
        self.map.remove(value).is_some()
    }
}

impl FromIterator<uint> for TrieSet {
    fn from_iter<Iter: Iterator<uint>>(iter: Iter) -> TrieSet {
        let mut set = TrieSet::new();
        set.extend(iter);
        set
    }
}

impl Extend<uint> for TrieSet {
    fn extend<Iter: Iterator<uint>>(&mut self, mut iter: Iter) {
        for elem in iter {
            self.insert(elem);
        }
    }
}

#[unstable = "matches collection reform specification, waiting for dust to settle"]
impl BitOr<TrieSet, TrieSet> for TrieSet {
    /// Returns the union of `self` and `rhs` as a new `TrieSet`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let a: TrieSet = vec![1, 2, 3].into_iter().collect();
    /// let b: TrieSet = vec![3, 4, 5].into_iter().collect();
    ///
    /// let set: TrieSet = a | b;
    /// let v: Vec<uint> = set.iter().collect();
    /// assert_eq!(v, vec![1u, 2, 3, 4, 5]);
    /// ```
    fn bitor(&self, rhs: &TrieSet) -> TrieSet {
        self.union(rhs).collect()
    }
}

#[unstable = "matches collection reform specification, waiting for dust to settle"]
impl BitAnd<TrieSet, TrieSet> for TrieSet {
    /// Returns the intersection of `self` and `rhs` as a new `TrieSet`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let a: TrieSet = vec![1, 2, 3].into_iter().collect();
    /// let b: TrieSet = vec![2, 3, 4].into_iter().collect();
    ///
    /// let set: TrieSet = a & b;
    /// let v: Vec<uint> = set.iter().collect();
    /// assert_eq!(v, vec![2u, 3]);
    /// ```
    fn bitand(&self, rhs: &TrieSet) -> TrieSet {
        self.intersection(rhs).collect()
    }
}

#[unstable = "matches collection reform specification, waiting for dust to settle"]
impl BitXor<TrieSet, TrieSet> for TrieSet {
    /// Returns the symmetric difference of `self` and `rhs` as a new `TrieSet`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let a: TrieSet = vec![1, 2, 3].into_iter().collect();
    /// let b: TrieSet = vec![3, 4, 5].into_iter().collect();
    ///
    /// let set: TrieSet = a ^ b;
    /// let v: Vec<uint> = set.iter().collect();
    /// assert_eq!(v, vec![1u, 2, 4, 5]);
    /// ```
    fn bitxor(&self, rhs: &TrieSet) -> TrieSet {
        self.symmetric_difference(rhs).collect()
    }
}

#[unstable = "matches collection reform specification, waiting for dust to settle"]
impl Sub<TrieSet, TrieSet> for TrieSet {
    /// Returns the difference of `self` and `rhs` as a new `TrieSet`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieSet;
    ///
    /// let a: TrieSet = vec![1, 2, 3].into_iter().collect();
    /// let b: TrieSet = vec![3, 4, 5].into_iter().collect();
    ///
    /// let set: TrieSet = a - b;
    /// let v: Vec<uint> = set.iter().collect();
    /// assert_eq!(v, vec![1u, 2]);
    /// ```
    fn sub(&self, rhs: &TrieSet) -> TrieSet {
        self.difference(rhs).collect()
    }
}

/// A forward iterator over a set.
pub struct SetItems<'a> {
    iter: Entries<'a, ()>
}

/// An iterator producing elements in the set difference (in-order).
pub struct DifferenceItems<'a> {
    a: Peekable<uint, SetItems<'a>>,
    b: Peekable<uint, SetItems<'a>>,
}

/// An iterator producing elements in the set symmetric difference (in-order).
pub struct SymDifferenceItems<'a> {
    a: Peekable<uint, SetItems<'a>>,
    b: Peekable<uint, SetItems<'a>>,
}

/// An iterator producing elements in the set intersection (in-order).
pub struct IntersectionItems<'a> {
    a: Peekable<uint, SetItems<'a>>,
    b: Peekable<uint, SetItems<'a>>,
}

/// An iterator producing elements in the set union (in-order).
pub struct UnionItems<'a> {
    a: Peekable<uint, SetItems<'a>>,
    b: Peekable<uint, SetItems<'a>>,
}

/// Compare `x` and `y`, but return `short` if x is None and `long` if y is None
fn cmp_opt(x: Option<&uint>, y: Option<&uint>, short: Ordering, long: Ordering) -> Ordering {
    match (x, y) {
        (None    , _       ) => short,
        (_       , None    ) => long,
        (Some(x1), Some(y1)) => x1.cmp(y1),
    }
}

impl<'a> Iterator<uint> for SetItems<'a> {
    fn next(&mut self) -> Option<uint> {
        self.iter.next().map(|(key, _)| key)
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

impl<'a> Iterator<uint> for DifferenceItems<'a> {
    fn next(&mut self) -> Option<uint> {
        loop {
            match cmp_opt(self.a.peek(), self.b.peek(), Less, Less) {
                Less    => return self.a.next(),
                Equal   => { self.a.next(); self.b.next(); }
                Greater => { self.b.next(); }
            }
        }
    }
}

impl<'a> Iterator<uint> for SymDifferenceItems<'a> {
    fn next(&mut self) -> Option<uint> {
        loop {
            match cmp_opt(self.a.peek(), self.b.peek(), Greater, Less) {
                Less => return self.a.next(),
                Equal => { self.a.next(); self.b.next(); }
                Greater => return self.b.next(),
            }
        }
    }
}

impl<'a> Iterator<uint> for IntersectionItems<'a> {
    fn next(&mut self) -> Option<uint> {
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

impl<'a> Iterator<uint> for UnionItems<'a> {
    fn next(&mut self) -> Option<uint> {
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
    use std::uint;
    use vec::Vec;

    use super::TrieSet;

    #[test]
    fn test_sane_chunk() {
        let x = 1;
        let y = 1 << (uint::BITS - 1);

        let mut trie = TrieSet::new();

        assert!(trie.insert(x));
        assert!(trie.insert(y));

        assert_eq!(trie.len(), 2);

        let expected = [x, y];

        for (i, x) in trie.iter().enumerate() {
            assert_eq!(expected[i], x);
        }
    }

    #[test]
    fn test_from_iter() {
        let xs = vec![9u, 8, 7, 6, 5, 4, 3, 2, 1];

        let set: TrieSet = xs.iter().map(|&x| x).collect();

        for x in xs.iter() {
            assert!(set.contains(x));
        }
    }

    #[test]
    fn test_show() {
        let mut set = TrieSet::new();
        let empty = TrieSet::new();

        set.insert(1);
        set.insert(2);

        let set_str = format!("{}", set);

        assert!(set_str == "{1, 2}");
        assert_eq!(format!("{}", empty), "{}");
    }

    #[test]
    fn test_clone() {
        let mut a = TrieSet::new();

        a.insert(1);
        a.insert(2);
        a.insert(3);

        assert!(a.clone() == a);
    }

    #[test]
    fn test_lt() {
        let mut a = TrieSet::new();
        let mut b = TrieSet::new();

        assert!(!(a < b) && !(b < a));
        assert!(b.insert(2u));
        assert!(a < b);
        assert!(a.insert(3u));
        assert!(!(a < b) && b < a);
        assert!(b.insert(1));
        assert!(b < a);
        assert!(a.insert(0));
        assert!(a < b);
        assert!(a.insert(6));
        assert!(a < b && !(b < a));
    }

    #[test]
    fn test_ord() {
        let mut a = TrieSet::new();
        let mut b = TrieSet::new();

        assert!(a <= b && a >= b);
        assert!(a.insert(1u));
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        assert!(b.insert(2u));
        assert!(b > a && b >= a);
        assert!(a < b && a <= b);
    }

    fn check(a: &[uint],
             b: &[uint],
             expected: &[uint],
             f: |&TrieSet, &TrieSet, f: |uint| -> bool| -> bool) {
        let mut set_a = TrieSet::new();
        let mut set_b = TrieSet::new();

        for x in a.iter() { assert!(set_a.insert(*x)) }
        for y in b.iter() { assert!(set_b.insert(*y)) }

        let mut i = 0;
        f(&set_a, &set_b, |x| {
            assert_eq!(x, expected[i]);
            i += 1;
            true
        });
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_intersection() {
        fn check_intersection(a: &[uint], b: &[uint], expected: &[uint]) {
            check(a, b, expected, |x, y, f| x.intersection(y).all(f))
        }

        check_intersection(&[], &[], &[]);
        check_intersection(&[1, 2, 3], &[], &[]);
        check_intersection(&[], &[1, 2, 3], &[]);
        check_intersection(&[2], &[1, 2, 3], &[2]);
        check_intersection(&[1, 2, 3], &[2], &[2]);
        check_intersection(&[11, 1, 3, 77, 103, 5],
                           &[2, 11, 77, 5, 3],
                           &[3, 5, 11, 77]);
    }

    #[test]
    fn test_difference() {
        fn check_difference(a: &[uint], b: &[uint], expected: &[uint]) {
            check(a, b, expected, |x, y, f| x.difference(y).all(f))
        }

        check_difference(&[], &[], &[]);
        check_difference(&[1, 12], &[], &[1, 12]);
        check_difference(&[], &[1, 2, 3, 9], &[]);
        check_difference(&[1, 3, 5, 9, 11],
                         &[3, 9],
                         &[1, 5, 11]);
        check_difference(&[11, 22, 33, 40, 42],
                         &[14, 23, 34, 38, 39, 50],
                         &[11, 22, 33, 40, 42]);
    }

    #[test]
    fn test_symmetric_difference() {
        fn check_symmetric_difference(a: &[uint], b: &[uint], expected: &[uint]) {
            check(a, b, expected, |x, y, f| x.symmetric_difference(y).all(f))
        }

        check_symmetric_difference(&[], &[], &[]);
        check_symmetric_difference(&[1, 2, 3], &[2], &[1, 3]);
        check_symmetric_difference(&[2], &[1, 2, 3], &[1, 3]);
        check_symmetric_difference(&[1, 3, 5, 9, 11],
                                   &[3, 9, 14, 22],
                                   &[1, 5, 11, 14, 22]);
    }

    #[test]
    fn test_union() {
        fn check_union(a: &[uint], b: &[uint], expected: &[uint]) {
            check(a, b, expected, |x, y, f| x.union(y).all(f))
        }

        check_union(&[], &[], &[]);
        check_union(&[1, 2, 3], &[2], &[1, 2, 3]);
        check_union(&[2], &[1, 2, 3], &[1, 2, 3]);
        check_union(&[1, 3, 5, 9, 11, 16, 19, 24],
                    &[1, 5, 9, 13, 19],
                    &[1, 3, 5, 9, 11, 13, 16, 19, 24]);
    }

    #[test]
    fn test_bit_or() {
        let a: TrieSet = vec![1, 2, 3].into_iter().collect();
        let b: TrieSet = vec![3, 4, 5].into_iter().collect();

        let set: TrieSet = a | b;
        let v: Vec<uint> = set.iter().collect();
        assert_eq!(v, vec![1u, 2, 3, 4, 5]);
    }

    #[test]
    fn test_bit_and() {
        let a: TrieSet = vec![1, 2, 3].into_iter().collect();
        let b: TrieSet = vec![2, 3, 4].into_iter().collect();

        let set: TrieSet = a & b;
        let v: Vec<uint> = set.iter().collect();
        assert_eq!(v, vec![2u, 3]);
    }

    #[test]
    fn test_bit_xor() {
        let a: TrieSet = vec![1, 2, 3].into_iter().collect();
        let b: TrieSet = vec![3, 4, 5].into_iter().collect();

        let set: TrieSet = a ^ b;
        let v: Vec<uint> = set.iter().collect();
        assert_eq!(v, vec![1u, 2, 4, 5]);
    }

    #[test]
    fn test_sub() {
        let a: TrieSet = vec![1, 2, 3].into_iter().collect();
        let b: TrieSet = vec![3, 4, 5].into_iter().collect();

        let set: TrieSet = a - b;
        let v: Vec<uint> = set.iter().collect();
        assert_eq!(v, vec![1u, 2]);
    }
}
