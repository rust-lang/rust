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

use core::cmp::Ordering::{self, Less, Greater, Equal};
use core::default::Default;
use core::fmt::Debug;
use core::fmt;
use core::iter::{Peekable, Map, FromIterator, IntoIterator};
use core::ops::{BitOr, BitAnd, BitXor, Sub};

use borrow::Borrow;
use btree_map::{BTreeMap, Keys};
use Bound;

// FIXME(conventions): implement bounded iterators

/// A set based on a B-Tree.
///
/// See BTreeMap's documentation for a detailed discussion of this collection's performance
/// benefits and drawbacks.
#[derive(Clone, Hash, PartialEq, Eq, Ord, PartialOrd)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct BTreeSet<T>{
    map: BTreeMap<T, ()>,
}

/// An iterator over a BTreeSet's items.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    iter: Keys<'a, T, ()>
}

/// An owning iterator over a BTreeSet's items.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<T> {
    iter: Map<::btree_map::IntoIter<T, ()>, fn((T, ())) -> T>
}

/// An iterator over a sub-range of BTreeSet's items.
pub struct Range<'a, T: 'a> {
    iter: Map<::btree_map::Range<'a, T, ()>, fn((&'a T, &'a ())) -> &'a T>
}

/// A lazy iterator producing elements in the set difference (in-order).
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Difference<'a, T:'a> {
    a: Peekable<Iter<'a, T>>,
    b: Peekable<Iter<'a, T>>,
}

/// A lazy iterator producing elements in the set symmetric difference (in-order).
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SymmetricDifference<'a, T:'a> {
    a: Peekable<Iter<'a, T>>,
    b: Peekable<Iter<'a, T>>,
}

/// A lazy iterator producing elements in the set intersection (in-order).
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Intersection<'a, T:'a> {
    a: Peekable<Iter<'a, T>>,
    b: Peekable<Iter<'a, T>>,
}

/// A lazy iterator producing elements in the set union (in-order).
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Union<'a, T:'a> {
    a: Peekable<Iter<'a, T>>,
    b: Peekable<Iter<'a, T>>,
}

impl<T: Ord> BTreeSet<T> {
    /// Makes a new BTreeSet with a reasonable choice of B.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut set: BTreeSet<i32> = BTreeSet::new();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> BTreeSet<T> {
        BTreeSet { map: BTreeMap::new() }
    }

    /// Makes a new BTreeSet with the given B.
    ///
    /// B cannot be less than 2.
    #[unstable(feature = "collections",
               reason = "probably want this to be on the type, eventually")]
    pub fn with_b(b: usize) -> BTreeSet<T> {
        BTreeSet { map: BTreeMap::with_b(b) }
    }
}

impl<T> BTreeSet<T> {
    /// Gets an iterator over the BTreeSet's contents.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let set: BTreeSet<usize> = [1, 2, 3, 4].iter().cloned().collect();
    ///
    /// for x in set.iter() {
    ///     println!("{}", x);
    /// }
    ///
    /// let v: Vec<usize> = set.iter().cloned().collect();
    /// assert_eq!(v, vec![1,2,3,4]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter<T> {
        Iter { iter: self.map.keys() }
    }

    /// Gets an iterator for moving out the BtreeSet's contents.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let set: BTreeSet<usize> = [1, 2, 3, 4].iter().cloned().collect();
    ///
    /// let v: Vec<usize> = set.into_iter().collect();
    /// assert_eq!(v, vec![1,2,3,4]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_iter(self) -> IntoIter<T> {
        fn first<A, B>((a, _): (A, B)) -> A { a }
        let first: fn((T, ())) -> T = first; // coerce to fn pointer

        IntoIter { iter: self.map.into_iter().map(first) }
    }
}

impl<T: Ord> BTreeSet<T> {
    /// Constructs a double-ended iterator over a sub-range of elements in the set, starting
    /// at min, and ending at max. If min is `Unbounded`, then it will be treated as "negative
    /// infinity", and if max is `Unbounded`, then it will be treated as "positive infinity".
    /// Thus range(Unbounded, Unbounded) will yield the whole collection.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    /// use std::collections::Bound::{Included, Unbounded};
    ///
    /// let mut set = BTreeSet::new();
    /// set.insert(3);
    /// set.insert(5);
    /// set.insert(8);
    /// for &elem in set.range(Included(&4), Included(&8)) {
    ///     println!("{}", elem);
    /// }
    /// assert_eq!(Some(&5), set.range(Included(&4), Unbounded).next());
    /// ```
    #[unstable(feature = "collections",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn range<'a>(&'a self, min: Bound<&T>, max: Bound<&T>) -> Range<'a, T> {
        fn first<A, B>((a, _): (A, B)) -> A { a }
        let first: fn((&'a T, &'a ())) -> &'a T = first; // coerce to fn pointer

        Range { iter: self.map.range(min, max).map(first) }
    }
}

impl<T: Ord> BTreeSet<T> {
    /// Visits the values representing the difference, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut a = BTreeSet::new();
    /// a.insert(1);
    /// a.insert(2);
    ///
    /// let mut b = BTreeSet::new();
    /// b.insert(2);
    /// b.insert(3);
    ///
    /// let diff: Vec<usize> = a.difference(&b).cloned().collect();
    /// assert_eq!(diff, vec![1]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn difference<'a>(&'a self, other: &'a BTreeSet<T>) -> Difference<'a, T> {
        Difference{a: self.iter().peekable(), b: other.iter().peekable()}
    }

    /// Visits the values representing the symmetric difference, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut a = BTreeSet::new();
    /// a.insert(1);
    /// a.insert(2);
    ///
    /// let mut b = BTreeSet::new();
    /// b.insert(2);
    /// b.insert(3);
    ///
    /// let sym_diff: Vec<usize> = a.symmetric_difference(&b).cloned().collect();
    /// assert_eq!(sym_diff, vec![1,3]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn symmetric_difference<'a>(&'a self, other: &'a BTreeSet<T>)
        -> SymmetricDifference<'a, T> {
        SymmetricDifference{a: self.iter().peekable(), b: other.iter().peekable()}
    }

    /// Visits the values representing the intersection, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut a = BTreeSet::new();
    /// a.insert(1);
    /// a.insert(2);
    ///
    /// let mut b = BTreeSet::new();
    /// b.insert(2);
    /// b.insert(3);
    ///
    /// let intersection: Vec<usize> = a.intersection(&b).cloned().collect();
    /// assert_eq!(intersection, vec![2]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn intersection<'a>(&'a self, other: &'a BTreeSet<T>)
        -> Intersection<'a, T> {
        Intersection{a: self.iter().peekable(), b: other.iter().peekable()}
    }

    /// Visits the values representing the union, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut a = BTreeSet::new();
    /// a.insert(1);
    ///
    /// let mut b = BTreeSet::new();
    /// b.insert(2);
    ///
    /// let union: Vec<usize> = a.union(&b).cloned().collect();
    /// assert_eq!(union, vec![1,2]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn union<'a>(&'a self, other: &'a BTreeSet<T>) -> Union<'a, T> {
        Union{a: self.iter().peekable(), b: other.iter().peekable()}
    }

    /// Return the number of elements in the set
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut v = BTreeSet::new();
    /// assert_eq!(v.len(), 0);
    /// v.insert(1);
    /// assert_eq!(v.len(), 1);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize { self.map.len() }

    /// Returns true if the set contains no elements
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut v = BTreeSet::new();
    /// assert!(v.is_empty());
    /// v.insert(1);
    /// assert!(!v.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Clears the set, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut v = BTreeSet::new();
    /// v.insert(1);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn clear(&mut self) {
        self.map.clear()
    }

    /// Returns `true` if the set contains a value.
    ///
    /// The value may be any borrowed form of the set's value type,
    /// but the ordering on the borrowed form *must* match the
    /// ordering on the value type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let set: BTreeSet<_> = [1, 2, 3].iter().cloned().collect();
    /// assert_eq!(set.contains(&1), true);
    /// assert_eq!(set.contains(&4), false);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn contains<Q: ?Sized>(&self, value: &Q) -> bool where T: Borrow<Q>, Q: Ord {
        self.map.contains_key(value)
    }

    /// Returns `true` if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let a: BTreeSet<_> = [1, 2, 3].iter().cloned().collect();
    /// let mut b = BTreeSet::new();
    ///
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(4);
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(1);
    /// assert_eq!(a.is_disjoint(&b), false);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_disjoint(&self, other: &BTreeSet<T>) -> bool {
        self.intersection(other).next().is_none()
    }

    /// Returns `true` if the set is a subset of another.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let sup: BTreeSet<_> = [1, 2, 3].iter().cloned().collect();
    /// let mut set = BTreeSet::new();
    ///
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(2);
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(4);
    /// assert_eq!(set.is_subset(&sup), false);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_subset(&self, other: &BTreeSet<T>) -> bool {
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

    /// Returns `true` if the set is a superset of another.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let sub: BTreeSet<_> = [1, 2].iter().cloned().collect();
    /// let mut set = BTreeSet::new();
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_superset(&self, other: &BTreeSet<T>) -> bool {
        other.is_subset(self)
    }

    /// Adds a value to the set. Returns `true` if the value was not already
    /// present in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    ///
    /// assert_eq!(set.insert(2), true);
    /// assert_eq!(set.insert(2), false);
    /// assert_eq!(set.len(), 1);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(&mut self, value: T) -> bool {
        self.map.insert(value, ()).is_none()
    }

    /// Removes a value from the set. Returns `true` if the value was
    /// present in the set.
    ///
    /// The value may be any borrowed form of the set's value type,
    /// but the ordering on the borrowed form *must* match the
    /// ordering on the value type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    ///
    /// set.insert(2);
    /// assert_eq!(set.remove(&2), true);
    /// assert_eq!(set.remove(&2), false);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove<Q: ?Sized>(&mut self, value: &Q) -> bool where T: Borrow<Q>, Q: Ord {
        self.map.remove(value).is_some()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> FromIterator<T> for BTreeSet<T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> BTreeSet<T> {
        let mut set = BTreeSet::new();
        set.extend(iter);
        set
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> IntoIterator for BTreeSet<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        self.into_iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a BTreeSet<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> Extend<T> for BTreeSet<T> {
    #[inline]
    fn extend<Iter: IntoIterator<Item=T>>(&mut self, iter: Iter) {
        for elem in iter {
            self.insert(elem);
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> Default for BTreeSet<T> {
    #[stable(feature = "rust1", since = "1.0.0")]
    fn default() -> BTreeSet<T> {
        BTreeSet::new()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, 'b, T: Ord + Clone> Sub<&'b BTreeSet<T>> for &'a BTreeSet<T> {
    type Output = BTreeSet<T>;

    /// Returns the difference of `self` and `rhs` as a new `BTreeSet<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let a: BTreeSet<_> = vec![1, 2, 3].into_iter().collect();
    /// let b: BTreeSet<_> = vec![3, 4, 5].into_iter().collect();
    ///
    /// let result = &a - &b;
    /// let result_vec: Vec<_> = result.into_iter().collect();
    /// assert_eq!(result_vec, vec![1, 2]);
    /// ```
    fn sub(self, rhs: &BTreeSet<T>) -> BTreeSet<T> {
        self.difference(rhs).cloned().collect()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, 'b, T: Ord + Clone> BitXor<&'b BTreeSet<T>> for &'a BTreeSet<T> {
    type Output = BTreeSet<T>;

    /// Returns the symmetric difference of `self` and `rhs` as a new `BTreeSet<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let a: BTreeSet<_> = vec![1, 2, 3].into_iter().collect();
    /// let b: BTreeSet<_> = vec![2, 3, 4].into_iter().collect();
    ///
    /// let result = &a ^ &b;
    /// let result_vec: Vec<_> = result.into_iter().collect();
    /// assert_eq!(result_vec, vec![1, 4]);
    /// ```
    fn bitxor(self, rhs: &BTreeSet<T>) -> BTreeSet<T> {
        self.symmetric_difference(rhs).cloned().collect()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, 'b, T: Ord + Clone> BitAnd<&'b BTreeSet<T>> for &'a BTreeSet<T> {
    type Output = BTreeSet<T>;

    /// Returns the intersection of `self` and `rhs` as a new `BTreeSet<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let a: BTreeSet<_> = vec![1, 2, 3].into_iter().collect();
    /// let b: BTreeSet<_> = vec![2, 3, 4].into_iter().collect();
    ///
    /// let result = &a & &b;
    /// let result_vec: Vec<_> = result.into_iter().collect();
    /// assert_eq!(result_vec, vec![2, 3]);
    /// ```
    fn bitand(self, rhs: &BTreeSet<T>) -> BTreeSet<T> {
        self.intersection(rhs).cloned().collect()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, 'b, T: Ord + Clone> BitOr<&'b BTreeSet<T>> for &'a BTreeSet<T> {
    type Output = BTreeSet<T>;

    /// Returns the union of `self` and `rhs` as a new `BTreeSet<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let a: BTreeSet<_> = vec![1, 2, 3].into_iter().collect();
    /// let b: BTreeSet<_> = vec![3, 4, 5].into_iter().collect();
    ///
    /// let result = &a | &b;
    /// let result_vec: Vec<_> = result.into_iter().collect();
    /// assert_eq!(result_vec, vec![1, 2, 3, 4, 5]);
    /// ```
    fn bitor(self, rhs: &BTreeSet<T>) -> BTreeSet<T> {
        self.union(rhs).cloned().collect()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Debug> Debug for BTreeSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "BTreeSet {{"));

        for (i, x) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{:?}", *x));
        }

        write!(f, "}}")
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<&'a T> { self.iter.next_back() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for Iter<'a, T> {}


#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<T> { self.iter.next_back() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for IntoIter<T> {}


impl<'a, T> Iterator for Range<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> { self.iter.next() }
}
impl<'a, T> DoubleEndedIterator for Range<'a, T> {
    fn next_back(&mut self) -> Option<&'a T> { self.iter.next_back() }
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

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: Ord> Iterator for Difference<'a, T> {
    type Item = &'a T;

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

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: Ord> Iterator for SymmetricDifference<'a, T> {
    type Item = &'a T;

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

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: Ord> Iterator for Intersection<'a, T> {
    type Item = &'a T;

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

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: Ord> Iterator for Union<'a, T> {
    type Item = &'a T;

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
    use prelude::*;

    use super::BTreeSet;
    use std::hash::{self, SipHasher};

    #[test]
    fn test_clone_eq() {
      let mut m = BTreeSet::new();

      m.insert(1);
      m.insert(2);

      assert!(m.clone() == m);
    }

    #[test]
    fn test_hash() {
      let mut x = BTreeSet::new();
      let mut y = BTreeSet::new();

      x.insert(1);
      x.insert(2);
      x.insert(3);

      y.insert(3);
      y.insert(2);
      y.insert(1);

      assert!(hash::hash::<_, SipHasher>(&x) == hash::hash::<_, SipHasher>(&y));
    }

    struct Counter<'a, 'b> {
        i: &'a mut usize,
        expected: &'b [i32],
    }

    impl<'a, 'b, 'c> FnMut<(&'c i32,)> for Counter<'a, 'b> {
        type Output = bool;

        extern "rust-call" fn call_mut(&mut self, (&x,): (&'c i32,)) -> bool {
            assert_eq!(x, self.expected[*self.i]);
            *self.i += 1;
            true
        }
    }

    fn check<F>(a: &[i32], b: &[i32], expected: &[i32], f: F) where
        // FIXME Replace Counter with `Box<FnMut(_) -> _>`
        F: FnOnce(&BTreeSet<i32>, &BTreeSet<i32>, Counter) -> bool,
    {
        let mut set_a = BTreeSet::new();
        let mut set_b = BTreeSet::new();

        for x in a { assert!(set_a.insert(*x)) }
        for y in b { assert!(set_b.insert(*y)) }

        let mut i = 0;
        f(&set_a, &set_b, Counter { i: &mut i, expected: expected });
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_intersection() {
        fn check_intersection(a: &[i32], b: &[i32], expected: &[i32]) {
            check(a, b, expected, |x, y, f| x.intersection(y).all(f))
        }

        check_intersection(&[], &[], &[]);
        check_intersection(&[1, 2, 3], &[], &[]);
        check_intersection(&[], &[1, 2, 3], &[]);
        check_intersection(&[2], &[1, 2, 3], &[2]);
        check_intersection(&[1, 2, 3], &[2], &[2]);
        check_intersection(&[11, 1, 3, 77, 103, 5, -5],
                           &[2, 11, 77, -9, -42, 5, 3],
                           &[3, 5, 11, 77]);
    }

    #[test]
    fn test_difference() {
        fn check_difference(a: &[i32], b: &[i32], expected: &[i32]) {
            check(a, b, expected, |x, y, f| x.difference(y).all(f))
        }

        check_difference(&[], &[], &[]);
        check_difference(&[1, 12], &[], &[1, 12]);
        check_difference(&[], &[1, 2, 3, 9], &[]);
        check_difference(&[1, 3, 5, 9, 11],
                         &[3, 9],
                         &[1, 5, 11]);
        check_difference(&[-5, 11, 22, 33, 40, 42],
                         &[-12, -5, 14, 23, 34, 38, 39, 50],
                         &[11, 22, 33, 40, 42]);
    }

    #[test]
    fn test_symmetric_difference() {
        fn check_symmetric_difference(a: &[i32], b: &[i32], expected: &[i32]) {
            check(a, b, expected, |x, y, f| x.symmetric_difference(y).all(f))
        }

        check_symmetric_difference(&[], &[], &[]);
        check_symmetric_difference(&[1, 2, 3], &[2], &[1, 3]);
        check_symmetric_difference(&[2], &[1, 2, 3], &[1, 3]);
        check_symmetric_difference(&[1, 3, 5, 9, 11],
                                   &[-2, 3, 9, 14, 22],
                                   &[-2, 1, 5, 11, 14, 22]);
    }

    #[test]
    fn test_union() {
        fn check_union(a: &[i32], b: &[i32], expected: &[i32]) {
            check(a, b, expected, |x, y, f| x.union(y).all(f))
        }

        check_union(&[], &[], &[]);
        check_union(&[1, 2, 3], &[2], &[1, 2, 3]);
        check_union(&[2], &[1, 2, 3], &[1, 2, 3]);
        check_union(&[1, 3, 5, 9, 11, 16, 19, 24],
                    &[-2, 1, 5, 9, 13, 19],
                    &[-2, 1, 3, 5, 9, 11, 13, 16, 19, 24]);
    }

    #[test]
    fn test_zip() {
        let mut x = BTreeSet::new();
        x.insert(5);
        x.insert(12);
        x.insert(11);

        let mut y = BTreeSet::new();
        y.insert("foo");
        y.insert("bar");

        let x = x;
        let y = y;
        let mut z = x.iter().zip(y.iter());

        // FIXME: #5801: this needs a type hint to compile...
        let result: Option<(&usize, & &'static str)> = z.next();
        assert_eq!(result.unwrap(), (&5, &("bar")));

        let result: Option<(&usize, & &'static str)> = z.next();
        assert_eq!(result.unwrap(), (&11, &("foo")));

        let result: Option<(&usize, & &'static str)> = z.next();
        assert!(result.is_none());
    }

    #[test]
    fn test_from_iter() {
        let xs = [1, 2, 3, 4, 5, 6, 7, 8, 9];

        let set: BTreeSet<_> = xs.iter().cloned().collect();

        for x in &xs {
            assert!(set.contains(x));
        }
    }

    #[test]
    fn test_show() {
        let mut set = BTreeSet::new();
        let empty = BTreeSet::<i32>::new();

        set.insert(1);
        set.insert(2);

        let set_str = format!("{:?}", set);

        assert_eq!(set_str, "BTreeSet {1, 2}");
        assert_eq!(format!("{:?}", empty), "BTreeSet {}");
    }
}
