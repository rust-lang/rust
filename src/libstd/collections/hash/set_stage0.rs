// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15883

use borrow::Borrow;
use clone::Clone;
use cmp::{Eq, PartialEq};
use core::marker::Sized;
use default::Default;
use fmt::Debug;
use fmt;
use hash::{self, Hash};
use iter::{
    Iterator, IntoIterator, ExactSizeIterator, IteratorExt, FromIterator, Map, Chain, Extend,
};
use ops::{BitOr, BitAnd, BitXor, Sub};
use option::Option::{Some, None, self};

use super::map::{self, HashMap, Keys, INITIAL_CAPACITY, RandomState, Hasher};
use super::state::HashState;

// Future Optimization (FIXME!)
// =============================
//
// Iteration over zero sized values is a noop. There is no need
// for `bucket.val` in the case of HashSet. I suppose we would need HKT
// to get rid of it properly.

/// An implementation of a hash set using the underlying representation of a
/// HashMap where the value is (). As with the `HashMap` type, a `HashSet`
/// requires that the elements implement the `Eq` and `Hash` traits.
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// // Type inference lets us omit an explicit type signature (which
/// // would be `HashSet<&str>` in this example).
/// let mut books = HashSet::new();
///
/// // Add some books.
/// books.insert("A Dance With Dragons");
/// books.insert("To Kill a Mockingbird");
/// books.insert("The Odyssey");
/// books.insert("The Great Gatsby");
///
/// // Check for a specific one.
/// if !books.contains(&("The Winds of Winter")) {
///     println!("We have {} books, but The Winds of Winter ain't one.",
///              books.len());
/// }
///
/// // Remove a book.
/// books.remove(&"The Odyssey");
///
/// // Iterate over everything.
/// for book in books.iter() {
///     println!("{}", *book);
/// }
/// ```
///
/// The easiest way to use `HashSet` with a custom type is to derive
/// `Eq` and `Hash`. We must also derive `PartialEq`, this will in the
/// future be implied by `Eq`.
///
/// ```
/// use std::collections::HashSet;
/// #[derive(Hash, Eq, PartialEq, Debug)]
/// struct Viking<'a> {
///     name: &'a str,
///     power: usize,
/// }
///
/// let mut vikings = HashSet::new();
///
/// vikings.insert(Viking { name: "Einar", power: 9 });
/// vikings.insert(Viking { name: "Einar", power: 9 });
/// vikings.insert(Viking { name: "Olaf", power: 4 });
/// vikings.insert(Viking { name: "Harald", power: 8 });
///
/// // Use derived implementation to print the vikings.
/// for x in vikings.iter() {
///     println!("{:?}", x);
/// }
/// ```
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct HashSet<T, S = RandomState> {
    map: HashMap<T, (), S>
}

impl<T: Hash<Hasher> + Eq> HashSet<T, RandomState> {
    /// Create an empty HashSet.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let mut set: HashSet<int> = HashSet::new();
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> HashSet<T, RandomState> {
        HashSet::with_capacity(INITIAL_CAPACITY)
    }

    /// Create an empty HashSet with space for at least `n` elements in
    /// the hash table.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let mut set: HashSet<int> = HashSet::with_capacity(10);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(capacity: usize) -> HashSet<T, RandomState> {
        HashSet { map: HashMap::with_capacity(capacity) }
    }
}

impl<T, S, H> HashSet<T, S>
    where T: Eq + Hash<H>,
          S: HashState<Hasher=H>,
          H: hash::Hasher<Output=u64>
{
    /// Creates a new empty hash set which will use the given hasher to hash
    /// keys.
    ///
    /// The hash set is also created with the default initial capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let mut set = HashSet::with_hash_state(s);
    /// set.insert(2);
    /// ```
    #[inline]
    #[unstable(feature = "std_misc", reason = "hasher stuff is unclear")]
    pub fn with_hash_state(hash_state: S) -> HashSet<T, S> {
        HashSet::with_capacity_and_hash_state(INITIAL_CAPACITY, hash_state)
    }

    /// Create an empty HashSet with space for at least `capacity`
    /// elements in the hash table, using `hasher` to hash the keys.
    ///
    /// Warning: `hasher` is normally randomly generated, and
    /// is designed to allow `HashSet`s to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let mut set = HashSet::with_capacity_and_hash_state(10, s);
    /// set.insert(1);
    /// ```
    #[inline]
    #[unstable(feature = "std_misc", reason = "hasher stuff is unclear")]
    pub fn with_capacity_and_hash_state(capacity: usize, hash_state: S)
                                        -> HashSet<T, S> {
        HashSet {
            map: HashMap::with_capacity_and_hash_state(capacity, hash_state),
        }
    }

    /// Returns the number of elements the set can hold without reallocating.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let set: HashSet<int> = HashSet::with_capacity(100);
    /// assert!(set.capacity() >= 100);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn capacity(&self) -> usize {
        self.map.capacity()
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the `HashSet`. The collection may reserve more space to avoid
    /// frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new allocation size overflows `usize`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let mut set: HashSet<int> = HashSet::new();
    /// set.reserve(10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve(&mut self, additional: usize) {
        self.map.reserve(additional)
    }

    /// Shrinks the capacity of the set as much as possible. It will drop
    /// down as much as possible while maintaining the internal rules
    /// and possibly leaving some space in accordance with the resize policy.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut set: HashSet<int> = HashSet::with_capacity(100);
    /// set.insert(1);
    /// set.insert(2);
    /// assert!(set.capacity() >= 100);
    /// set.shrink_to_fit();
    /// assert!(set.capacity() >= 2);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn shrink_to_fit(&mut self) {
        self.map.shrink_to_fit()
    }

    /// An iterator visiting all elements in arbitrary order.
    /// Iterator element type is &'a T.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let mut set = HashSet::new();
    /// set.insert("a");
    /// set.insert("b");
    ///
    /// // Will print in an arbitrary order.
    /// for x in set.iter() {
    ///     println!("{}", x);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter<T> {
        Iter { iter: self.map.keys() }
    }

    /// Creates a consuming iterator, that is, one that moves each value out
    /// of the set in arbitrary order. The set cannot be used after calling
    /// this.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let mut set = HashSet::new();
    /// set.insert("a".to_string());
    /// set.insert("b".to_string());
    ///
    /// // Not possible to collect to a Vec<String> with a regular `.iter()`.
    /// let v: Vec<String> = set.into_iter().collect();
    ///
    /// // Will print in an arbitrary order.
    /// for x in v.iter() {
    ///     println!("{}", x);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_iter(self) -> IntoIter<T> {
        fn first<A, B>((a, _): (A, B)) -> A { a }
        let first: fn((T, ())) -> T = first;

        IntoIter { iter: self.map.into_iter().map(first) }
    }

    /// Visit the values representing the difference.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let a: HashSet<int> = [1, 2, 3].iter().map(|&x| x).collect();
    /// let b: HashSet<int> = [4, 2, 3, 4].iter().map(|&x| x).collect();
    ///
    /// // Can be seen as `a - b`.
    /// for x in a.difference(&b) {
    ///     println!("{}", x); // Print 1
    /// }
    ///
    /// let diff: HashSet<int> = a.difference(&b).map(|&x| x).collect();
    /// assert_eq!(diff, [1].iter().map(|&x| x).collect());
    ///
    /// // Note that difference is not symmetric,
    /// // and `b - a` means something else:
    /// let diff: HashSet<int> = b.difference(&a).map(|&x| x).collect();
    /// assert_eq!(diff, [4].iter().map(|&x| x).collect());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn difference<'a>(&'a self, other: &'a HashSet<T, S>) -> Difference<'a, T, S> {
        Difference {
            iter: self.iter(),
            other: other,
        }
    }

    /// Visit the values representing the symmetric difference.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let a: HashSet<int> = [1, 2, 3].iter().map(|&x| x).collect();
    /// let b: HashSet<int> = [4, 2, 3, 4].iter().map(|&x| x).collect();
    ///
    /// // Print 1, 4 in arbitrary order.
    /// for x in a.symmetric_difference(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let diff1: HashSet<int> = a.symmetric_difference(&b).map(|&x| x).collect();
    /// let diff2: HashSet<int> = b.symmetric_difference(&a).map(|&x| x).collect();
    ///
    /// assert_eq!(diff1, diff2);
    /// assert_eq!(diff1, [1, 4].iter().map(|&x| x).collect());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn symmetric_difference<'a>(&'a self, other: &'a HashSet<T, S>)
        -> SymmetricDifference<'a, T, S> {
        SymmetricDifference { iter: self.difference(other).chain(other.difference(self)) }
    }

    /// Visit the values representing the intersection.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let a: HashSet<int> = [1, 2, 3].iter().map(|&x| x).collect();
    /// let b: HashSet<int> = [4, 2, 3, 4].iter().map(|&x| x).collect();
    ///
    /// // Print 2, 3 in arbitrary order.
    /// for x in a.intersection(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let diff: HashSet<int> = a.intersection(&b).map(|&x| x).collect();
    /// assert_eq!(diff, [2, 3].iter().map(|&x| x).collect());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn intersection<'a>(&'a self, other: &'a HashSet<T, S>) -> Intersection<'a, T, S> {
        Intersection {
            iter: self.iter(),
            other: other,
        }
    }

    /// Visit the values representing the union.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let a: HashSet<int> = [1, 2, 3].iter().map(|&x| x).collect();
    /// let b: HashSet<int> = [4, 2, 3, 4].iter().map(|&x| x).collect();
    ///
    /// // Print 1, 2, 3, 4 in arbitrary order.
    /// for x in a.union(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let diff: HashSet<int> = a.union(&b).map(|&x| x).collect();
    /// assert_eq!(diff, [1, 2, 3, 4].iter().map(|&x| x).collect());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn union<'a>(&'a self, other: &'a HashSet<T, S>) -> Union<'a, T, S> {
        Union { iter: self.iter().chain(other.difference(self)) }
    }

    /// Return the number of elements in the set
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut v = HashSet::new();
    /// assert_eq!(v.len(), 0);
    /// v.insert(1);
    /// assert_eq!(v.len(), 1);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize { self.map.len() }

    /// Returns true if the set contains no elements
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut v = HashSet::new();
    /// assert!(v.is_empty());
    /// v.insert(1);
    /// assert!(!v.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool { self.map.len() == 0 }

    /// Clears the set, returning all elements in an iterator.
    #[inline]
    #[unstable(feature = "std_misc",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn drain(&mut self) -> Drain<T> {
        fn first<A, B>((a, _): (A, B)) -> A { a }
        let first: fn((T, ())) -> T = first; // coerce to fn pointer

        Drain { iter: self.map.drain().map(first) }
    }

    /// Clears the set, removing all values.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut v = HashSet::new();
    /// v.insert(1);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn clear(&mut self) { self.map.clear() }

    /// Returns `true` if the set contains a value.
    ///
    /// The value may be any borrowed form of the set's value type, but
    /// `Hash` and `Eq` on the borrowed form *must* match those for
    /// the value type.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let set: HashSet<_> = [1, 2, 3].iter().cloned().collect();
    /// assert_eq!(set.contains(&1), true);
    /// assert_eq!(set.contains(&4), false);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn contains<Q: ?Sized>(&self, value: &Q) -> bool
        where T: Borrow<Q>, Q: Hash<H> + Eq
    {
        self.map.contains_key(value)
    }

    /// Returns `true` if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let a: HashSet<_> = [1, 2, 3].iter().cloned().collect();
    /// let mut b = HashSet::new();
    ///
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(4);
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(1);
    /// assert_eq!(a.is_disjoint(&b), false);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_disjoint(&self, other: &HashSet<T, S>) -> bool {
        self.iter().all(|v| !other.contains(v))
    }

    /// Returns `true` if the set is a subset of another.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let sup: HashSet<_> = [1, 2, 3].iter().cloned().collect();
    /// let mut set = HashSet::new();
    ///
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(2);
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(4);
    /// assert_eq!(set.is_subset(&sup), false);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_subset(&self, other: &HashSet<T, S>) -> bool {
        self.iter().all(|v| other.contains(v))
    }

    /// Returns `true` if the set is a superset of another.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let sub: HashSet<_> = [1, 2].iter().cloned().collect();
    /// let mut set = HashSet::new();
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_superset(&self, other: &HashSet<T, S>) -> bool {
        other.is_subset(self)
    }

    /// Adds a value to the set. Returns `true` if the value was not already
    /// present in the set.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut set = HashSet::new();
    ///
    /// assert_eq!(set.insert(2), true);
    /// assert_eq!(set.insert(2), false);
    /// assert_eq!(set.len(), 1);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(&mut self, value: T) -> bool { self.map.insert(value, ()).is_none() }

    /// Removes a value from the set. Returns `true` if the value was
    /// present in the set.
    ///
    /// The value may be any borrowed form of the set's value type, but
    /// `Hash` and `Eq` on the borrowed form *must* match those for
    /// the value type.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut set = HashSet::new();
    ///
    /// set.insert(2);
    /// assert_eq!(set.remove(&2), true);
    /// assert_eq!(set.remove(&2), false);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove<Q: ?Sized>(&mut self, value: &Q) -> bool
        where T: Borrow<Q>, Q: Hash<H> + Eq
    {
        self.map.remove(value).is_some()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S, H> PartialEq for HashSet<T, S>
    where T: Eq + Hash<H>,
          S: HashState<Hasher=H>,
          H: hash::Hasher<Output=u64>
{
    fn eq(&self, other: &HashSet<T, S>) -> bool {
        if self.len() != other.len() { return false; }

        self.iter().all(|key| other.contains(key))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S, H> Eq for HashSet<T, S>
    where T: Eq + Hash<H>,
          S: HashState<Hasher=H>,
          H: hash::Hasher<Output=u64>
{}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S, H> fmt::Debug for HashSet<T, S>
    where T: Eq + Hash<H> + fmt::Debug,
          S: HashState<Hasher=H>,
          H: hash::Hasher<Output=u64>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "HashSet {{"));

        for (i, x) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{:?}", *x));
        }

        write!(f, "}}")
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S, H> FromIterator<T> for HashSet<T, S>
    where T: Eq + Hash<H>,
          S: HashState<Hasher=H> + Default,
          H: hash::Hasher<Output=u64>
{
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> HashSet<T, S> {
        let iter = iter.into_iter();
        let lower = iter.size_hint().0;
        let mut set = HashSet::with_capacity_and_hash_state(lower, Default::default());
        set.extend(iter);
        set
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S, H> Extend<T> for HashSet<T, S>
    where T: Eq + Hash<H>,
          S: HashState<Hasher=H>,
          H: hash::Hasher<Output=u64>
{
    fn extend<I: IntoIterator<Item=T>>(&mut self, iter: I) {
        for k in iter {
            self.insert(k);
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S, H> Default for HashSet<T, S>
    where T: Eq + Hash<H>,
          S: HashState<Hasher=H> + Default,
          H: hash::Hasher<Output=u64>
{
    #[stable(feature = "rust1", since = "1.0.0")]
    fn default() -> HashSet<T, S> {
        HashSet::with_hash_state(Default::default())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, 'b, T, S, H> BitOr<&'b HashSet<T, S>> for &'a HashSet<T, S>
    where T: Eq + Hash<H> + Clone,
          S: HashState<Hasher=H> + Default,
          H: hash::Hasher<Output=u64>
{
    type Output = HashSet<T, S>;

    /// Returns the union of `self` and `rhs` as a new `HashSet<T, S>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let a: HashSet<_> = vec![1, 2, 3].into_iter().collect();
    /// let b: HashSet<_> = vec![3, 4, 5].into_iter().collect();
    ///
    /// let set = &a | &b;
    ///
    /// let mut i = 0;
    /// let expected = [1, 2, 3, 4, 5];
    /// for x in set.iter() {
    ///     assert!(expected.contains(x));
    ///     i += 1;
    /// }
    /// assert_eq!(i, expected.len());
    /// ```
    fn bitor(self, rhs: &HashSet<T, S>) -> HashSet<T, S> {
        self.union(rhs).cloned().collect()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, 'b, T, S, H> BitAnd<&'b HashSet<T, S>> for &'a HashSet<T, S>
    where T: Eq + Hash<H> + Clone,
          S: HashState<Hasher=H> + Default,
          H: hash::Hasher<Output=u64>
{
    type Output = HashSet<T, S>;

    /// Returns the intersection of `self` and `rhs` as a new `HashSet<T, S>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let a: HashSet<_> = vec![1, 2, 3].into_iter().collect();
    /// let b: HashSet<_> = vec![2, 3, 4].into_iter().collect();
    ///
    /// let set = &a & &b;
    ///
    /// let mut i = 0;
    /// let expected = [2, 3];
    /// for x in set.iter() {
    ///     assert!(expected.contains(x));
    ///     i += 1;
    /// }
    /// assert_eq!(i, expected.len());
    /// ```
    fn bitand(self, rhs: &HashSet<T, S>) -> HashSet<T, S> {
        self.intersection(rhs).cloned().collect()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, 'b, T, S, H> BitXor<&'b HashSet<T, S>> for &'a HashSet<T, S>
    where T: Eq + Hash<H> + Clone,
          S: HashState<Hasher=H> + Default,
          H: hash::Hasher<Output=u64>
{
    type Output = HashSet<T, S>;

    /// Returns the symmetric difference of `self` and `rhs` as a new `HashSet<T, S>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let a: HashSet<_> = vec![1, 2, 3].into_iter().collect();
    /// let b: HashSet<_> = vec![3, 4, 5].into_iter().collect();
    ///
    /// let set = &a ^ &b;
    ///
    /// let mut i = 0;
    /// let expected = [1, 2, 4, 5];
    /// for x in set.iter() {
    ///     assert!(expected.contains(x));
    ///     i += 1;
    /// }
    /// assert_eq!(i, expected.len());
    /// ```
    fn bitxor(self, rhs: &HashSet<T, S>) -> HashSet<T, S> {
        self.symmetric_difference(rhs).cloned().collect()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, 'b, T, S, H> Sub<&'b HashSet<T, S>> for &'a HashSet<T, S>
    where T: Eq + Hash<H> + Clone,
          S: HashState<Hasher=H> + Default,
          H: hash::Hasher<Output=u64>
{
    type Output = HashSet<T, S>;

    /// Returns the difference of `self` and `rhs` as a new `HashSet<T, S>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let a: HashSet<_> = vec![1, 2, 3].into_iter().collect();
    /// let b: HashSet<_> = vec![3, 4, 5].into_iter().collect();
    ///
    /// let set = &a - &b;
    ///
    /// let mut i = 0;
    /// let expected = [1, 2];
    /// for x in set.iter() {
    ///     assert!(expected.contains(x));
    ///     i += 1;
    /// }
    /// assert_eq!(i, expected.len());
    /// ```
    fn sub(self, rhs: &HashSet<T, S>) -> HashSet<T, S> {
        self.difference(rhs).cloned().collect()
    }
}

/// HashSet iterator
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, K: 'a> {
    iter: Keys<'a, K, ()>
}

/// HashSet move iterator
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<K> {
    iter: Map<map::IntoIter<K, ()>, fn((K, ())) -> K>
}

/// HashSet drain iterator
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Drain<'a, K: 'a> {
    iter: Map<map::Drain<'a, K, ()>, fn((K, ())) -> K>,
}

/// Intersection iterator
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Intersection<'a, T: 'a, S: 'a> {
    // iterator of the first set
    iter: Iter<'a, T>,
    // the second set
    other: &'a HashSet<T, S>,
}

/// Difference iterator
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Difference<'a, T: 'a, S: 'a> {
    // iterator of the first set
    iter: Iter<'a, T>,
    // the second set
    other: &'a HashSet<T, S>,
}

/// Symmetric difference iterator.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SymmetricDifference<'a, T: 'a, S: 'a> {
    iter: Chain<Difference<'a, T, S>, Difference<'a, T, S>>
}

/// Set union iterator.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Union<'a, T: 'a, S: 'a> {
    iter: Chain<Iter<'a, T>, Difference<'a, T, S>>
}

impl<'a, T, S, H> IntoIterator for &'a HashSet<T, S>
    where T: Eq + Hash<H>,
          S: HashState<Hasher=H>,
          H: hash::Hasher<Output=u64>
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<T, S, H> IntoIterator for HashSet<T, S>
    where T: Eq + Hash<H>,
          S: HashState<Hasher=H>,
          H: hash::Hasher<Output=u64>
{
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        self.into_iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K> Iterator for Iter<'a, K> {
    type Item = &'a K;

    fn next(&mut self) -> Option<&'a K> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K> ExactSizeIterator for Iter<'a, K> {
    fn len(&self) -> usize { self.iter.len() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K> Iterator for IntoIter<K> {
    type Item = K;

    fn next(&mut self) -> Option<K> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<K> ExactSizeIterator for IntoIter<K> {
    fn len(&self) -> usize { self.iter.len() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K> Iterator for Drain<'a, K> {
    type Item = K;

    fn next(&mut self) -> Option<K> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K> ExactSizeIterator for Drain<'a, K> {
    fn len(&self) -> usize { self.iter.len() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, S, H> Iterator for Intersection<'a, T, S>
    where T: Eq + Hash<H>,
          S: HashState<Hasher=H>,
          H: hash::Hasher<Output=u64>
{
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        loop {
            match self.iter.next() {
                None => return None,
                Some(elt) => if self.other.contains(elt) {
                    return Some(elt)
                },
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, S, H> Iterator for Difference<'a, T, S>
    where T: Eq + Hash<H>,
          S: HashState<Hasher=H>,
          H: hash::Hasher<Output=u64>
{
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        loop {
            match self.iter.next() {
                None => return None,
                Some(elt) => if !self.other.contains(elt) {
                    return Some(elt)
                },
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, S, H> Iterator for SymmetricDifference<'a, T, S>
    where T: Eq + Hash<H>,
          S: HashState<Hasher=H>,
          H: hash::Hasher<Output=u64>
{
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, S, H> Iterator for Union<'a, T, S>
    where T: Eq + Hash<H>,
          S: HashState<Hasher=H>,
          H: hash::Hasher<Output=u64>
{
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

#[cfg(test)]
mod test_set {
    use prelude::v1::*;

    use super::HashSet;

    #[test]
    fn test_disjoint() {
        let mut xs = HashSet::new();
        let mut ys = HashSet::new();
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(xs.insert(5));
        assert!(ys.insert(11));
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(xs.insert(7));
        assert!(xs.insert(19));
        assert!(xs.insert(4));
        assert!(ys.insert(2));
        assert!(ys.insert(-11));
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(ys.insert(7));
        assert!(!xs.is_disjoint(&ys));
        assert!(!ys.is_disjoint(&xs));
    }

    #[test]
    fn test_subset_and_superset() {
        let mut a = HashSet::new();
        assert!(a.insert(0));
        assert!(a.insert(5));
        assert!(a.insert(11));
        assert!(a.insert(7));

        let mut b = HashSet::new();
        assert!(b.insert(0));
        assert!(b.insert(7));
        assert!(b.insert(19));
        assert!(b.insert(250));
        assert!(b.insert(11));
        assert!(b.insert(200));

        assert!(!a.is_subset(&b));
        assert!(!a.is_superset(&b));
        assert!(!b.is_subset(&a));
        assert!(!b.is_superset(&a));

        assert!(b.insert(5));

        assert!(a.is_subset(&b));
        assert!(!a.is_superset(&b));
        assert!(!b.is_subset(&a));
        assert!(b.is_superset(&a));
    }

    #[test]
    fn test_iterate() {
        let mut a = HashSet::new();
        for i in 0..32 {
            assert!(a.insert(i));
        }
        let mut observed: u32 = 0;
        for k in &a {
            observed |= 1 << *k;
        }
        assert_eq!(observed, 0xFFFF_FFFF);
    }

    #[test]
    fn test_intersection() {
        let mut a = HashSet::new();
        let mut b = HashSet::new();

        assert!(a.insert(11));
        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(77));
        assert!(a.insert(103));
        assert!(a.insert(5));
        assert!(a.insert(-5));

        assert!(b.insert(2));
        assert!(b.insert(11));
        assert!(b.insert(77));
        assert!(b.insert(-9));
        assert!(b.insert(-42));
        assert!(b.insert(5));
        assert!(b.insert(3));

        let mut i = 0;
        let expected = [3, 5, 11, 77];
        for x in a.intersection(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_difference() {
        let mut a = HashSet::new();
        let mut b = HashSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));

        assert!(b.insert(3));
        assert!(b.insert(9));

        let mut i = 0;
        let expected = [1, 5, 11];
        for x in a.difference(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_symmetric_difference() {
        let mut a = HashSet::new();
        let mut b = HashSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));

        assert!(b.insert(-2));
        assert!(b.insert(3));
        assert!(b.insert(9));
        assert!(b.insert(14));
        assert!(b.insert(22));

        let mut i = 0;
        let expected = [-2, 1, 5, 11, 14, 22];
        for x in a.symmetric_difference(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_union() {
        let mut a = HashSet::new();
        let mut b = HashSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));
        assert!(a.insert(16));
        assert!(a.insert(19));
        assert!(a.insert(24));

        assert!(b.insert(-2));
        assert!(b.insert(1));
        assert!(b.insert(5));
        assert!(b.insert(9));
        assert!(b.insert(13));
        assert!(b.insert(19));

        let mut i = 0;
        let expected = [-2, 1, 3, 5, 9, 11, 13, 16, 19, 24];
        for x in a.union(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_from_iter() {
        let xs = [1, 2, 3, 4, 5, 6, 7, 8, 9];

        let set: HashSet<_> = xs.iter().cloned().collect();

        for x in &xs {
            assert!(set.contains(x));
        }
    }

    #[test]
    fn test_move_iter() {
        let hs = {
            let mut hs = HashSet::new();

            hs.insert('a');
            hs.insert('b');

            hs
        };

        let v = hs.into_iter().collect::<Vec<char>>();
        assert!(['a', 'b'] == v || ['b', 'a'] == v);
    }

    #[test]
    fn test_eq() {
        // These constants once happened to expose a bug in insert().
        // I'm keeping them around to prevent a regression.
        let mut s1 = HashSet::new();

        s1.insert(1);
        s1.insert(2);
        s1.insert(3);

        let mut s2 = HashSet::new();

        s2.insert(1);
        s2.insert(2);

        assert!(s1 != s2);

        s2.insert(3);

        assert_eq!(s1, s2);
    }

    #[test]
    fn test_show() {
        let mut set = HashSet::new();
        let empty = HashSet::<i32>::new();

        set.insert(1);
        set.insert(2);

        let set_str = format!("{:?}", set);

        assert!(set_str == "HashSet {1, 2}" || set_str == "HashSet {2, 1}");
        assert_eq!(format!("{:?}", empty), "HashSet {}");
    }

    #[test]
    fn test_trivial_drain() {
        let mut s = HashSet::<i32>::new();
        for _ in s.drain() {}
        assert!(s.is_empty());
        drop(s);

        let mut s = HashSet::<i32>::new();
        drop(s.drain());
        assert!(s.is_empty());
    }

    #[test]
    fn test_drain() {
        let mut s: HashSet<_> = (1..100).collect();

        // try this a bunch of times to make sure we don't screw up internal state.
        for _ in 0..20 {
            assert_eq!(s.len(), 99);

            {
                let mut last_i = 0;
                let mut d = s.drain();
                for (i, x) in d.by_ref().take(50).enumerate() {
                    last_i = i;
                    assert!(x != 0);
                }
                assert_eq!(last_i, 49);
            }

            for _ in &s { panic!("s should be empty!"); }

            // reset to try again.
            s.extend(1..100);
        }
    }
}
