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

use borrow::BorrowFrom;
use cmp::{Eq, Equiv, PartialEq};
use core::kinds::Sized;
use default::Default;
use fmt::Show;
use fmt;
use hash::{Hash, Hasher, RandomSipHasher};
use iter::{Iterator, FromIterator, FilterMap, Chain, Repeat, Zip, Extend};
use iter;
use option::{Some, None};
use result::{Ok, Err};

use super::map::{HashMap, Entries, MoveEntries, INITIAL_CAPACITY};

// FIXME(conventions): implement BitOr, BitAnd, BitXor, and Sub
// FIXME(conventions): update capacity management to match other collections (no auto-shrink)


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
/// #[deriving(Hash, Eq, PartialEq, Show)]
/// struct Viking<'a> {
///     name: &'a str,
///     power: uint,
/// }
///
/// let mut vikings = HashSet::new();
///
/// vikings.insert(Viking { name: "Einar", power: 9u });
/// vikings.insert(Viking { name: "Einar", power: 9u });
/// vikings.insert(Viking { name: "Olaf", power: 4u });
/// vikings.insert(Viking { name: "Harald", power: 8u });
///
/// // Use derived implementation to print the vikings.
/// for x in vikings.iter() {
///     println!("{}", x);
/// }
/// ```
#[deriving(Clone)]
pub struct HashSet<T, H = RandomSipHasher> {
    map: HashMap<T, (), H>
}

impl<T: Hash + Eq> HashSet<T, RandomSipHasher> {
    /// Create an empty HashSet.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let mut set: HashSet<int> = HashSet::new();
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn new() -> HashSet<T, RandomSipHasher> { unimplemented!() }

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
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn with_capacity(capacity: uint) -> HashSet<T, RandomSipHasher> { unimplemented!() }
}

impl<T: Eq + Hash<S>, S, H: Hasher<S>> HashSet<T, H> {
    /// Creates a new empty hash set which will use the given hasher to hash
    /// keys.
    ///
    /// The hash set is also created with the default initial capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// use std::hash::sip::SipHasher;
    ///
    /// let h = SipHasher::new();
    /// let mut set = HashSet::with_hasher(h);
    /// set.insert(2u);
    /// ```
    #[inline]
    pub fn with_hasher(hasher: H) -> HashSet<T, H> { unimplemented!() }

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
    /// use std::hash::sip::SipHasher;
    ///
    /// let h = SipHasher::new();
    /// let mut set = HashSet::with_capacity_and_hasher(10u, h);
    /// set.insert(1i);
    /// ```
    #[inline]
    pub fn with_capacity_and_hasher(capacity: uint, hasher: H) -> HashSet<T, H> { unimplemented!() }

    /// Reserve space for at least `n` elements in the hash table.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let mut set: HashSet<int> = HashSet::new();
    /// set.reserve(10);
    /// ```
    pub fn reserve(&mut self, n: uint) { unimplemented!() }

    /// Deprecated: use `contains` and `BorrowFrom`.
    #[deprecated = "use contains and BorrowFrom"]
    #[allow(deprecated)]
    pub fn contains_equiv<Sized? Q: Hash<S> + Equiv<T>>(&self, value: &Q) -> bool { unimplemented!() }

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
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn iter<'a>(&'a self) -> SetItems<'a, T> { unimplemented!() }

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
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn into_iter(self) -> SetMoveItems<T> { unimplemented!() }

    /// Visit the values representing the difference.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let a: HashSet<int> = [1i, 2, 3].iter().map(|&x| x).collect();
    /// let b: HashSet<int> = [4i, 2, 3, 4].iter().map(|&x| x).collect();
    ///
    /// // Can be seen as `a - b`.
    /// for x in a.difference(&b) {
    ///     println!("{}", x); // Print 1
    /// }
    ///
    /// let diff: HashSet<int> = a.difference(&b).map(|&x| x).collect();
    /// assert_eq!(diff, [1i].iter().map(|&x| x).collect());
    ///
    /// // Note that difference is not symmetric,
    /// // and `b - a` means something else:
    /// let diff: HashSet<int> = b.difference(&a).map(|&x| x).collect();
    /// assert_eq!(diff, [4i].iter().map(|&x| x).collect());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn difference<'a>(&'a self, other: &'a HashSet<T, H>) -> SetAlgebraItems<'a, T, H> { unimplemented!() }

    /// Visit the values representing the symmetric difference.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let a: HashSet<int> = [1i, 2, 3].iter().map(|&x| x).collect();
    /// let b: HashSet<int> = [4i, 2, 3, 4].iter().map(|&x| x).collect();
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
    /// assert_eq!(diff1, [1i, 4].iter().map(|&x| x).collect());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn symmetric_difference<'a>(&'a self, other: &'a HashSet<T, H>)
        -> Chain<SetAlgebraItems<'a, T, H>, SetAlgebraItems<'a, T, H>> { unimplemented!() }

    /// Visit the values representing the intersection.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let a: HashSet<int> = [1i, 2, 3].iter().map(|&x| x).collect();
    /// let b: HashSet<int> = [4i, 2, 3, 4].iter().map(|&x| x).collect();
    ///
    /// // Print 2, 3 in arbitrary order.
    /// for x in a.intersection(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let diff: HashSet<int> = a.intersection(&b).map(|&x| x).collect();
    /// assert_eq!(diff, [2i, 3].iter().map(|&x| x).collect());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn intersection<'a>(&'a self, other: &'a HashSet<T, H>)
        -> SetAlgebraItems<'a, T, H> { unimplemented!() }

    /// Visit the values representing the union.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let a: HashSet<int> = [1i, 2, 3].iter().map(|&x| x).collect();
    /// let b: HashSet<int> = [4i, 2, 3, 4].iter().map(|&x| x).collect();
    ///
    /// // Print 1, 2, 3, 4 in arbitrary order.
    /// for x in a.union(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let diff: HashSet<int> = a.union(&b).map(|&x| x).collect();
    /// assert_eq!(diff, [1i, 2, 3, 4].iter().map(|&x| x).collect());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn union<'a>(&'a self, other: &'a HashSet<T, H>)
        -> Chain<SetItems<'a, T>, SetAlgebraItems<'a, T, H>> { unimplemented!() }

    /// Return the number of elements in the set
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut v = HashSet::new();
    /// assert_eq!(v.len(), 0);
    /// v.insert(1u);
    /// assert_eq!(v.len(), 1);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn len(&self) -> uint { unimplemented!() }

    /// Returns true if the set contains no elements
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut v = HashSet::new();
    /// assert!(v.is_empty());
    /// v.insert(1u);
    /// assert!(!v.is_empty());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn is_empty(&self) -> bool { unimplemented!() }

    /// Clears the set, removing all values.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut v = HashSet::new();
    /// v.insert(1u);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn clear(&mut self) { unimplemented!() }

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
    /// let set: HashSet<uint> = [1, 2, 3].iter().map(|&x| x).collect();
    /// assert_eq!(set.contains(&1), true);
    /// assert_eq!(set.contains(&4), false);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn contains<Sized? Q>(&self, value: &Q) -> bool
        where Q: BorrowFrom<T> + Hash<S> + Eq
    { unimplemented!() }

    /// Returns `true` if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let a: HashSet<uint> = [1, 2, 3].iter().map(|&x| x).collect();
    /// let mut b: HashSet<uint> = HashSet::new();
    ///
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(4);
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(1);
    /// assert_eq!(a.is_disjoint(&b), false);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn is_disjoint(&self, other: &HashSet<T, H>) -> bool { unimplemented!() }

    /// Returns `true` if the set is a subset of another.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let sup: HashSet<uint> = [1, 2, 3].iter().map(|&x| x).collect();
    /// let mut set: HashSet<uint> = HashSet::new();
    ///
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(2);
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(4);
    /// assert_eq!(set.is_subset(&sup), false);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn is_subset(&self, other: &HashSet<T, H>) -> bool { unimplemented!() }

    /// Returns `true` if the set is a superset of another.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let sub: HashSet<uint> = [1, 2].iter().map(|&x| x).collect();
    /// let mut set: HashSet<uint> = HashSet::new();
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
    pub fn is_superset(&self, other: &HashSet<T, H>) -> bool { unimplemented!() }

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
    /// assert_eq!(set.insert(2u), true);
    /// assert_eq!(set.insert(2), false);
    /// assert_eq!(set.len(), 1);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn insert(&mut self, value: T) -> bool { unimplemented!() }

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
    /// set.insert(2u);
    /// assert_eq!(set.remove(&2), true);
    /// assert_eq!(set.remove(&2), false);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn remove<Sized? Q>(&mut self, value: &Q) -> bool
        where Q: BorrowFrom<T> + Hash<S> + Eq
    { unimplemented!() }
}

impl<T: Eq + Hash<S>, S, H: Hasher<S>> PartialEq for HashSet<T, H> {
    fn eq(&self, other: &HashSet<T, H>) -> bool { unimplemented!() }
}

impl<T: Eq + Hash<S>, S, H: Hasher<S>> Eq for HashSet<T, H> {}

impl<T: Eq + Hash<S> + fmt::Show, S, H: Hasher<S>> fmt::Show for HashSet<T, H> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { unimplemented!() }
}

impl<T: Eq + Hash<S>, S, H: Hasher<S> + Default> FromIterator<T> for HashSet<T, H> {
    fn from_iter<I: Iterator<T>>(iter: I) -> HashSet<T, H> { unimplemented!() }
}

impl<T: Eq + Hash<S>, S, H: Hasher<S> + Default> Extend<T> for HashSet<T, H> {
    fn extend<I: Iterator<T>>(&mut self, mut iter: I) { unimplemented!() }
}

impl<T: Eq + Hash<S>, S, H: Hasher<S> + Default> Default for HashSet<T, H> {
    fn default() -> HashSet<T, H> { unimplemented!() }
}

/// HashSet iterator
pub type SetItems<'a, K> =
    iter::Map<'static, (&'a K, &'a ()), &'a K, Entries<'a, K, ()>>;

/// HashSet move iterator
pub type SetMoveItems<K> =
    iter::Map<'static, (K, ()), K, MoveEntries<K, ()>>;

// `Repeat` is used to feed the filter closure an explicit capture
// of a reference to the other set
/// Set operations iterator
pub type SetAlgebraItems<'a, T, H> =
    FilterMap<'static, (&'a HashSet<T, H>, &'a T), &'a T,
              Zip<Repeat<&'a HashSet<T, H>>, SetItems<'a, T>>>;
