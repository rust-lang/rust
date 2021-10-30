// This is pretty much entirely stolen from TreeSet, since BTreeMap has an identical interface
// to TreeMap

use crate::vec::Vec;
use core::borrow::Borrow;
use core::cmp::Ordering::{Equal, Greater, Less};
use core::cmp::{max, min};
use core::fmt::{self, Debug};
use core::iter::{FromIterator, FusedIterator, Peekable};
use core::ops::{BitAnd, BitOr, BitXor, RangeBounds, Sub};

use super::map::{BTreeMap, Keys};
use super::merge_iter::MergeIterInner;
use super::Recover;

// FIXME(conventions): implement bounded iterators

/// A set based on a B-Tree.
///
/// See [`BTreeMap`]'s documentation for a detailed discussion of this collection's performance
/// benefits and drawbacks.
///
/// It is a logic error for an item to be modified in such a way that the item's ordering relative
/// to any other item, as determined by the [`Ord`] trait, changes while it is in the set. This is
/// normally only possible through [`Cell`], [`RefCell`], global state, I/O, or unsafe code.
/// The behavior resulting from such a logic error is not specified (it could include panics,
/// incorrect results, aborts, memory leaks, or non-termination) but will not be undefined
/// behavior.
///
/// [`Ord`]: core::cmp::Ord
/// [`Cell`]: core::cell::Cell
/// [`RefCell`]: core::cell::RefCell
///
/// # Examples
///
/// ```
/// use std::collections::BTreeSet;
///
/// // Type inference lets us omit an explicit type signature (which
/// // would be `BTreeSet<&str>` in this example).
/// let mut books = BTreeSet::new();
///
/// // Add some books.
/// books.insert("A Dance With Dragons");
/// books.insert("To Kill a Mockingbird");
/// books.insert("The Odyssey");
/// books.insert("The Great Gatsby");
///
/// // Check for a specific one.
/// if !books.contains("The Winds of Winter") {
///     println!("We have {} books, but The Winds of Winter ain't one.",
///              books.len());
/// }
///
/// // Remove a book.
/// books.remove("The Odyssey");
///
/// // Iterate over everything.
/// for book in &books {
///     println!("{}", book);
/// }
/// ```
///
/// A `BTreeSet` with a known list of items can be initialized from an array:
///
/// ```
/// use std::collections::BTreeSet;
///
/// let set = BTreeSet::from([1, 2, 3]);
/// ```
#[derive(Hash, PartialEq, Eq, Ord, PartialOrd)]
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "BTreeSet")]
pub struct BTreeSet<T> {
    map: BTreeMap<T, ()>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Clone> Clone for BTreeSet<T> {
    fn clone(&self) -> Self {
        BTreeSet { map: self.map.clone() }
    }

    fn clone_from(&mut self, other: &Self) {
        self.map.clone_from(&other.map);
    }
}

/// An iterator over the items of a `BTreeSet`.
///
/// This `struct` is created by the [`iter`] method on [`BTreeSet`].
/// See its documentation for more.
///
/// [`iter`]: BTreeSet::iter
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    iter: Keys<'a, T, ()>,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Iter").field(&self.iter.clone()).finish()
    }
}

/// An owning iterator over the items of a `BTreeSet`.
///
/// This `struct` is created by the [`into_iter`] method on [`BTreeSet`]
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
///
/// [`into_iter`]: BTreeSet#method.into_iter
/// [`IntoIterator`]: core::iter::IntoIterator
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
pub struct IntoIter<T> {
    iter: super::map::IntoIter<T, ()>,
}

/// An iterator over a sub-range of items in a `BTreeSet`.
///
/// This `struct` is created by the [`range`] method on [`BTreeSet`].
/// See its documentation for more.
///
/// [`range`]: BTreeSet::range
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Debug)]
#[stable(feature = "btree_range", since = "1.17.0")]
pub struct Range<'a, T: 'a> {
    iter: super::map::Range<'a, T, ()>,
}

/// A lazy iterator producing elements in the difference of `BTreeSet`s.
///
/// This `struct` is created by the [`difference`] method on [`BTreeSet`].
/// See its documentation for more.
///
/// [`difference`]: BTreeSet::difference
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Difference<'a, T: 'a> {
    inner: DifferenceInner<'a, T>,
}
#[derive(Debug)]
enum DifferenceInner<'a, T: 'a> {
    Stitch {
        // iterate all of `self` and some of `other`, spotting matches along the way
        self_iter: Iter<'a, T>,
        other_iter: Peekable<Iter<'a, T>>,
    },
    Search {
        // iterate `self`, look up in `other`
        self_iter: Iter<'a, T>,
        other_set: &'a BTreeSet<T>,
    },
    Iterate(Iter<'a, T>), // simply produce all values in `self`
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug> fmt::Debug for Difference<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Difference").field(&self.inner).finish()
    }
}

/// A lazy iterator producing elements in the symmetric difference of `BTreeSet`s.
///
/// This `struct` is created by the [`symmetric_difference`] method on
/// [`BTreeSet`]. See its documentation for more.
///
/// [`symmetric_difference`]: BTreeSet::symmetric_difference
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SymmetricDifference<'a, T: 'a>(MergeIterInner<Iter<'a, T>>);

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug> fmt::Debug for SymmetricDifference<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SymmetricDifference").field(&self.0).finish()
    }
}

/// A lazy iterator producing elements in the intersection of `BTreeSet`s.
///
/// This `struct` is created by the [`intersection`] method on [`BTreeSet`].
/// See its documentation for more.
///
/// [`intersection`]: BTreeSet::intersection
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Intersection<'a, T: 'a> {
    inner: IntersectionInner<'a, T>,
}
#[derive(Debug)]
enum IntersectionInner<'a, T: 'a> {
    Stitch {
        // iterate similarly sized sets jointly, spotting matches along the way
        a: Iter<'a, T>,
        b: Iter<'a, T>,
    },
    Search {
        // iterate a small set, look up in the large set
        small_iter: Iter<'a, T>,
        large_set: &'a BTreeSet<T>,
    },
    Answer(Option<&'a T>), // return a specific value or emptiness
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug> fmt::Debug for Intersection<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Intersection").field(&self.inner).finish()
    }
}

/// A lazy iterator producing elements in the union of `BTreeSet`s.
///
/// This `struct` is created by the [`union`] method on [`BTreeSet`].
/// See its documentation for more.
///
/// [`union`]: BTreeSet::union
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Union<'a, T: 'a>(MergeIterInner<Iter<'a, T>>);

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug> fmt::Debug for Union<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Union").field(&self.0).finish()
    }
}

// This constant is used by functions that compare two sets.
// It estimates the relative size at which searching performs better
// than iterating, based on the benchmarks in
// https://github.com/ssomers/rust_bench_btreeset_intersection.
// It's used to divide rather than multiply sizes, to rule out overflow,
// and it's a power of two to make that division cheap.
const ITER_PERFORMANCE_TIPPING_SIZE_DIFF: usize = 16;

impl<T> BTreeSet<T> {
    /// Makes a new, empty `BTreeSet`.
    ///
    /// Does not allocate anything on its own.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(unused_mut)]
    /// use std::collections::BTreeSet;
    ///
    /// let mut set: BTreeSet<i32> = BTreeSet::new();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_btree_new", issue = "71835")]
    #[must_use]
    pub const fn new() -> BTreeSet<T> {
        BTreeSet { map: BTreeMap::new() }
    }

    /// Constructs a double-ended iterator over a sub-range of elements in the set.
    /// The simplest way is to use the range syntax `min..max`, thus `range(min..max)` will
    /// yield elements from min (inclusive) to max (exclusive).
    /// The range may also be entered as `(Bound<T>, Bound<T>)`, so for example
    /// `range((Excluded(4), Included(10)))` will yield a left-exclusive, right-inclusive
    /// range from 4 to 10.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    /// use std::ops::Bound::Included;
    ///
    /// let mut set = BTreeSet::new();
    /// set.insert(3);
    /// set.insert(5);
    /// set.insert(8);
    /// for &elem in set.range((Included(&4), Included(&8))) {
    ///     println!("{}", elem);
    /// }
    /// assert_eq!(Some(&5), set.range(4..).next());
    /// ```
    #[stable(feature = "btree_range", since = "1.17.0")]
    pub fn range<K: ?Sized, R>(&self, range: R) -> Range<'_, T>
    where
        K: Ord,
        T: Borrow<K> + Ord,
        R: RangeBounds<K>,
    {
        Range { iter: self.map.range(range) }
    }

    /// Visits the values representing the difference,
    /// i.e., the values that are in `self` but not in `other`,
    /// in ascending order.
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
    /// let diff: Vec<_> = a.difference(&b).cloned().collect();
    /// assert_eq!(diff, [1]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn difference<'a>(&'a self, other: &'a BTreeSet<T>) -> Difference<'a, T>
    where
        T: Ord,
    {
        let (self_min, self_max) =
            if let (Some(self_min), Some(self_max)) = (self.first(), self.last()) {
                (self_min, self_max)
            } else {
                return Difference { inner: DifferenceInner::Iterate(self.iter()) };
            };
        let (other_min, other_max) =
            if let (Some(other_min), Some(other_max)) = (other.first(), other.last()) {
                (other_min, other_max)
            } else {
                return Difference { inner: DifferenceInner::Iterate(self.iter()) };
            };
        Difference {
            inner: match (self_min.cmp(other_max), self_max.cmp(other_min)) {
                (Greater, _) | (_, Less) => DifferenceInner::Iterate(self.iter()),
                (Equal, _) => {
                    let mut self_iter = self.iter();
                    self_iter.next();
                    DifferenceInner::Iterate(self_iter)
                }
                (_, Equal) => {
                    let mut self_iter = self.iter();
                    self_iter.next_back();
                    DifferenceInner::Iterate(self_iter)
                }
                _ if self.len() <= other.len() / ITER_PERFORMANCE_TIPPING_SIZE_DIFF => {
                    DifferenceInner::Search { self_iter: self.iter(), other_set: other }
                }
                _ => DifferenceInner::Stitch {
                    self_iter: self.iter(),
                    other_iter: other.iter().peekable(),
                },
            },
        }
    }

    /// Visits the values representing the symmetric difference,
    /// i.e., the values that are in `self` or in `other` but not in both,
    /// in ascending order.
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
    /// let sym_diff: Vec<_> = a.symmetric_difference(&b).cloned().collect();
    /// assert_eq!(sym_diff, [1, 3]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn symmetric_difference<'a>(&'a self, other: &'a BTreeSet<T>) -> SymmetricDifference<'a, T>
    where
        T: Ord,
    {
        SymmetricDifference(MergeIterInner::new(self.iter(), other.iter()))
    }

    /// Visits the values representing the intersection,
    /// i.e., the values that are both in `self` and `other`,
    /// in ascending order.
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
    /// let intersection: Vec<_> = a.intersection(&b).cloned().collect();
    /// assert_eq!(intersection, [2]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn intersection<'a>(&'a self, other: &'a BTreeSet<T>) -> Intersection<'a, T>
    where
        T: Ord,
    {
        let (self_min, self_max) =
            if let (Some(self_min), Some(self_max)) = (self.first(), self.last()) {
                (self_min, self_max)
            } else {
                return Intersection { inner: IntersectionInner::Answer(None) };
            };
        let (other_min, other_max) =
            if let (Some(other_min), Some(other_max)) = (other.first(), other.last()) {
                (other_min, other_max)
            } else {
                return Intersection { inner: IntersectionInner::Answer(None) };
            };
        Intersection {
            inner: match (self_min.cmp(other_max), self_max.cmp(other_min)) {
                (Greater, _) | (_, Less) => IntersectionInner::Answer(None),
                (Equal, _) => IntersectionInner::Answer(Some(self_min)),
                (_, Equal) => IntersectionInner::Answer(Some(self_max)),
                _ if self.len() <= other.len() / ITER_PERFORMANCE_TIPPING_SIZE_DIFF => {
                    IntersectionInner::Search { small_iter: self.iter(), large_set: other }
                }
                _ if other.len() <= self.len() / ITER_PERFORMANCE_TIPPING_SIZE_DIFF => {
                    IntersectionInner::Search { small_iter: other.iter(), large_set: self }
                }
                _ => IntersectionInner::Stitch { a: self.iter(), b: other.iter() },
            },
        }
    }

    /// Visits the values representing the union,
    /// i.e., all the values in `self` or `other`, without duplicates,
    /// in ascending order.
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
    /// let union: Vec<_> = a.union(&b).cloned().collect();
    /// assert_eq!(union, [1, 2]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn union<'a>(&'a self, other: &'a BTreeSet<T>) -> Union<'a, T>
    where
        T: Ord,
    {
        Union(MergeIterInner::new(self.iter(), other.iter()))
    }

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
    pub fn contains<Q: ?Sized>(&self, value: &Q) -> bool
    where
        T: Borrow<Q> + Ord,
        Q: Ord,
    {
        self.map.contains_key(value)
    }

    /// Returns a reference to the value in the set, if any, that is equal to the given value.
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
    /// assert_eq!(set.get(&2), Some(&2));
    /// assert_eq!(set.get(&4), None);
    /// ```
    #[stable(feature = "set_recovery", since = "1.9.0")]
    pub fn get<Q: ?Sized>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q> + Ord,
        Q: Ord,
    {
        Recover::get(&self.map, value)
    }

    /// Returns `true` if `self` has no elements in common with `other`.
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_disjoint(&self, other: &BTreeSet<T>) -> bool
    where
        T: Ord,
    {
        self.intersection(other).next().is_none()
    }

    /// Returns `true` if the set is a subset of another,
    /// i.e., `other` contains at least all the values in `self`.
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_subset(&self, other: &BTreeSet<T>) -> bool
    where
        T: Ord,
    {
        // Same result as self.difference(other).next().is_none()
        // but the code below is faster (hugely in some cases).
        if self.len() > other.len() {
            return false;
        }
        let (self_min, self_max) =
            if let (Some(self_min), Some(self_max)) = (self.first(), self.last()) {
                (self_min, self_max)
            } else {
                return true; // self is empty
            };
        let (other_min, other_max) =
            if let (Some(other_min), Some(other_max)) = (other.first(), other.last()) {
                (other_min, other_max)
            } else {
                return false; // other is empty
            };
        let mut self_iter = self.iter();
        match self_min.cmp(other_min) {
            Less => return false,
            Equal => {
                self_iter.next();
            }
            Greater => (),
        }
        match self_max.cmp(other_max) {
            Greater => return false,
            Equal => {
                self_iter.next_back();
            }
            Less => (),
        }
        if self_iter.len() <= other.len() / ITER_PERFORMANCE_TIPPING_SIZE_DIFF {
            for next in self_iter {
                if !other.contains(next) {
                    return false;
                }
            }
        } else {
            let mut other_iter = other.iter();
            other_iter.next();
            other_iter.next_back();
            let mut self_next = self_iter.next();
            while let Some(self1) = self_next {
                match other_iter.next().map_or(Less, |other1| self1.cmp(other1)) {
                    Less => return false,
                    Equal => self_next = self_iter.next(),
                    Greater => (),
                }
            }
        }
        true
    }

    /// Returns `true` if the set is a superset of another,
    /// i.e., `self` contains at least all the values in `other`.
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_superset(&self, other: &BTreeSet<T>) -> bool
    where
        T: Ord,
    {
        other.is_subset(self)
    }

    /// Returns a reference to the first value in the set, if any.
    /// This value is always the minimum of all values in the set.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(map_first_last)]
    /// use std::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    /// assert_eq!(set.first(), None);
    /// set.insert(1);
    /// assert_eq!(set.first(), Some(&1));
    /// set.insert(2);
    /// assert_eq!(set.first(), Some(&1));
    /// ```
    #[must_use]
    #[unstable(feature = "map_first_last", issue = "62924")]
    pub fn first(&self) -> Option<&T>
    where
        T: Ord,
    {
        self.map.first_key_value().map(|(k, _)| k)
    }

    /// Returns a reference to the last value in the set, if any.
    /// This value is always the maximum of all values in the set.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(map_first_last)]
    /// use std::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    /// assert_eq!(set.last(), None);
    /// set.insert(1);
    /// assert_eq!(set.last(), Some(&1));
    /// set.insert(2);
    /// assert_eq!(set.last(), Some(&2));
    /// ```
    #[must_use]
    #[unstable(feature = "map_first_last", issue = "62924")]
    pub fn last(&self) -> Option<&T>
    where
        T: Ord,
    {
        self.map.last_key_value().map(|(k, _)| k)
    }

    /// Removes the first value from the set and returns it, if any.
    /// The first value is always the minimum value in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(map_first_last)]
    /// use std::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    ///
    /// set.insert(1);
    /// while let Some(n) = set.pop_first() {
    ///     assert_eq!(n, 1);
    /// }
    /// assert!(set.is_empty());
    /// ```
    #[unstable(feature = "map_first_last", issue = "62924")]
    pub fn pop_first(&mut self) -> Option<T>
    where
        T: Ord,
    {
        self.map.pop_first().map(|kv| kv.0)
    }

    /// Removes the last value from the set and returns it, if any.
    /// The last value is always the maximum value in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(map_first_last)]
    /// use std::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    ///
    /// set.insert(1);
    /// while let Some(n) = set.pop_last() {
    ///     assert_eq!(n, 1);
    /// }
    /// assert!(set.is_empty());
    /// ```
    #[unstable(feature = "map_first_last", issue = "62924")]
    pub fn pop_last(&mut self) -> Option<T>
    where
        T: Ord,
    {
        self.map.pop_last().map(|kv| kv.0)
    }

    /// Adds a value to the set.
    ///
    /// If the set did not have this value present, `true` is returned.
    ///
    /// If the set did have this value present, `false` is returned, and the
    /// entry is not updated. See the [module-level documentation] for more.
    ///
    /// [module-level documentation]: index.html#insert-and-complex-keys
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
    pub fn insert(&mut self, value: T) -> bool
    where
        T: Ord,
    {
        self.map.insert(value, ()).is_none()
    }

    /// Adds a value to the set, replacing the existing value, if any, that is equal to the given
    /// one. Returns the replaced value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    /// set.insert(Vec::<i32>::new());
    ///
    /// assert_eq!(set.get(&[][..]).unwrap().capacity(), 0);
    /// set.replace(Vec::with_capacity(10));
    /// assert_eq!(set.get(&[][..]).unwrap().capacity(), 10);
    /// ```
    #[stable(feature = "set_recovery", since = "1.9.0")]
    pub fn replace(&mut self, value: T) -> Option<T>
    where
        T: Ord,
    {
        Recover::replace(&mut self.map, value)
    }

    /// Removes a value from the set. Returns whether the value was
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
    pub fn remove<Q: ?Sized>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q> + Ord,
        Q: Ord,
    {
        self.map.remove(value).is_some()
    }

    /// Removes and returns the value in the set, if any, that is equal to the given one.
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
    /// let mut set: BTreeSet<_> = [1, 2, 3].iter().cloned().collect();
    /// assert_eq!(set.take(&2), Some(2));
    /// assert_eq!(set.take(&2), None);
    /// ```
    #[stable(feature = "set_recovery", since = "1.9.0")]
    pub fn take<Q: ?Sized>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q> + Ord,
        Q: Ord,
    {
        Recover::take(&mut self.map, value)
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` such that `f(&e)` returns `false`.
    /// The elements are visited in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let xs = [1, 2, 3, 4, 5, 6];
    /// let mut set: BTreeSet<i32> = xs.iter().cloned().collect();
    /// // Keep only the even numbers.
    /// set.retain(|&k| k % 2 == 0);
    /// assert!(set.iter().eq([2, 4, 6].iter()));
    /// ```
    #[stable(feature = "btree_retain", since = "1.53.0")]
    pub fn retain<F>(&mut self, mut f: F)
    where
        T: Ord,
        F: FnMut(&T) -> bool,
    {
        self.drain_filter(|v| !f(v));
    }

    /// Moves all elements from `other` into `Self`, leaving `other` empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut a = BTreeSet::new();
    /// a.insert(1);
    /// a.insert(2);
    /// a.insert(3);
    ///
    /// let mut b = BTreeSet::new();
    /// b.insert(3);
    /// b.insert(4);
    /// b.insert(5);
    ///
    /// a.append(&mut b);
    ///
    /// assert_eq!(a.len(), 5);
    /// assert_eq!(b.len(), 0);
    ///
    /// assert!(a.contains(&1));
    /// assert!(a.contains(&2));
    /// assert!(a.contains(&3));
    /// assert!(a.contains(&4));
    /// assert!(a.contains(&5));
    /// ```
    #[stable(feature = "btree_append", since = "1.11.0")]
    pub fn append(&mut self, other: &mut Self)
    where
        T: Ord,
    {
        self.map.append(&mut other.map);
    }

    /// Splits the collection into two at the given value. Returns everything after the given value,
    /// including the value.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut a = BTreeSet::new();
    /// a.insert(1);
    /// a.insert(2);
    /// a.insert(3);
    /// a.insert(17);
    /// a.insert(41);
    ///
    /// let b = a.split_off(&3);
    ///
    /// assert_eq!(a.len(), 2);
    /// assert_eq!(b.len(), 3);
    ///
    /// assert!(a.contains(&1));
    /// assert!(a.contains(&2));
    ///
    /// assert!(b.contains(&3));
    /// assert!(b.contains(&17));
    /// assert!(b.contains(&41));
    /// ```
    #[stable(feature = "btree_split_off", since = "1.11.0")]
    pub fn split_off<Q: ?Sized + Ord>(&mut self, value: &Q) -> Self
    where
        T: Borrow<Q> + Ord,
    {
        BTreeSet { map: self.map.split_off(value) }
    }

    /// Creates an iterator that visits all values in ascending order and uses a closure
    /// to determine if a value should be removed.
    ///
    /// If the closure returns `true`, the value is removed from the set and yielded. If
    /// the closure returns `false`, or panics, the value remains in the set and will
    /// not be yielded.
    ///
    /// If the iterator is only partially consumed or not consumed at all, each of the
    /// remaining values is still subjected to the closure and removed and dropped if it
    /// returns `true`.
    ///
    /// It is unspecified how many more values will be subjected to the closure if a
    /// panic occurs in the closure, or if a panic occurs while dropping a value, or if
    /// the `DrainFilter` itself is leaked.
    ///
    /// # Examples
    ///
    /// Splitting a set into even and odd values, reusing the original set:
    ///
    /// ```
    /// #![feature(btree_drain_filter)]
    /// use std::collections::BTreeSet;
    ///
    /// let mut set: BTreeSet<i32> = (0..8).collect();
    /// let evens: BTreeSet<_> = set.drain_filter(|v| v % 2 == 0).collect();
    /// let odds = set;
    /// assert_eq!(evens.into_iter().collect::<Vec<_>>(), vec![0, 2, 4, 6]);
    /// assert_eq!(odds.into_iter().collect::<Vec<_>>(), vec![1, 3, 5, 7]);
    /// ```
    #[unstable(feature = "btree_drain_filter", issue = "70530")]
    pub fn drain_filter<'a, F>(&'a mut self, pred: F) -> DrainFilter<'a, T, F>
    where
        T: Ord,
        F: 'a + FnMut(&T) -> bool,
    {
        DrainFilter { pred, inner: self.map.drain_filter_inner() }
    }

    /// Gets an iterator that visits the values in the `BTreeSet` in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let set: BTreeSet<usize> = [1, 2, 3].iter().cloned().collect();
    /// let mut set_iter = set.iter();
    /// assert_eq!(set_iter.next(), Some(&1));
    /// assert_eq!(set_iter.next(), Some(&2));
    /// assert_eq!(set_iter.next(), Some(&3));
    /// assert_eq!(set_iter.next(), None);
    /// ```
    ///
    /// Values returned by the iterator are returned in ascending order:
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let set: BTreeSet<usize> = [3, 1, 2].iter().cloned().collect();
    /// let mut set_iter = set.iter();
    /// assert_eq!(set_iter.next(), Some(&1));
    /// assert_eq!(set_iter.next(), Some(&2));
    /// assert_eq!(set_iter.next(), Some(&3));
    /// assert_eq!(set_iter.next(), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter { iter: self.map.keys() }
    }

    /// Returns the number of elements in the set.
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
    #[rustc_const_unstable(feature = "const_btree_new", issue = "71835")]
    pub const fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the set contains no elements.
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
    #[rustc_const_unstable(feature = "const_btree_new", issue = "71835")]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> FromIterator<T> for BTreeSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> BTreeSet<T> {
        let mut inputs: Vec<_> = iter.into_iter().collect();

        if inputs.is_empty() {
            return BTreeSet::new();
        }

        // use stable sort to preserve the insertion order.
        inputs.sort();
        let iter = inputs.into_iter().map(|k| (k, ()));
        let map = BTreeMap::bulk_build_from_sorted_iter(iter);
        BTreeSet { map }
    }
}

#[stable(feature = "std_collections_from_array", since = "1.56.0")]
impl<T: Ord, const N: usize> From<[T; N]> for BTreeSet<T> {
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let set1 = BTreeSet::from([1, 2, 3, 4]);
    /// let set2: BTreeSet<_> = [1, 2, 3, 4].into();
    /// assert_eq!(set1, set2);
    /// ```
    fn from(mut arr: [T; N]) -> Self {
        if N == 0 {
            return BTreeSet::new();
        }

        // use stable sort to preserve the insertion order.
        arr.sort();
        let iter = core::array::IntoIter::new(arr).map(|k| (k, ()));
        let map = BTreeMap::bulk_build_from_sorted_iter(iter);
        BTreeSet { map }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> IntoIterator for BTreeSet<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    /// Gets an iterator for moving out the `BTreeSet`'s contents.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let set: BTreeSet<usize> = [1, 2, 3, 4].iter().cloned().collect();
    ///
    /// let v: Vec<_> = set.into_iter().collect();
    /// assert_eq!(v, [1, 2, 3, 4]);
    /// ```
    fn into_iter(self) -> IntoIter<T> {
        IntoIter { iter: self.map.into_iter() }
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

/// An iterator produced by calling `drain_filter` on BTreeSet.
#[unstable(feature = "btree_drain_filter", issue = "70530")]
pub struct DrainFilter<'a, T, F>
where
    T: 'a,
    F: 'a + FnMut(&T) -> bool,
{
    pred: F,
    inner: super::map::DrainFilterInner<'a, T, ()>,
}

#[unstable(feature = "btree_drain_filter", issue = "70530")]
impl<T, F> Drop for DrainFilter<'_, T, F>
where
    F: FnMut(&T) -> bool,
{
    fn drop(&mut self) {
        self.for_each(drop);
    }
}

#[unstable(feature = "btree_drain_filter", issue = "70530")]
impl<T, F> fmt::Debug for DrainFilter<'_, T, F>
where
    T: fmt::Debug,
    F: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DrainFilter").field(&self.inner.peek().map(|(k, _)| k)).finish()
    }
}

#[unstable(feature = "btree_drain_filter", issue = "70530")]
impl<'a, T, F> Iterator for DrainFilter<'_, T, F>
where
    F: 'a + FnMut(&T) -> bool,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let pred = &mut self.pred;
        let mut mapped_pred = |k: &T, _v: &mut ()| pred(k);
        self.inner.next(&mut mapped_pred).map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[unstable(feature = "btree_drain_filter", issue = "70530")]
impl<T, F> FusedIterator for DrainFilter<'_, T, F> where F: FnMut(&T) -> bool {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> Extend<T> for BTreeSet<T> {
    #[inline]
    fn extend<Iter: IntoIterator<Item = T>>(&mut self, iter: Iter) {
        iter.into_iter().for_each(move |elem| {
            self.insert(elem);
        });
    }

    #[inline]
    fn extend_one(&mut self, elem: T) {
        self.insert(elem);
    }
}

#[stable(feature = "extend_ref", since = "1.2.0")]
impl<'a, T: 'a + Ord + Copy> Extend<&'a T> for BTreeSet<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }

    #[inline]
    fn extend_one(&mut self, &elem: &'a T) {
        self.insert(elem);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for BTreeSet<T> {
    /// Creates an empty `BTreeSet`.
    fn default() -> BTreeSet<T> {
        BTreeSet::new()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord + Clone> Sub<&BTreeSet<T>> for &BTreeSet<T> {
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
    /// assert_eq!(result_vec, [1, 2]);
    /// ```
    fn sub(self, rhs: &BTreeSet<T>) -> BTreeSet<T> {
        self.difference(rhs).cloned().collect()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord + Clone> BitXor<&BTreeSet<T>> for &BTreeSet<T> {
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
    /// assert_eq!(result_vec, [1, 4]);
    /// ```
    fn bitxor(self, rhs: &BTreeSet<T>) -> BTreeSet<T> {
        self.symmetric_difference(rhs).cloned().collect()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord + Clone> BitAnd<&BTreeSet<T>> for &BTreeSet<T> {
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
    /// assert_eq!(result_vec, [2, 3]);
    /// ```
    fn bitand(self, rhs: &BTreeSet<T>) -> BTreeSet<T> {
        self.intersection(rhs).cloned().collect()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord + Clone> BitOr<&BTreeSet<T>> for &BTreeSet<T> {
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
    /// assert_eq!(result_vec, [1, 2, 3, 4, 5]);
    /// ```
    fn bitor(self, rhs: &BTreeSet<T>) -> BTreeSet<T> {
        self.union(rhs).cloned().collect()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Debug> Debug for BTreeSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Iter { iter: self.iter.clone() }
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn last(mut self) -> Option<&'a T> {
        self.next_back()
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }

    fn max(mut self) -> Option<&'a T> {
        self.next_back()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<&'a T> {
        self.iter.next_back()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for Iter<'_, T> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for Iter<'_, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.iter.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back().map(|(k, _)| k)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for IntoIter<T> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for IntoIter<T> {}

#[stable(feature = "btree_range", since = "1.17.0")]
impl<T> Clone for Range<'_, T> {
    fn clone(&self) -> Self {
        Range { iter: self.iter.clone() }
    }
}

#[stable(feature = "btree_range", since = "1.17.0")]
impl<'a, T> Iterator for Range<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.iter.next().map(|(k, _)| k)
    }

    fn last(mut self) -> Option<&'a T> {
        self.next_back()
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }

    fn max(mut self) -> Option<&'a T> {
        self.next_back()
    }
}

#[stable(feature = "btree_range", since = "1.17.0")]
impl<'a, T> DoubleEndedIterator for Range<'a, T> {
    fn next_back(&mut self) -> Option<&'a T> {
        self.iter.next_back().map(|(k, _)| k)
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for Range<'_, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Difference<'_, T> {
    fn clone(&self) -> Self {
        Difference {
            inner: match &self.inner {
                DifferenceInner::Stitch { self_iter, other_iter } => DifferenceInner::Stitch {
                    self_iter: self_iter.clone(),
                    other_iter: other_iter.clone(),
                },
                DifferenceInner::Search { self_iter, other_set } => {
                    DifferenceInner::Search { self_iter: self_iter.clone(), other_set }
                }
                DifferenceInner::Iterate(iter) => DifferenceInner::Iterate(iter.clone()),
            },
        }
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: Ord> Iterator for Difference<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        match &mut self.inner {
            DifferenceInner::Stitch { self_iter, other_iter } => {
                let mut self_next = self_iter.next()?;
                loop {
                    match other_iter.peek().map_or(Less, |other_next| self_next.cmp(other_next)) {
                        Less => return Some(self_next),
                        Equal => {
                            self_next = self_iter.next()?;
                            other_iter.next();
                        }
                        Greater => {
                            other_iter.next();
                        }
                    }
                }
            }
            DifferenceInner::Search { self_iter, other_set } => loop {
                let self_next = self_iter.next()?;
                if !other_set.contains(&self_next) {
                    return Some(self_next);
                }
            },
            DifferenceInner::Iterate(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (self_len, other_len) = match &self.inner {
            DifferenceInner::Stitch { self_iter, other_iter } => {
                (self_iter.len(), other_iter.len())
            }
            DifferenceInner::Search { self_iter, other_set } => (self_iter.len(), other_set.len()),
            DifferenceInner::Iterate(iter) => (iter.len(), 0),
        };
        (self_len.saturating_sub(other_len), Some(self_len))
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T: Ord> FusedIterator for Difference<'_, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for SymmetricDifference<'_, T> {
    fn clone(&self) -> Self {
        SymmetricDifference(self.0.clone())
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: Ord> Iterator for SymmetricDifference<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        loop {
            let (a_next, b_next) = self.0.nexts(Self::Item::cmp);
            if a_next.and(b_next).is_none() {
                return a_next.or(b_next);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_len, b_len) = self.0.lens();
        // No checked_add, because even if a and b refer to the same set,
        // and T is an empty type, the storage overhead of sets limits
        // the number of elements to less than half the range of usize.
        (0, Some(a_len + b_len))
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T: Ord> FusedIterator for SymmetricDifference<'_, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Intersection<'_, T> {
    fn clone(&self) -> Self {
        Intersection {
            inner: match &self.inner {
                IntersectionInner::Stitch { a, b } => {
                    IntersectionInner::Stitch { a: a.clone(), b: b.clone() }
                }
                IntersectionInner::Search { small_iter, large_set } => {
                    IntersectionInner::Search { small_iter: small_iter.clone(), large_set }
                }
                IntersectionInner::Answer(answer) => IntersectionInner::Answer(*answer),
            },
        }
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: Ord> Iterator for Intersection<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        match &mut self.inner {
            IntersectionInner::Stitch { a, b } => {
                let mut a_next = a.next()?;
                let mut b_next = b.next()?;
                loop {
                    match a_next.cmp(b_next) {
                        Less => a_next = a.next()?,
                        Greater => b_next = b.next()?,
                        Equal => return Some(a_next),
                    }
                }
            }
            IntersectionInner::Search { small_iter, large_set } => loop {
                let small_next = small_iter.next()?;
                if large_set.contains(&small_next) {
                    return Some(small_next);
                }
            },
            IntersectionInner::Answer(answer) => answer.take(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            IntersectionInner::Stitch { a, b } => (0, Some(min(a.len(), b.len()))),
            IntersectionInner::Search { small_iter, .. } => (0, Some(small_iter.len())),
            IntersectionInner::Answer(None) => (0, Some(0)),
            IntersectionInner::Answer(Some(_)) => (1, Some(1)),
        }
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T: Ord> FusedIterator for Intersection<'_, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Union<'_, T> {
    fn clone(&self) -> Self {
        Union(self.0.clone())
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: Ord> Iterator for Union<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let (a_next, b_next) = self.0.nexts(Self::Item::cmp);
        a_next.or(b_next)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_len, b_len) = self.0.lens();
        // No checked_add - see SymmetricDifference::size_hint.
        (max(a_len, b_len), Some(a_len + b_len))
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T: Ord> FusedIterator for Union<'_, T> {}

#[cfg(test)]
mod tests;
