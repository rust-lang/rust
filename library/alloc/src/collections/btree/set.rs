use core::borrow::Borrow;
use core::cmp::Ordering::{self, Equal, Greater, Less};
use core::cmp::{max, min};
use core::fmt::{self, Debug};
use core::hash::{Hash, Hasher};
use core::iter::{FusedIterator, Peekable};
use core::mem::ManuallyDrop;
use core::ops::{BitAnd, BitOr, BitXor, Bound, RangeBounds, Sub};

use super::map::{self, BTreeMap, Keys};
use super::merge_iter::MergeIterInner;
use super::set_val::SetValZST;
use crate::alloc::{Allocator, Global};
use crate::vec::Vec;

mod entry;

#[unstable(feature = "btree_set_entry", issue = "133549")]
pub use self::entry::{Entry, OccupiedEntry, VacantEntry};

/// An ordered set based on a B-Tree.
///
/// See [`BTreeMap`]'s documentation for a detailed discussion of this collection's performance
/// benefits and drawbacks.
///
/// It is a logic error for an item to be modified in such a way that the item's ordering relative
/// to any other item, as determined by the [`Ord`] trait, changes while it is in the set. This is
/// normally only possible through [`Cell`], [`RefCell`], global state, I/O, or unsafe code.
/// The behavior resulting from such a logic error is not specified, but will be encapsulated to the
/// `BTreeSet` that observed the logic error and not result in undefined behavior. This could
/// include panics, incorrect results, aborts, memory leaks, and non-termination.
///
/// Iterators returned by [`BTreeSet::iter`] and [`BTreeSet::into_iter`] produce their items in order, and take worst-case
/// logarithmic and amortized constant time per item returned.
///
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
///     println!("{book}");
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
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "BTreeSet")]
pub struct BTreeSet<
    T,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator + Clone = Global,
> {
    map: BTreeMap<T, SetValZST, A>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Hash, A: Allocator + Clone> Hash for BTreeSet<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.map.hash(state)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: PartialEq, A: Allocator + Clone> PartialEq for BTreeSet<T, A> {
    fn eq(&self, other: &BTreeSet<T, A>) -> bool {
        self.map.eq(&other.map)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Eq, A: Allocator + Clone> Eq for BTreeSet<T, A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: PartialOrd, A: Allocator + Clone> PartialOrd for BTreeSet<T, A> {
    fn partial_cmp(&self, other: &BTreeSet<T, A>) -> Option<Ordering> {
        self.map.partial_cmp(&other.map)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord, A: Allocator + Clone> Ord for BTreeSet<T, A> {
    fn cmp(&self, other: &BTreeSet<T, A>) -> Ordering {
        self.map.cmp(&other.map)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Clone, A: Allocator + Clone> Clone for BTreeSet<T, A> {
    fn clone(&self) -> Self {
        BTreeSet { map: self.map.clone() }
    }

    fn clone_from(&mut self, source: &Self) {
        self.map.clone_from(&source.map);
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
    iter: Keys<'a, T, SetValZST>,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Iter").field(&self.iter).finish()
    }
}

/// An owning iterator over the items of a `BTreeSet` in ascending order.
///
/// This `struct` is created by the [`into_iter`] method on [`BTreeSet`]
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
///
/// [`into_iter`]: BTreeSet#method.into_iter
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
pub struct IntoIter<
    T,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator + Clone = Global,
> {
    iter: super::map::IntoIter<T, SetValZST, A>,
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
    iter: super::map::Range<'a, T, SetValZST>,
}

/// A lazy iterator producing elements in the difference of `BTreeSet`s.
///
/// This `struct` is created by the [`difference`] method on [`BTreeSet`].
/// See its documentation for more.
///
/// [`difference`]: BTreeSet::difference
#[must_use = "this returns the difference as an iterator, \
              without modifying either input set"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Difference<
    'a,
    T: 'a,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator + Clone = Global,
> {
    inner: DifferenceInner<'a, T, A>,
}
enum DifferenceInner<'a, T: 'a, A: Allocator + Clone> {
    Stitch {
        // iterate all of `self` and some of `other`, spotting matches along the way
        self_iter: Iter<'a, T>,
        other_iter: Peekable<Iter<'a, T>>,
    },
    Search {
        // iterate `self`, look up in `other`
        self_iter: Iter<'a, T>,
        other_set: &'a BTreeSet<T, A>,
    },
    Iterate(Iter<'a, T>), // simply produce all elements in `self`
}

// Explicit Debug impl necessary because of issue #26925
impl<T: Debug, A: Allocator + Clone> Debug for DifferenceInner<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DifferenceInner::Stitch { self_iter, other_iter } => f
                .debug_struct("Stitch")
                .field("self_iter", self_iter)
                .field("other_iter", other_iter)
                .finish(),
            DifferenceInner::Search { self_iter, other_set } => f
                .debug_struct("Search")
                .field("self_iter", self_iter)
                .field("other_iter", other_set)
                .finish(),
            DifferenceInner::Iterate(x) => f.debug_tuple("Iterate").field(x).finish(),
        }
    }
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug, A: Allocator + Clone> fmt::Debug for Difference<'_, T, A> {
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
#[must_use = "this returns the difference as an iterator, \
              without modifying either input set"]
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
#[must_use = "this returns the intersection as an iterator, \
              without modifying either input set"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Intersection<
    'a,
    T: 'a,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator + Clone = Global,
> {
    inner: IntersectionInner<'a, T, A>,
}
enum IntersectionInner<'a, T: 'a, A: Allocator + Clone> {
    Stitch {
        // iterate similarly sized sets jointly, spotting matches along the way
        a: Iter<'a, T>,
        b: Iter<'a, T>,
    },
    Search {
        // iterate a small set, look up in the large set
        small_iter: Iter<'a, T>,
        large_set: &'a BTreeSet<T, A>,
    },
    Answer(Option<&'a T>), // return a specific element or emptiness
}

// Explicit Debug impl necessary because of issue #26925
impl<T: Debug, A: Allocator + Clone> Debug for IntersectionInner<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntersectionInner::Stitch { a, b } => {
                f.debug_struct("Stitch").field("a", a).field("b", b).finish()
            }
            IntersectionInner::Search { small_iter, large_set } => f
                .debug_struct("Search")
                .field("small_iter", small_iter)
                .field("large_set", large_set)
                .finish(),
            IntersectionInner::Answer(x) => f.debug_tuple("Answer").field(x).finish(),
        }
    }
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: Debug, A: Allocator + Clone> Debug for Intersection<'_, T, A> {
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
#[must_use = "this returns the union as an iterator, \
              without modifying either input set"]
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
    #[rustc_const_stable(feature = "const_btree_new", since = "1.66.0")]
    #[must_use]
    pub const fn new() -> BTreeSet<T> {
        BTreeSet { map: BTreeMap::new() }
    }
}

impl<T, A: Allocator + Clone> BTreeSet<T, A> {
    /// Makes a new `BTreeSet` with a reasonable choice of B.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(unused_mut)]
    /// # #![feature(allocator_api)]
    /// # #![feature(btreemap_alloc)]
    /// use std::collections::BTreeSet;
    /// use std::alloc::Global;
    ///
    /// let mut set: BTreeSet<i32> = BTreeSet::new_in(Global);
    /// ```
    #[unstable(feature = "btreemap_alloc", issue = "32838")]
    pub const fn new_in(alloc: A) -> BTreeSet<T, A> {
        BTreeSet { map: BTreeMap::new_in(alloc) }
    }

    /// Constructs a double-ended iterator over a sub-range of elements in the set.
    /// The simplest way is to use the range syntax `min..max`, thus `range(min..max)` will
    /// yield elements from min (inclusive) to max (exclusive).
    /// The range may also be entered as `(Bound<T>, Bound<T>)`, so for example
    /// `range((Excluded(4), Included(10)))` will yield a left-exclusive, right-inclusive
    /// range from 4 to 10.
    ///
    /// # Panics
    ///
    /// Panics if range `start > end`.
    /// Panics if range `start == end` and both bounds are `Excluded`.
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
    ///     println!("{elem}");
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

    /// Visits the elements representing the difference,
    /// i.e., the elements that are in `self` but not in `other`,
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
    pub fn difference<'a>(&'a self, other: &'a BTreeSet<T, A>) -> Difference<'a, T, A>
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

    /// Visits the elements representing the symmetric difference,
    /// i.e., the elements that are in `self` or in `other` but not in both,
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
    pub fn symmetric_difference<'a>(
        &'a self,
        other: &'a BTreeSet<T, A>,
    ) -> SymmetricDifference<'a, T>
    where
        T: Ord,
    {
        SymmetricDifference(MergeIterInner::new(self.iter(), other.iter()))
    }

    /// Visits the elements representing the intersection,
    /// i.e., the elements that are both in `self` and `other`,
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
    pub fn intersection<'a>(&'a self, other: &'a BTreeSet<T, A>) -> Intersection<'a, T, A>
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

    /// Visits the elements representing the union,
    /// i.e., all the elements in `self` or `other`, without duplicates,
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
    pub fn union<'a>(&'a self, other: &'a BTreeSet<T, A>) -> Union<'a, T>
    where
        T: Ord,
    {
        Union(MergeIterInner::new(self.iter(), other.iter()))
    }

    /// Clears the set, removing all elements.
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
    pub fn clear(&mut self)
    where
        A: Clone,
    {
        self.map.clear()
    }

    /// Returns `true` if the set contains an element equal to the value.
    ///
    /// The value may be any borrowed form of the set's element type,
    /// but the ordering on the borrowed form *must* match the
    /// ordering on the element type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let set = BTreeSet::from([1, 2, 3]);
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

    /// Returns a reference to the element in the set, if any, that is equal to
    /// the value.
    ///
    /// The value may be any borrowed form of the set's element type,
    /// but the ordering on the borrowed form *must* match the
    /// ordering on the element type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let set = BTreeSet::from([1, 2, 3]);
    /// assert_eq!(set.get(&2), Some(&2));
    /// assert_eq!(set.get(&4), None);
    /// ```
    #[stable(feature = "set_recovery", since = "1.9.0")]
    pub fn get<Q: ?Sized>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q> + Ord,
        Q: Ord,
    {
        self.map.get_key_value(value).map(|(k, _)| k)
    }

    /// Returns `true` if `self` has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let a = BTreeSet::from([1, 2, 3]);
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
    pub fn is_disjoint(&self, other: &BTreeSet<T, A>) -> bool
    where
        T: Ord,
    {
        self.intersection(other).next().is_none()
    }

    /// Returns `true` if the set is a subset of another,
    /// i.e., `other` contains at least all the elements in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let sup = BTreeSet::from([1, 2, 3]);
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
    pub fn is_subset(&self, other: &BTreeSet<T, A>) -> bool
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
    /// i.e., `self` contains at least all the elements in `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let sub = BTreeSet::from([1, 2]);
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
    pub fn is_superset(&self, other: &BTreeSet<T, A>) -> bool
    where
        T: Ord,
    {
        other.is_subset(self)
    }

    /// Returns a reference to the first element in the set, if any.
    /// This element is always the minimum of all elements in the set.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
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
    #[stable(feature = "map_first_last", since = "1.66.0")]
    #[rustc_confusables("front")]
    pub fn first(&self) -> Option<&T>
    where
        T: Ord,
    {
        self.map.first_key_value().map(|(k, _)| k)
    }

    /// Returns a reference to the last element in the set, if any.
    /// This element is always the maximum of all elements in the set.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
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
    #[stable(feature = "map_first_last", since = "1.66.0")]
    #[rustc_confusables("back")]
    pub fn last(&self) -> Option<&T>
    where
        T: Ord,
    {
        self.map.last_key_value().map(|(k, _)| k)
    }

    /// Removes the first element from the set and returns it, if any.
    /// The first element is always the minimum element in the set.
    ///
    /// # Examples
    ///
    /// ```
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
    #[stable(feature = "map_first_last", since = "1.66.0")]
    pub fn pop_first(&mut self) -> Option<T>
    where
        T: Ord,
    {
        self.map.pop_first().map(|kv| kv.0)
    }

    /// Removes the last element from the set and returns it, if any.
    /// The last element is always the maximum element in the set.
    ///
    /// # Examples
    ///
    /// ```
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
    #[stable(feature = "map_first_last", since = "1.66.0")]
    pub fn pop_last(&mut self) -> Option<T>
    where
        T: Ord,
    {
        self.map.pop_last().map(|kv| kv.0)
    }

    /// Adds a value to the set.
    ///
    /// Returns whether the value was newly inserted. That is:
    ///
    /// - If the set did not previously contain an equal value, `true` is
    ///   returned.
    /// - If the set already contained an equal value, `false` is returned, and
    ///   the entry is not updated.
    ///
    /// See the [module-level documentation] for more.
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
    #[rustc_confusables("push", "put")]
    pub fn insert(&mut self, value: T) -> bool
    where
        T: Ord,
    {
        self.map.insert(value, SetValZST::default()).is_none()
    }

    /// Adds a value to the set, replacing the existing element, if any, that is
    /// equal to the value. Returns the replaced element.
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
    #[rustc_confusables("swap")]
    pub fn replace(&mut self, value: T) -> Option<T>
    where
        T: Ord,
    {
        self.map.replace(value)
    }

    /// Inserts the given `value` into the set if it is not present, then
    /// returns a reference to the value in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_set_entry)]
    ///
    /// use std::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::from([1, 2, 3]);
    /// assert_eq!(set.len(), 3);
    /// assert_eq!(set.get_or_insert(2), &2);
    /// assert_eq!(set.get_or_insert(100), &100);
    /// assert_eq!(set.len(), 4); // 100 was inserted
    /// ```
    #[inline]
    #[unstable(feature = "btree_set_entry", issue = "133549")]
    pub fn get_or_insert(&mut self, value: T) -> &T
    where
        T: Ord,
    {
        self.map.entry(value).insert_entry(SetValZST).into_key()
    }

    /// Inserts a value computed from `f` into the set if the given `value` is
    /// not present, then returns a reference to the value in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_set_entry)]
    ///
    /// use std::collections::BTreeSet;
    ///
    /// let mut set: BTreeSet<String> = ["cat", "dog", "horse"]
    ///     .iter().map(|&pet| pet.to_owned()).collect();
    ///
    /// assert_eq!(set.len(), 3);
    /// for &pet in &["cat", "dog", "fish"] {
    ///     let value = set.get_or_insert_with(pet, str::to_owned);
    ///     assert_eq!(value, pet);
    /// }
    /// assert_eq!(set.len(), 4); // a new "fish" was inserted
    /// ```
    #[inline]
    #[unstable(feature = "btree_set_entry", issue = "133549")]
    pub fn get_or_insert_with<Q: ?Sized, F>(&mut self, value: &Q, f: F) -> &T
    where
        T: Borrow<Q> + Ord,
        Q: Ord,
        F: FnOnce(&Q) -> T,
    {
        self.map.get_or_insert_with(value, f)
    }

    /// Gets the given value's corresponding entry in the set for in-place manipulation.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_set_entry)]
    ///
    /// use std::collections::BTreeSet;
    /// use std::collections::btree_set::Entry::*;
    ///
    /// let mut singles = BTreeSet::new();
    /// let mut dupes = BTreeSet::new();
    ///
    /// for ch in "a short treatise on fungi".chars() {
    ///     if let Vacant(dupe_entry) = dupes.entry(ch) {
    ///         // We haven't already seen a duplicate, so
    ///         // check if we've at least seen it once.
    ///         match singles.entry(ch) {
    ///             Vacant(single_entry) => {
    ///                 // We found a new character for the first time.
    ///                 single_entry.insert()
    ///             }
    ///             Occupied(single_entry) => {
    ///                 // We've already seen this once, "move" it to dupes.
    ///                 single_entry.remove();
    ///                 dupe_entry.insert();
    ///             }
    ///         }
    ///     }
    /// }
    ///
    /// assert!(!singles.contains(&'t') && dupes.contains(&'t'));
    /// assert!(singles.contains(&'u') && !dupes.contains(&'u'));
    /// assert!(!singles.contains(&'v') && !dupes.contains(&'v'));
    /// ```
    #[inline]
    #[unstable(feature = "btree_set_entry", issue = "133549")]
    pub fn entry(&mut self, value: T) -> Entry<'_, T, A>
    where
        T: Ord,
    {
        match self.map.entry(value) {
            map::Entry::Occupied(entry) => Entry::Occupied(OccupiedEntry { inner: entry }),
            map::Entry::Vacant(entry) => Entry::Vacant(VacantEntry { inner: entry }),
        }
    }

    /// If the set contains an element equal to the value, removes it from the
    /// set and drops it. Returns whether such an element was present.
    ///
    /// The value may be any borrowed form of the set's element type,
    /// but the ordering on the borrowed form *must* match the
    /// ordering on the element type.
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

    /// Removes and returns the element in the set, if any, that is equal to
    /// the value.
    ///
    /// The value may be any borrowed form of the set's element type,
    /// but the ordering on the borrowed form *must* match the
    /// ordering on the element type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::from([1, 2, 3]);
    /// assert_eq!(set.take(&2), Some(2));
    /// assert_eq!(set.take(&2), None);
    /// ```
    #[stable(feature = "set_recovery", since = "1.9.0")]
    pub fn take<Q: ?Sized>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q> + Ord,
        Q: Ord,
    {
        self.map.remove_entry(value).map(|(k, _)| k)
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns `false`.
    /// The elements are visited in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::from([1, 2, 3, 4, 5, 6]);
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
        self.extract_if(.., |v| !f(v)).for_each(drop);
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
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
        A: Clone,
    {
        self.map.append(&mut other.map);
    }

    /// Splits the collection into two at the value. Returns a new collection
    /// with all elements greater than or equal to the value.
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
        A: Clone,
    {
        BTreeSet { map: self.map.split_off(value) }
    }

    /// Creates an iterator that visits elements in the specified range in ascending order and
    /// uses a closure to determine if an element should be removed.
    ///
    /// If the closure returns `true`, the element is removed from the set and
    /// yielded. If the closure returns `false`, or panics, the element remains
    /// in the set and will not be yielded.
    ///
    /// If the returned `ExtractIf` is not exhausted, e.g. because it is dropped without iterating
    /// or the iteration short-circuits, then the remaining elements will be retained.
    /// Use [`retain`] with a negated predicate if you do not need the returned iterator.
    ///
    /// [`retain`]: BTreeSet::retain
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_extract_if)]
    /// use std::collections::BTreeSet;
    ///
    /// // Splitting a set into even and odd values, reusing the original set:
    /// let mut set: BTreeSet<i32> = (0..8).collect();
    /// let evens: BTreeSet<_> = set.extract_if(.., |v| v % 2 == 0).collect();
    /// let odds = set;
    /// assert_eq!(evens.into_iter().collect::<Vec<_>>(), vec![0, 2, 4, 6]);
    /// assert_eq!(odds.into_iter().collect::<Vec<_>>(), vec![1, 3, 5, 7]);
    ///
    /// // Splitting a set into low and high halves, reusing the original set:
    /// let mut set: BTreeSet<i32> = (0..8).collect();
    /// let low: BTreeSet<_> = set.extract_if(0..4, |_v| true).collect();
    /// let high = set;
    /// assert_eq!(low.into_iter().collect::<Vec<_>>(), [0, 1, 2, 3]);
    /// assert_eq!(high.into_iter().collect::<Vec<_>>(), [4, 5, 6, 7]);
    /// ```
    #[unstable(feature = "btree_extract_if", issue = "70530")]
    pub fn extract_if<'a, F, R>(&'a mut self, range: R, pred: F) -> ExtractIf<'a, T, R, F, A>
    where
        T: Ord,
        R: RangeBounds<T>,
        F: 'a + FnMut(&T) -> bool,
    {
        let (inner, alloc) = self.map.extract_if_inner(range);
        ExtractIf { pred, inner, alloc }
    }

    /// Gets an iterator that visits the elements in the `BTreeSet` in ascending
    /// order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let set = BTreeSet::from([3, 1, 2]);
    /// let mut set_iter = set.iter();
    /// assert_eq!(set_iter.next(), Some(&1));
    /// assert_eq!(set_iter.next(), Some(&2));
    /// assert_eq!(set_iter.next(), Some(&3));
    /// assert_eq!(set_iter.next(), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[cfg_attr(not(test), rustc_diagnostic_item = "btreeset_iter")]
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(
        feature = "const_btree_len",
        issue = "71835",
        implied_by = "const_btree_new"
    )]
    #[rustc_confusables("length", "size")]
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(
        feature = "const_btree_len",
        issue = "71835",
        implied_by = "const_btree_new"
    )]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a [`Cursor`] pointing at the gap before the smallest element
    /// greater than the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap before the smallest element greater than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap before the smallest element greater than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap before the smallest element in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_cursors)]
    ///
    /// use std::collections::BTreeSet;
    /// use std::ops::Bound;
    ///
    /// let set = BTreeSet::from([1, 2, 3, 4]);
    ///
    /// let cursor = set.lower_bound(Bound::Included(&2));
    /// assert_eq!(cursor.peek_prev(), Some(&1));
    /// assert_eq!(cursor.peek_next(), Some(&2));
    ///
    /// let cursor = set.lower_bound(Bound::Excluded(&2));
    /// assert_eq!(cursor.peek_prev(), Some(&2));
    /// assert_eq!(cursor.peek_next(), Some(&3));
    ///
    /// let cursor = set.lower_bound(Bound::Unbounded);
    /// assert_eq!(cursor.peek_prev(), None);
    /// assert_eq!(cursor.peek_next(), Some(&1));
    /// ```
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn lower_bound<Q: ?Sized>(&self, bound: Bound<&Q>) -> Cursor<'_, T>
    where
        T: Borrow<Q> + Ord,
        Q: Ord,
    {
        Cursor { inner: self.map.lower_bound(bound) }
    }

    /// Returns a [`CursorMut`] pointing at the gap before the smallest element
    /// greater than the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap before the smallest element greater than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap before the smallest element greater than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap before the smallest element in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_cursors)]
    ///
    /// use std::collections::BTreeSet;
    /// use std::ops::Bound;
    ///
    /// let mut set = BTreeSet::from([1, 2, 3, 4]);
    ///
    /// let mut cursor = set.lower_bound_mut(Bound::Included(&2));
    /// assert_eq!(cursor.peek_prev(), Some(&1));
    /// assert_eq!(cursor.peek_next(), Some(&2));
    ///
    /// let mut cursor = set.lower_bound_mut(Bound::Excluded(&2));
    /// assert_eq!(cursor.peek_prev(), Some(&2));
    /// assert_eq!(cursor.peek_next(), Some(&3));
    ///
    /// let mut cursor = set.lower_bound_mut(Bound::Unbounded);
    /// assert_eq!(cursor.peek_prev(), None);
    /// assert_eq!(cursor.peek_next(), Some(&1));
    /// ```
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn lower_bound_mut<Q: ?Sized>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, T, A>
    where
        T: Borrow<Q> + Ord,
        Q: Ord,
    {
        CursorMut { inner: self.map.lower_bound_mut(bound) }
    }

    /// Returns a [`Cursor`] pointing at the gap after the greatest element
    /// smaller than the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap after the greatest element smaller than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap after the greatest element smaller than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap after the greatest element in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_cursors)]
    ///
    /// use std::collections::BTreeSet;
    /// use std::ops::Bound;
    ///
    /// let set = BTreeSet::from([1, 2, 3, 4]);
    ///
    /// let cursor = set.upper_bound(Bound::Included(&3));
    /// assert_eq!(cursor.peek_prev(), Some(&3));
    /// assert_eq!(cursor.peek_next(), Some(&4));
    ///
    /// let cursor = set.upper_bound(Bound::Excluded(&3));
    /// assert_eq!(cursor.peek_prev(), Some(&2));
    /// assert_eq!(cursor.peek_next(), Some(&3));
    ///
    /// let cursor = set.upper_bound(Bound::Unbounded);
    /// assert_eq!(cursor.peek_prev(), Some(&4));
    /// assert_eq!(cursor.peek_next(), None);
    /// ```
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn upper_bound<Q: ?Sized>(&self, bound: Bound<&Q>) -> Cursor<'_, T>
    where
        T: Borrow<Q> + Ord,
        Q: Ord,
    {
        Cursor { inner: self.map.upper_bound(bound) }
    }

    /// Returns a [`CursorMut`] pointing at the gap after the greatest element
    /// smaller than the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap after the greatest element smaller than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap after the greatest element smaller than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap after the greatest element in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_cursors)]
    ///
    /// use std::collections::BTreeSet;
    /// use std::ops::Bound;
    ///
    /// let mut set = BTreeSet::from([1, 2, 3, 4]);
    ///
    /// let mut cursor = set.upper_bound_mut(Bound::Included(&3));
    /// assert_eq!(cursor.peek_prev(), Some(&3));
    /// assert_eq!(cursor.peek_next(), Some(&4));
    ///
    /// let mut cursor = set.upper_bound_mut(Bound::Excluded(&3));
    /// assert_eq!(cursor.peek_prev(), Some(&2));
    /// assert_eq!(cursor.peek_next(), Some(&3));
    ///
    /// let mut cursor = set.upper_bound_mut(Bound::Unbounded);
    /// assert_eq!(cursor.peek_prev(), Some(&4));
    /// assert_eq!(cursor.peek_next(), None);
    /// ```
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn upper_bound_mut<Q: ?Sized>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, T, A>
    where
        T: Borrow<Q> + Ord,
        Q: Ord,
    {
        CursorMut { inner: self.map.upper_bound_mut(bound) }
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
        BTreeSet::from_sorted_iter(inputs.into_iter(), Global)
    }
}

impl<T: Ord, A: Allocator + Clone> BTreeSet<T, A> {
    fn from_sorted_iter<I: Iterator<Item = T>>(iter: I, alloc: A) -> BTreeSet<T, A> {
        let iter = iter.map(|k| (k, SetValZST::default()));
        let map = BTreeMap::bulk_build_from_sorted_iter(iter, alloc);
        BTreeSet { map }
    }
}

#[stable(feature = "std_collections_from_array", since = "1.56.0")]
impl<T: Ord, const N: usize> From<[T; N]> for BTreeSet<T> {
    /// Converts a `[T; N]` into a `BTreeSet<T>`.
    ///
    /// If the array contains any equal values,
    /// all but one will be dropped.
    ///
    /// # Examples
    ///
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
        let iter = IntoIterator::into_iter(arr).map(|k| (k, SetValZST::default()));
        let map = BTreeMap::bulk_build_from_sorted_iter(iter, Global);
        BTreeSet { map }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, A: Allocator + Clone> IntoIterator for BTreeSet<T, A> {
    type Item = T;
    type IntoIter = IntoIter<T, A>;

    /// Gets an iterator for moving out the `BTreeSet`'s contents in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let set = BTreeSet::from([1, 2, 3, 4]);
    ///
    /// let v: Vec<_> = set.into_iter().collect();
    /// assert_eq!(v, [1, 2, 3, 4]);
    /// ```
    fn into_iter(self) -> IntoIter<T, A> {
        IntoIter { iter: self.map.into_iter() }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, A: Allocator + Clone> IntoIterator for &'a BTreeSet<T, A> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

/// An iterator produced by calling `extract_if` on BTreeSet.
#[unstable(feature = "btree_extract_if", issue = "70530")]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ExtractIf<
    'a,
    T,
    R,
    F,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator + Clone = Global,
> {
    pred: F,
    inner: super::map::ExtractIfInner<'a, T, SetValZST, R>,
    /// The BTreeMap will outlive this IntoIter so we don't care about drop order for `alloc`.
    alloc: A,
}

#[unstable(feature = "btree_extract_if", issue = "70530")]
impl<T, R, F, A> fmt::Debug for ExtractIf<'_, T, R, F, A>
where
    T: fmt::Debug,
    A: Allocator + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtractIf")
            .field("peek", &self.inner.peek().map(|(k, _)| k))
            .finish_non_exhaustive()
    }
}

#[unstable(feature = "btree_extract_if", issue = "70530")]
impl<'a, T, R, F, A: Allocator + Clone> Iterator for ExtractIf<'_, T, R, F, A>
where
    T: PartialOrd,
    R: RangeBounds<T>,
    F: 'a + FnMut(&T) -> bool,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let pred = &mut self.pred;
        let mut mapped_pred = |k: &T, _v: &mut SetValZST| pred(k);
        self.inner.next(&mut mapped_pred, self.alloc.clone()).map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[unstable(feature = "btree_extract_if", issue = "70530")]
impl<T, R, F, A: Allocator + Clone> FusedIterator for ExtractIf<'_, T, R, F, A>
where
    T: PartialOrd,
    R: RangeBounds<T>,
    F: FnMut(&T) -> bool,
{
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord, A: Allocator + Clone> Extend<T> for BTreeSet<T, A> {
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
impl<'a, T: 'a + Ord + Copy, A: Allocator + Clone> Extend<&'a T> for BTreeSet<T, A> {
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
impl<T: Ord + Clone, A: Allocator + Clone> Sub<&BTreeSet<T, A>> for &BTreeSet<T, A> {
    type Output = BTreeSet<T, A>;

    /// Returns the difference of `self` and `rhs` as a new `BTreeSet<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let a = BTreeSet::from([1, 2, 3]);
    /// let b = BTreeSet::from([3, 4, 5]);
    ///
    /// let result = &a - &b;
    /// assert_eq!(result, BTreeSet::from([1, 2]));
    /// ```
    fn sub(self, rhs: &BTreeSet<T, A>) -> BTreeSet<T, A> {
        BTreeSet::from_sorted_iter(
            self.difference(rhs).cloned(),
            ManuallyDrop::into_inner(self.map.alloc.clone()),
        )
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord + Clone, A: Allocator + Clone> BitXor<&BTreeSet<T, A>> for &BTreeSet<T, A> {
    type Output = BTreeSet<T, A>;

    /// Returns the symmetric difference of `self` and `rhs` as a new `BTreeSet<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let a = BTreeSet::from([1, 2, 3]);
    /// let b = BTreeSet::from([2, 3, 4]);
    ///
    /// let result = &a ^ &b;
    /// assert_eq!(result, BTreeSet::from([1, 4]));
    /// ```
    fn bitxor(self, rhs: &BTreeSet<T, A>) -> BTreeSet<T, A> {
        BTreeSet::from_sorted_iter(
            self.symmetric_difference(rhs).cloned(),
            ManuallyDrop::into_inner(self.map.alloc.clone()),
        )
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord + Clone, A: Allocator + Clone> BitAnd<&BTreeSet<T, A>> for &BTreeSet<T, A> {
    type Output = BTreeSet<T, A>;

    /// Returns the intersection of `self` and `rhs` as a new `BTreeSet<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let a = BTreeSet::from([1, 2, 3]);
    /// let b = BTreeSet::from([2, 3, 4]);
    ///
    /// let result = &a & &b;
    /// assert_eq!(result, BTreeSet::from([2, 3]));
    /// ```
    fn bitand(self, rhs: &BTreeSet<T, A>) -> BTreeSet<T, A> {
        BTreeSet::from_sorted_iter(
            self.intersection(rhs).cloned(),
            ManuallyDrop::into_inner(self.map.alloc.clone()),
        )
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord + Clone, A: Allocator + Clone> BitOr<&BTreeSet<T, A>> for &BTreeSet<T, A> {
    type Output = BTreeSet<T, A>;

    /// Returns the union of `self` and `rhs` as a new `BTreeSet<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let a = BTreeSet::from([1, 2, 3]);
    /// let b = BTreeSet::from([3, 4, 5]);
    ///
    /// let result = &a | &b;
    /// assert_eq!(result, BTreeSet::from([1, 2, 3, 4, 5]));
    /// ```
    fn bitor(self, rhs: &BTreeSet<T, A>) -> BTreeSet<T, A> {
        BTreeSet::from_sorted_iter(
            self.union(rhs).cloned(),
            ManuallyDrop::into_inner(self.map.alloc.clone()),
        )
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Debug, A: Allocator + Clone> Debug for BTreeSet<T, A> {
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

    fn min(mut self) -> Option<&'a T>
    where
        &'a T: Ord,
    {
        self.next()
    }

    fn max(mut self) -> Option<&'a T>
    where
        &'a T: Ord,
    {
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
impl<T, A: Allocator + Clone> Iterator for IntoIter<T, A> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.iter.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[stable(feature = "default_iters", since = "1.70.0")]
impl<T> Default for Iter<'_, T> {
    /// Creates an empty `btree_set::Iter`.
    ///
    /// ```
    /// # use std::collections::btree_set;
    /// let iter: btree_set::Iter<'_, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        Iter { iter: Default::default() }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, A: Allocator + Clone> DoubleEndedIterator for IntoIter<T, A> {
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back().map(|(k, _)| k)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T, A: Allocator + Clone> ExactSizeIterator for IntoIter<T, A> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, A: Allocator + Clone> FusedIterator for IntoIter<T, A> {}

#[stable(feature = "default_iters", since = "1.70.0")]
impl<T, A> Default for IntoIter<T, A>
where
    A: Allocator + Default + Clone,
{
    /// Creates an empty `btree_set::IntoIter`.
    ///
    /// ```
    /// # use std::collections::btree_set;
    /// let iter: btree_set::IntoIter<u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        IntoIter { iter: Default::default() }
    }
}

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

    fn min(mut self) -> Option<&'a T>
    where
        &'a T: Ord,
    {
        self.next()
    }

    fn max(mut self) -> Option<&'a T>
    where
        &'a T: Ord,
    {
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

#[stable(feature = "default_iters", since = "1.70.0")]
impl<T> Default for Range<'_, T> {
    /// Creates an empty `btree_set::Range`.
    ///
    /// ```
    /// # use std::collections::btree_set;
    /// let iter: btree_set::Range<'_, u8> = Default::default();
    /// assert_eq!(iter.count(), 0);
    /// ```
    fn default() -> Self {
        Range { iter: Default::default() }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, A: Allocator + Clone> Clone for Difference<'_, T, A> {
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
impl<'a, T: Ord, A: Allocator + Clone> Iterator for Difference<'a, T, A> {
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
impl<T: Ord, A: Allocator + Clone> FusedIterator for Difference<'_, T, A> {}

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
        // and T is a zero-sized type, the storage overhead of sets limits
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
impl<T, A: Allocator + Clone> Clone for Intersection<'_, T, A> {
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
impl<'a, T: Ord, A: Allocator + Clone> Iterator for Intersection<'a, T, A> {
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
impl<T: Ord, A: Allocator + Clone> FusedIterator for Intersection<'_, T, A> {}

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

/// A cursor over a `BTreeSet`.
///
/// A `Cursor` is like an iterator, except that it can freely seek back-and-forth.
///
/// Cursors always point to a gap between two elements in the set, and can
/// operate on the two immediately adjacent elements.
///
/// A `Cursor` is created with the [`BTreeSet::lower_bound`] and [`BTreeSet::upper_bound`] methods.
#[derive(Clone)]
#[unstable(feature = "btree_cursors", issue = "107540")]
pub struct Cursor<'a, K: 'a> {
    inner: super::map::Cursor<'a, K, SetValZST>,
}

#[unstable(feature = "btree_cursors", issue = "107540")]
impl<K: Debug> Debug for Cursor<'_, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Cursor")
    }
}

/// A cursor over a `BTreeSet` with editing operations.
///
/// A `Cursor` is like an iterator, except that it can freely seek back-and-forth, and can
/// safely mutate the set during iteration. This is because the lifetime of its yielded
/// references is tied to its own lifetime, instead of just the underlying map. This means
/// cursors cannot yield multiple elements at once.
///
/// Cursors always point to a gap between two elements in the set, and can
/// operate on the two immediately adjacent elements.
///
/// A `CursorMut` is created with the [`BTreeSet::lower_bound_mut`] and [`BTreeSet::upper_bound_mut`]
/// methods.
#[unstable(feature = "btree_cursors", issue = "107540")]
pub struct CursorMut<'a, K: 'a, #[unstable(feature = "allocator_api", issue = "32838")] A = Global>
{
    inner: super::map::CursorMut<'a, K, SetValZST, A>,
}

#[unstable(feature = "btree_cursors", issue = "107540")]
impl<K: Debug, A> Debug for CursorMut<'_, K, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CursorMut")
    }
}

/// A cursor over a `BTreeSet` with editing operations, and which allows
/// mutating elements.
///
/// A `Cursor` is like an iterator, except that it can freely seek back-and-forth, and can
/// safely mutate the set during iteration. This is because the lifetime of its yielded
/// references is tied to its own lifetime, instead of just the underlying set. This means
/// cursors cannot yield multiple elements at once.
///
/// Cursors always point to a gap between two elements in the set, and can
/// operate on the two immediately adjacent elements.
///
/// A `CursorMutKey` is created from a [`CursorMut`] with the
/// [`CursorMut::with_mutable_key`] method.
///
/// # Safety
///
/// Since this cursor allows mutating elements, you must ensure that the
/// `BTreeSet` invariants are maintained. Specifically:
///
/// * The newly inserted element must be unique in the tree.
/// * All elements in the tree must remain in sorted order.
#[unstable(feature = "btree_cursors", issue = "107540")]
pub struct CursorMutKey<
    'a,
    K: 'a,
    #[unstable(feature = "allocator_api", issue = "32838")] A = Global,
> {
    inner: super::map::CursorMutKey<'a, K, SetValZST, A>,
}

#[unstable(feature = "btree_cursors", issue = "107540")]
impl<K: Debug, A> Debug for CursorMutKey<'_, K, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CursorMutKey")
    }
}

impl<'a, K> Cursor<'a, K> {
    /// Advances the cursor to the next gap, returning the element that it
    /// moved over.
    ///
    /// If the cursor is already at the end of the set then `None` is returned
    /// and the cursor is not moved.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn next(&mut self) -> Option<&'a K> {
        self.inner.next().map(|(k, _)| k)
    }

    /// Advances the cursor to the previous gap, returning the element that it
    /// moved over.
    ///
    /// If the cursor is already at the start of the set then `None` is returned
    /// and the cursor is not moved.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn prev(&mut self) -> Option<&'a K> {
        self.inner.prev().map(|(k, _)| k)
    }

    /// Returns a reference to next element without moving the cursor.
    ///
    /// If the cursor is at the end of the set then `None` is returned
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn peek_next(&self) -> Option<&'a K> {
        self.inner.peek_next().map(|(k, _)| k)
    }

    /// Returns a reference to the previous element without moving the cursor.
    ///
    /// If the cursor is at the start of the set then `None` is returned.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn peek_prev(&self) -> Option<&'a K> {
        self.inner.peek_prev().map(|(k, _)| k)
    }
}

impl<'a, T, A> CursorMut<'a, T, A> {
    /// Advances the cursor to the next gap, returning the element that it
    /// moved over.
    ///
    /// If the cursor is already at the end of the set then `None` is returned
    /// and the cursor is not moved.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn next(&mut self) -> Option<&T> {
        self.inner.next().map(|(k, _)| k)
    }

    /// Advances the cursor to the previous gap, returning the element that it
    /// moved over.
    ///
    /// If the cursor is already at the start of the set then `None` is returned
    /// and the cursor is not moved.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn prev(&mut self) -> Option<&T> {
        self.inner.prev().map(|(k, _)| k)
    }

    /// Returns a reference to the next element without moving the cursor.
    ///
    /// If the cursor is at the end of the set then `None` is returned.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn peek_next(&mut self) -> Option<&T> {
        self.inner.peek_next().map(|(k, _)| k)
    }

    /// Returns a reference to the previous element without moving the cursor.
    ///
    /// If the cursor is at the start of the set then `None` is returned.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn peek_prev(&mut self) -> Option<&T> {
        self.inner.peek_prev().map(|(k, _)| k)
    }

    /// Returns a read-only cursor pointing to the same location as the
    /// `CursorMut`.
    ///
    /// The lifetime of the returned `Cursor` is bound to that of the
    /// `CursorMut`, which means it cannot outlive the `CursorMut` and that the
    /// `CursorMut` is frozen for the lifetime of the `Cursor`.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn as_cursor(&self) -> Cursor<'_, T> {
        Cursor { inner: self.inner.as_cursor() }
    }

    /// Converts the cursor into a [`CursorMutKey`], which allows mutating
    /// elements in the tree.
    ///
    /// # Safety
    ///
    /// Since this cursor allows mutating elements, you must ensure that the
    /// `BTreeSet` invariants are maintained. Specifically:
    ///
    /// * The newly inserted element must be unique in the tree.
    /// * All elements in the tree must remain in sorted order.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub unsafe fn with_mutable_key(self) -> CursorMutKey<'a, T, A> {
        CursorMutKey { inner: unsafe { self.inner.with_mutable_key() } }
    }
}

impl<'a, T, A> CursorMutKey<'a, T, A> {
    /// Advances the cursor to the next gap, returning the  element that it
    /// moved over.
    ///
    /// If the cursor is already at the end of the set then `None` is returned
    /// and the cursor is not moved.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn next(&mut self) -> Option<&mut T> {
        self.inner.next().map(|(k, _)| k)
    }

    /// Advances the cursor to the previous gap, returning the element that it
    /// moved over.
    ///
    /// If the cursor is already at the start of the set then `None` is returned
    /// and the cursor is not moved.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn prev(&mut self) -> Option<&mut T> {
        self.inner.prev().map(|(k, _)| k)
    }

    /// Returns a reference to the next element without moving the cursor.
    ///
    /// If the cursor is at the end of the set then `None` is returned
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn peek_next(&mut self) -> Option<&mut T> {
        self.inner.peek_next().map(|(k, _)| k)
    }

    /// Returns a reference to the previous element without moving the cursor.
    ///
    /// If the cursor is at the start of the set then `None` is returned.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn peek_prev(&mut self) -> Option<&mut T> {
        self.inner.peek_prev().map(|(k, _)| k)
    }

    /// Returns a read-only cursor pointing to the same location as the
    /// `CursorMutKey`.
    ///
    /// The lifetime of the returned `Cursor` is bound to that of the
    /// `CursorMutKey`, which means it cannot outlive the `CursorMutKey` and that the
    /// `CursorMutKey` is frozen for the lifetime of the `Cursor`.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn as_cursor(&self) -> Cursor<'_, T> {
        Cursor { inner: self.inner.as_cursor() }
    }
}

impl<'a, T: Ord, A: Allocator + Clone> CursorMut<'a, T, A> {
    /// Inserts a new element into the set in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted element.
    ///
    /// # Safety
    ///
    /// You must ensure that the `BTreeSet` invariants are maintained.
    /// Specifically:
    ///
    /// * The newly inserted element must be unique in the tree.
    /// * All elements in the tree must remain in sorted order.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub unsafe fn insert_after_unchecked(&mut self, value: T) {
        unsafe { self.inner.insert_after_unchecked(value, SetValZST) }
    }

    /// Inserts a new element into the set in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted element.
    ///
    /// # Safety
    ///
    /// You must ensure that the `BTreeSet` invariants are maintained.
    /// Specifically:
    ///
    /// * The newly inserted element must be unique in the tree.
    /// * All elements in the tree must remain in sorted order.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub unsafe fn insert_before_unchecked(&mut self, value: T) {
        unsafe { self.inner.insert_before_unchecked(value, SetValZST) }
    }

    /// Inserts a new element into the set in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted element.
    ///
    /// If the inserted element is not greater than the element before the
    /// cursor (if any), or if it not less than the element after the cursor (if
    /// any), then an [`UnorderedKeyError`] is returned since this would
    /// invalidate the [`Ord`] invariant between the elements of the set.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn insert_after(&mut self, value: T) -> Result<(), UnorderedKeyError> {
        self.inner.insert_after(value, SetValZST)
    }

    /// Inserts a new element into the set in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted element.
    ///
    /// If the inserted element is not greater than the element before the
    /// cursor (if any), or if it not less than the element after the cursor (if
    /// any), then an [`UnorderedKeyError`] is returned since this would
    /// invalidate the [`Ord`] invariant between the elements of the set.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn insert_before(&mut self, value: T) -> Result<(), UnorderedKeyError> {
        self.inner.insert_before(value, SetValZST)
    }

    /// Removes the next element from the `BTreeSet`.
    ///
    /// The element that was removed is returned. The cursor position is
    /// unchanged (before the removed element).
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn remove_next(&mut self) -> Option<T> {
        self.inner.remove_next().map(|(k, _)| k)
    }

    /// Removes the preceding element from the `BTreeSet`.
    ///
    /// The element that was removed is returned. The cursor position is
    /// unchanged (after the removed element).
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn remove_prev(&mut self) -> Option<T> {
        self.inner.remove_prev().map(|(k, _)| k)
    }
}

impl<'a, T: Ord, A: Allocator + Clone> CursorMutKey<'a, T, A> {
    /// Inserts a new element into the set in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted element.
    ///
    /// # Safety
    ///
    /// You must ensure that the `BTreeSet` invariants are maintained.
    /// Specifically:
    ///
    /// * The key of the newly inserted element must be unique in the tree.
    /// * All elements in the tree must remain in sorted order.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub unsafe fn insert_after_unchecked(&mut self, value: T) {
        unsafe { self.inner.insert_after_unchecked(value, SetValZST) }
    }

    /// Inserts a new element into the set in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted element.
    ///
    /// # Safety
    ///
    /// You must ensure that the `BTreeSet` invariants are maintained.
    /// Specifically:
    ///
    /// * The newly inserted element must be unique in the tree.
    /// * All elements in the tree must remain in sorted order.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub unsafe fn insert_before_unchecked(&mut self, value: T) {
        unsafe { self.inner.insert_before_unchecked(value, SetValZST) }
    }

    /// Inserts a new element into the set in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted element.
    ///
    /// If the inserted element is not greater than the element before the
    /// cursor (if any), or if it not less than the element after the cursor (if
    /// any), then an [`UnorderedKeyError`] is returned since this would
    /// invalidate the [`Ord`] invariant between the elements of the set.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn insert_after(&mut self, value: T) -> Result<(), UnorderedKeyError> {
        self.inner.insert_after(value, SetValZST)
    }

    /// Inserts a new element into the set in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted element.
    ///
    /// If the inserted element is not greater than the element before the
    /// cursor (if any), or if it not less than the element after the cursor (if
    /// any), then an [`UnorderedKeyError`] is returned since this would
    /// invalidate the [`Ord`] invariant between the elements of the set.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn insert_before(&mut self, value: T) -> Result<(), UnorderedKeyError> {
        self.inner.insert_before(value, SetValZST)
    }

    /// Removes the next element from the `BTreeSet`.
    ///
    /// The element that was removed is returned. The cursor position is
    /// unchanged (before the removed element).
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn remove_next(&mut self) -> Option<T> {
        self.inner.remove_next().map(|(k, _)| k)
    }

    /// Removes the preceding element from the `BTreeSet`.
    ///
    /// The element that was removed is returned. The cursor position is
    /// unchanged (after the removed element).
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn remove_prev(&mut self) -> Option<T> {
        self.inner.remove_prev().map(|(k, _)| k)
    }
}

#[unstable(feature = "btree_cursors", issue = "107540")]
pub use super::map::UnorderedKeyError;

#[cfg(test)]
mod tests;
