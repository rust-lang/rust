#[cfg(test)]
mod tests;

use hashbrown::hash_set as base;

use crate::borrow::Borrow;
use crate::collections::TryReserveError;
use crate::fmt;
use crate::hash::{BuildHasher, Hash};
use crate::iter::{Chain, FromIterator, FusedIterator};
use crate::ops::{BitAnd, BitOr, BitXor, Sub};

use super::map::{map_try_reserve_error, RandomState};

// Future Optimization (FIXME!)
// ============================
//
// Iteration over zero sized values is a noop. There is no need
// for `bucket.val` in the case of HashSet. I suppose we would need HKT
// to get rid of it properly.

/// A [hash set] implemented as a `HashMap` where the value is `()`.
///
/// As with the [`HashMap`] type, a `HashSet` requires that the elements
/// implement the [`Eq`] and [`Hash`] traits. This can frequently be achieved by
/// using `#[derive(PartialEq, Eq, Hash)]`. If you implement these yourself,
/// it is important that the following property holds:
///
/// ```text
/// k1 == k2 -> hash(k1) == hash(k2)
/// ```
///
/// In other words, if two keys are equal, their hashes must be equal.
///
///
/// It is a logic error for an item to be modified in such a way that the
/// item's hash, as determined by the [`Hash`] trait, or its equality, as
/// determined by the [`Eq`] trait, changes while it is in the set. This is
/// normally only possible through [`Cell`], [`RefCell`], global state, I/O, or
/// unsafe code. The behavior resulting from such a logic error is not
/// specified (it could include panics, incorrect results, aborts, memory
/// leaks, or non-termination) but will not be undefined behavior.
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
/// // Type inference lets us omit an explicit type signature (which
/// // would be `HashSet<String>` in this example).
/// let mut books = HashSet::new();
///
/// // Add some books.
/// books.insert("A Dance With Dragons".to_string());
/// books.insert("To Kill a Mockingbird".to_string());
/// books.insert("The Odyssey".to_string());
/// books.insert("The Great Gatsby".to_string());
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
/// The easiest way to use `HashSet` with a custom type is to derive
/// [`Eq`] and [`Hash`]. We must also derive [`PartialEq`], this will in the
/// future be implied by [`Eq`].
///
/// ```
/// use std::collections::HashSet;
/// #[derive(Hash, Eq, PartialEq, Debug)]
/// struct Viking {
///     name: String,
///     power: usize,
/// }
///
/// let mut vikings = HashSet::new();
///
/// vikings.insert(Viking { name: "Einar".to_string(), power: 9 });
/// vikings.insert(Viking { name: "Einar".to_string(), power: 9 });
/// vikings.insert(Viking { name: "Olaf".to_string(), power: 4 });
/// vikings.insert(Viking { name: "Harald".to_string(), power: 8 });
///
/// // Use derived implementation to print the vikings.
/// for x in &vikings {
///     println!("{:?}", x);
/// }
/// ```
///
/// A `HashSet` with a known list of items can be initialized from an array:
///
/// ```
/// use std::collections::HashSet;
///
/// let viking_names = HashSet::from(["Einar", "Olaf", "Harald"]);
/// ```
///
/// [hash set]: crate::collections#use-the-set-variant-of-any-of-these-maps-when
/// [`HashMap`]: crate::collections::HashMap
/// [`RefCell`]: crate::cell::RefCell
/// [`Cell`]: crate::cell::Cell
#[cfg_attr(not(test), rustc_diagnostic_item = "HashSet")]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct HashSet<T, S = RandomState> {
    base: base::HashSet<T, S>,
}

impl<T> HashSet<T, RandomState> {
    /// Creates an empty `HashSet`.
    ///
    /// The hash set is initially created with a capacity of 0, so it will not allocate until it
    /// is first inserted into.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let set: HashSet<i32> = HashSet::new();
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> HashSet<T, RandomState> {
        Default::default()
    }

    /// Creates an empty `HashSet` with the specified capacity.
    ///
    /// The hash set will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash set will not allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let set: HashSet<i32> = HashSet::with_capacity(10);
    /// assert!(set.capacity() >= 10);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(capacity: usize) -> HashSet<T, RandomState> {
        HashSet { base: base::HashSet::with_capacity_and_hasher(capacity, Default::default()) }
    }
}

impl<T, S> HashSet<T, S> {
    /// Returns the number of elements the set can hold without reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let set: HashSet<i32> = HashSet::with_capacity(100);
    /// assert!(set.capacity() >= 100);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn capacity(&self) -> usize {
        self.base.capacity()
    }

    /// An iterator visiting all elements in arbitrary order.
    /// The iterator element type is `&'a T`.
    ///
    /// # Examples
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
    #[inline]
    #[cfg_attr(not(bootstrap), rustc_lint_query_instability)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter { base: self.base.iter() }
    }

    /// Returns the number of elements in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut v = HashSet::new();
    /// assert_eq!(v.len(), 0);
    /// v.insert(1);
    /// assert_eq!(v.len(), 1);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize {
        self.base.len()
    }

    /// Returns `true` if the set contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut v = HashSet::new();
    /// assert!(v.is_empty());
    /// v.insert(1);
    /// assert!(!v.is_empty());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool {
        self.base.is_empty()
    }

    /// Clears the set, returning all elements in an iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut set: HashSet<_> = [1, 2, 3].iter().cloned().collect();
    /// assert!(!set.is_empty());
    ///
    /// // print 1, 2, 3 in an arbitrary order
    /// for i in set.drain() {
    ///     println!("{}", i);
    /// }
    ///
    /// assert!(set.is_empty());
    /// ```
    #[inline]
    #[cfg_attr(not(bootstrap), rustc_lint_query_instability)]
    #[stable(feature = "drain", since = "1.6.0")]
    pub fn drain(&mut self) -> Drain<'_, T> {
        Drain { base: self.base.drain() }
    }

    /// Creates an iterator which uses a closure to determine if a value should be removed.
    ///
    /// If the closure returns true, then the value is removed and yielded.
    /// If the closure returns false, the value will remain in the list and will not be yielded
    /// by the iterator.
    ///
    /// If the iterator is only partially consumed or not consumed at all, each of the remaining
    /// values will still be subjected to the closure and removed and dropped if it returns true.
    ///
    /// It is unspecified how many more values will be subjected to the closure
    /// if a panic occurs in the closure, or if a panic occurs while dropping a value, or if the
    /// `DrainFilter` itself is leaked.
    ///
    /// # Examples
    ///
    /// Splitting a set into even and odd values, reusing the original set:
    ///
    /// ```
    /// #![feature(hash_drain_filter)]
    /// use std::collections::HashSet;
    ///
    /// let mut set: HashSet<i32> = (0..8).collect();
    /// let drained: HashSet<i32> = set.drain_filter(|v| v % 2 == 0).collect();
    ///
    /// let mut evens = drained.into_iter().collect::<Vec<_>>();
    /// let mut odds = set.into_iter().collect::<Vec<_>>();
    /// evens.sort();
    /// odds.sort();
    ///
    /// assert_eq!(evens, vec![0, 2, 4, 6]);
    /// assert_eq!(odds, vec![1, 3, 5, 7]);
    /// ```
    #[inline]
    #[cfg_attr(not(bootstrap), rustc_lint_query_instability)]
    #[unstable(feature = "hash_drain_filter", issue = "59618")]
    pub fn drain_filter<F>(&mut self, pred: F) -> DrainFilter<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        DrainFilter { base: self.base.drain_filter(pred) }
    }

    /// Clears the set, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut v = HashSet::new();
    /// v.insert(1);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn clear(&mut self) {
        self.base.clear()
    }

    /// Creates a new empty hash set which will use the given hasher to hash
    /// keys.
    ///
    /// The hash set is also created with the default initial capacity.
    ///
    /// Warning: `hasher` is normally randomly generated, and
    /// is designed to allow `HashSet`s to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    /// The `hash_builder` passed should implement the [`BuildHasher`] trait for
    /// the HashMap to be useful, see its documentation for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let mut set = HashSet::with_hasher(s);
    /// set.insert(2);
    /// ```
    #[inline]
    #[stable(feature = "hashmap_build_hasher", since = "1.7.0")]
    pub fn with_hasher(hasher: S) -> HashSet<T, S> {
        HashSet { base: base::HashSet::with_hasher(hasher) }
    }

    /// Creates an empty `HashSet` with the specified capacity, using
    /// `hasher` to hash the keys.
    ///
    /// The hash set will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash set will not allocate.
    ///
    /// Warning: `hasher` is normally randomly generated, and
    /// is designed to allow `HashSet`s to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    /// The `hash_builder` passed should implement the [`BuildHasher`] trait for
    /// the HashMap to be useful, see its documentation for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let mut set = HashSet::with_capacity_and_hasher(10, s);
    /// set.insert(1);
    /// ```
    #[inline]
    #[stable(feature = "hashmap_build_hasher", since = "1.7.0")]
    pub fn with_capacity_and_hasher(capacity: usize, hasher: S) -> HashSet<T, S> {
        HashSet { base: base::HashSet::with_capacity_and_hasher(capacity, hasher) }
    }

    /// Returns a reference to the set's [`BuildHasher`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let hasher = RandomState::new();
    /// let set: HashSet<i32> = HashSet::with_hasher(hasher);
    /// let hasher: &RandomState = set.hasher();
    /// ```
    #[inline]
    #[stable(feature = "hashmap_public_hasher", since = "1.9.0")]
    pub fn hasher(&self) -> &S {
        self.base.hasher()
    }
}

impl<T, S> HashSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the `HashSet`. The collection may reserve more space to avoid
    /// frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new allocation size overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let mut set: HashSet<i32> = HashSet::new();
    /// set.reserve(10);
    /// assert!(set.capacity() >= 10);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve(&mut self, additional: usize) {
        self.base.reserve(additional)
    }

    /// Tries to reserve capacity for at least `additional` more elements to be inserted
    /// in the given `HashSet<K, V>`. The collection may reserve more space to avoid
    /// frequent reallocations.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let mut set: HashSet<i32> = HashSet::new();
    /// set.try_reserve(10).expect("why is the test harness OOMing on 10 bytes?");
    /// ```
    #[inline]
    #[stable(feature = "try_reserve", since = "1.57.0")]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.base.try_reserve(additional).map_err(map_try_reserve_error)
    }

    /// Shrinks the capacity of the set as much as possible. It will drop
    /// down as much as possible while maintaining the internal rules
    /// and possibly leaving some space in accordance with the resize policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut set = HashSet::with_capacity(100);
    /// set.insert(1);
    /// set.insert(2);
    /// assert!(set.capacity() >= 100);
    /// set.shrink_to_fit();
    /// assert!(set.capacity() >= 2);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn shrink_to_fit(&mut self) {
        self.base.shrink_to_fit()
    }

    /// Shrinks the capacity of the set with a lower limit. It will drop
    /// down no lower than the supplied limit while maintaining the internal rules
    /// and possibly leaving some space in accordance with the resize policy.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut set = HashSet::with_capacity(100);
    /// set.insert(1);
    /// set.insert(2);
    /// assert!(set.capacity() >= 100);
    /// set.shrink_to(10);
    /// assert!(set.capacity() >= 10);
    /// set.shrink_to(0);
    /// assert!(set.capacity() >= 2);
    /// ```
    #[inline]
    #[stable(feature = "shrink_to", since = "1.56.0")]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.base.shrink_to(min_capacity)
    }

    /// Visits the values representing the difference,
    /// i.e., the values that are in `self` but not in `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let a: HashSet<_> = [1, 2, 3].iter().cloned().collect();
    /// let b: HashSet<_> = [4, 2, 3, 4].iter().cloned().collect();
    ///
    /// // Can be seen as `a - b`.
    /// for x in a.difference(&b) {
    ///     println!("{}", x); // Print 1
    /// }
    ///
    /// let diff: HashSet<_> = a.difference(&b).collect();
    /// assert_eq!(diff, [1].iter().collect());
    ///
    /// // Note that difference is not symmetric,
    /// // and `b - a` means something else:
    /// let diff: HashSet<_> = b.difference(&a).collect();
    /// assert_eq!(diff, [4].iter().collect());
    /// ```
    #[inline]
    #[cfg_attr(not(bootstrap), rustc_lint_query_instability)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn difference<'a>(&'a self, other: &'a HashSet<T, S>) -> Difference<'a, T, S> {
        Difference { iter: self.iter(), other }
    }

    /// Visits the values representing the symmetric difference,
    /// i.e., the values that are in `self` or in `other` but not in both.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let a: HashSet<_> = [1, 2, 3].iter().cloned().collect();
    /// let b: HashSet<_> = [4, 2, 3, 4].iter().cloned().collect();
    ///
    /// // Print 1, 4 in arbitrary order.
    /// for x in a.symmetric_difference(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let diff1: HashSet<_> = a.symmetric_difference(&b).collect();
    /// let diff2: HashSet<_> = b.symmetric_difference(&a).collect();
    ///
    /// assert_eq!(diff1, diff2);
    /// assert_eq!(diff1, [1, 4].iter().collect());
    /// ```
    #[inline]
    #[cfg_attr(not(bootstrap), rustc_lint_query_instability)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn symmetric_difference<'a>(
        &'a self,
        other: &'a HashSet<T, S>,
    ) -> SymmetricDifference<'a, T, S> {
        SymmetricDifference { iter: self.difference(other).chain(other.difference(self)) }
    }

    /// Visits the values representing the intersection,
    /// i.e., the values that are both in `self` and `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let a: HashSet<_> = [1, 2, 3].iter().cloned().collect();
    /// let b: HashSet<_> = [4, 2, 3, 4].iter().cloned().collect();
    ///
    /// // Print 2, 3 in arbitrary order.
    /// for x in a.intersection(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let intersection: HashSet<_> = a.intersection(&b).collect();
    /// assert_eq!(intersection, [2, 3].iter().collect());
    /// ```
    #[inline]
    #[cfg_attr(not(bootstrap), rustc_lint_query_instability)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn intersection<'a>(&'a self, other: &'a HashSet<T, S>) -> Intersection<'a, T, S> {
        if self.len() <= other.len() {
            Intersection { iter: self.iter(), other }
        } else {
            Intersection { iter: other.iter(), other: self }
        }
    }

    /// Visits the values representing the union,
    /// i.e., all the values in `self` or `other`, without duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// let a: HashSet<_> = [1, 2, 3].iter().cloned().collect();
    /// let b: HashSet<_> = [4, 2, 3, 4].iter().cloned().collect();
    ///
    /// // Print 1, 2, 3, 4 in arbitrary order.
    /// for x in a.union(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let union: HashSet<_> = a.union(&b).collect();
    /// assert_eq!(union, [1, 2, 3, 4].iter().collect());
    /// ```
    #[inline]
    #[cfg_attr(not(bootstrap), rustc_lint_query_instability)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn union<'a>(&'a self, other: &'a HashSet<T, S>) -> Union<'a, T, S> {
        if self.len() >= other.len() {
            Union { iter: self.iter().chain(other.difference(self)) }
        } else {
            Union { iter: other.iter().chain(self.difference(other)) }
        }
    }

    /// Returns `true` if the set contains a value.
    ///
    /// The value may be any borrowed form of the set's value type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the value type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let set: HashSet<_> = [1, 2, 3].iter().cloned().collect();
    /// assert_eq!(set.contains(&1), true);
    /// assert_eq!(set.contains(&4), false);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn contains<Q: ?Sized>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.base.contains(value)
    }

    /// Returns a reference to the value in the set, if any, that is equal to the given value.
    ///
    /// The value may be any borrowed form of the set's value type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the value type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let set: HashSet<_> = [1, 2, 3].iter().cloned().collect();
    /// assert_eq!(set.get(&2), Some(&2));
    /// assert_eq!(set.get(&4), None);
    /// ```
    #[inline]
    #[stable(feature = "set_recovery", since = "1.9.0")]
    pub fn get<Q: ?Sized>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.base.get(value)
    }

    /// Inserts the given `value` into the set if it is not present, then
    /// returns a reference to the value in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(hash_set_entry)]
    ///
    /// use std::collections::HashSet;
    ///
    /// let mut set: HashSet<_> = [1, 2, 3].iter().cloned().collect();
    /// assert_eq!(set.len(), 3);
    /// assert_eq!(set.get_or_insert(2), &2);
    /// assert_eq!(set.get_or_insert(100), &100);
    /// assert_eq!(set.len(), 4); // 100 was inserted
    /// ```
    #[inline]
    #[unstable(feature = "hash_set_entry", issue = "60896")]
    pub fn get_or_insert(&mut self, value: T) -> &T {
        // Although the raw entry gives us `&mut T`, we only return `&T` to be consistent with
        // `get`. Key mutation is "raw" because you're not supposed to affect `Eq` or `Hash`.
        self.base.get_or_insert(value)
    }

    /// Inserts an owned copy of the given `value` into the set if it is not
    /// present, then returns a reference to the value in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(hash_set_entry)]
    ///
    /// use std::collections::HashSet;
    ///
    /// let mut set: HashSet<String> = ["cat", "dog", "horse"]
    ///     .iter().map(|&pet| pet.to_owned()).collect();
    ///
    /// assert_eq!(set.len(), 3);
    /// for &pet in &["cat", "dog", "fish"] {
    ///     let value = set.get_or_insert_owned(pet);
    ///     assert_eq!(value, pet);
    /// }
    /// assert_eq!(set.len(), 4); // a new "fish" was inserted
    /// ```
    #[inline]
    #[unstable(feature = "hash_set_entry", issue = "60896")]
    pub fn get_or_insert_owned<Q: ?Sized>(&mut self, value: &Q) -> &T
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = T>,
    {
        // Although the raw entry gives us `&mut T`, we only return `&T` to be consistent with
        // `get`. Key mutation is "raw" because you're not supposed to affect `Eq` or `Hash`.
        self.base.get_or_insert_owned(value)
    }

    /// Inserts a value computed from `f` into the set if the given `value` is
    /// not present, then returns a reference to the value in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(hash_set_entry)]
    ///
    /// use std::collections::HashSet;
    ///
    /// let mut set: HashSet<String> = ["cat", "dog", "horse"]
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
    #[unstable(feature = "hash_set_entry", issue = "60896")]
    pub fn get_or_insert_with<Q: ?Sized, F>(&mut self, value: &Q, f: F) -> &T
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
        F: FnOnce(&Q) -> T,
    {
        // Although the raw entry gives us `&mut T`, we only return `&T` to be consistent with
        // `get`. Key mutation is "raw" because you're not supposed to affect `Eq` or `Hash`.
        self.base.get_or_insert_with(value, f)
    }

    /// Returns `true` if `self` has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    ///
    /// # Examples
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
        if self.len() <= other.len() {
            self.iter().all(|v| !other.contains(v))
        } else {
            other.iter().all(|v| !self.contains(v))
        }
    }

    /// Returns `true` if the set is a subset of another,
    /// i.e., `other` contains at least all the values in `self`.
    ///
    /// # Examples
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
        if self.len() <= other.len() { self.iter().all(|v| other.contains(v)) } else { false }
    }

    /// Returns `true` if the set is a superset of another,
    /// i.e., `self` contains at least all the values in `other`.
    ///
    /// # Examples
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

    /// Adds a value to the set.
    ///
    /// If the set did not have this value present, `true` is returned.
    ///
    /// If the set did have this value present, `false` is returned.
    ///
    /// # Examples
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
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(&mut self, value: T) -> bool {
        self.base.insert(value)
    }

    /// Adds a value to the set, replacing the existing value, if any, that is equal to the given
    /// one. Returns the replaced value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut set = HashSet::new();
    /// set.insert(Vec::<i32>::new());
    ///
    /// assert_eq!(set.get(&[][..]).unwrap().capacity(), 0);
    /// set.replace(Vec::with_capacity(10));
    /// assert_eq!(set.get(&[][..]).unwrap().capacity(), 10);
    /// ```
    #[inline]
    #[stable(feature = "set_recovery", since = "1.9.0")]
    pub fn replace(&mut self, value: T) -> Option<T> {
        self.base.replace(value)
    }

    /// Removes a value from the set. Returns whether the value was
    /// present in the set.
    ///
    /// The value may be any borrowed form of the set's value type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the value type.
    ///
    /// # Examples
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
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove<Q: ?Sized>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.base.remove(value)
    }

    /// Removes and returns the value in the set, if any, that is equal to the given one.
    ///
    /// The value may be any borrowed form of the set's value type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the value type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut set: HashSet<_> = [1, 2, 3].iter().cloned().collect();
    /// assert_eq!(set.take(&2), Some(2));
    /// assert_eq!(set.take(&2), None);
    /// ```
    #[inline]
    #[stable(feature = "set_recovery", since = "1.9.0")]
    pub fn take<Q: ?Sized>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.base.take(value)
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` such that `f(&e)` returns `false`.
    /// The elements are visited in unsorted (and unspecified) order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let xs = [1, 2, 3, 4, 5, 6];
    /// let mut set: HashSet<i32> = xs.iter().cloned().collect();
    /// set.retain(|&k| k % 2 == 0);
    /// assert_eq!(set.len(), 3);
    /// ```
    #[cfg_attr(not(bootstrap), rustc_lint_query_instability)]
    #[stable(feature = "retain_hash_collection", since = "1.18.0")]
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.base.retain(f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S> Clone for HashSet<T, S>
where
    T: Clone,
    S: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { base: self.base.clone() }
    }

    #[inline]
    fn clone_from(&mut self, other: &Self) {
        self.base.clone_from(&other.base);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S> PartialEq for HashSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    fn eq(&self, other: &HashSet<T, S>) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter().all(|key| other.contains(key))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S> Eq for HashSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S> fmt::Debug for HashSet<T, S>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S> FromIterator<T> for HashSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher + Default,
{
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> HashSet<T, S> {
        let mut set = HashSet::with_hasher(Default::default());
        set.extend(iter);
        set
    }
}

#[stable(feature = "std_collections_from_array", since = "1.56.0")]
// Note: as what is currently the most convenient built-in way to construct
// a HashSet, a simple usage of this function must not *require* the user
// to provide a type annotation in order to infer the third type parameter
// (the hasher parameter, conventionally "S").
// To that end, this impl is defined using RandomState as the concrete
// type of S, rather than being generic over `S: BuildHasher + Default`.
// It is expected that users who want to specify a hasher will manually use
// `with_capacity_and_hasher`.
// If type parameter defaults worked on impls, and if type parameter
// defaults could be mixed with const generics, then perhaps
// this could be generalized.
// See also the equivalent impl on HashMap.
impl<T, const N: usize> From<[T; N]> for HashSet<T, RandomState>
where
    T: Eq + Hash,
{
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let set1 = HashSet::from([1, 2, 3, 4]);
    /// let set2: HashSet<_> = [1, 2, 3, 4].into();
    /// assert_eq!(set1, set2);
    /// ```
    fn from(arr: [T; N]) -> Self {
        crate::array::IntoIter::new(arr).collect()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S> Extend<T> for HashSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.base.extend(iter);
    }

    #[inline]
    fn extend_one(&mut self, item: T) {
        self.base.insert(item);
    }

    #[inline]
    fn extend_reserve(&mut self, additional: usize) {
        self.base.extend_reserve(additional);
    }
}

#[stable(feature = "hash_extend_copy", since = "1.4.0")]
impl<'a, T, S> Extend<&'a T> for HashSet<T, S>
where
    T: 'a + Eq + Hash + Copy,
    S: BuildHasher,
{
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }

    #[inline]
    fn extend_one(&mut self, &item: &'a T) {
        self.base.insert(item);
    }

    #[inline]
    fn extend_reserve(&mut self, additional: usize) {
        Extend::<T>::extend_reserve(self, additional)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S> Default for HashSet<T, S>
where
    S: Default,
{
    /// Creates an empty `HashSet<T, S>` with the `Default` value for the hasher.
    #[inline]
    fn default() -> HashSet<T, S> {
        HashSet { base: Default::default() }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S> BitOr<&HashSet<T, S>> for &HashSet<T, S>
where
    T: Eq + Hash + Clone,
    S: BuildHasher + Default,
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
    /// for x in &set {
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
impl<T, S> BitAnd<&HashSet<T, S>> for &HashSet<T, S>
where
    T: Eq + Hash + Clone,
    S: BuildHasher + Default,
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
    /// for x in &set {
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
impl<T, S> BitXor<&HashSet<T, S>> for &HashSet<T, S>
where
    T: Eq + Hash + Clone,
    S: BuildHasher + Default,
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
    /// for x in &set {
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
impl<T, S> Sub<&HashSet<T, S>> for &HashSet<T, S>
where
    T: Eq + Hash + Clone,
    S: BuildHasher + Default,
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
    /// for x in &set {
    ///     assert!(expected.contains(x));
    ///     i += 1;
    /// }
    /// assert_eq!(i, expected.len());
    /// ```
    fn sub(self, rhs: &HashSet<T, S>) -> HashSet<T, S> {
        self.difference(rhs).cloned().collect()
    }
}

/// An iterator over the items of a `HashSet`.
///
/// This `struct` is created by the [`iter`] method on [`HashSet`].
/// See its documentation for more.
///
/// [`iter`]: HashSet::iter
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
///
/// let a: HashSet<u32> = vec![1, 2, 3].into_iter().collect();
///
/// let mut iter = a.iter();
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, K: 'a> {
    base: base::Iter<'a, K>,
}

/// An owning iterator over the items of a `HashSet`.
///
/// This `struct` is created by the [`into_iter`] method on [`HashSet`]
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
///
/// [`into_iter`]: IntoIterator::into_iter
/// [`IntoIterator`]: crate::iter::IntoIterator
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
///
/// let a: HashSet<u32> = vec![1, 2, 3].into_iter().collect();
///
/// let mut iter = a.into_iter();
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<K> {
    base: base::IntoIter<K>,
}

/// A draining iterator over the items of a `HashSet`.
///
/// This `struct` is created by the [`drain`] method on [`HashSet`].
/// See its documentation for more.
///
/// [`drain`]: HashSet::drain
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
///
/// let mut a: HashSet<u32> = vec![1, 2, 3].into_iter().collect();
///
/// let mut drain = a.drain();
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Drain<'a, K: 'a> {
    base: base::Drain<'a, K>,
}

/// A draining, filtering iterator over the items of a `HashSet`.
///
/// This `struct` is created by the [`drain_filter`] method on [`HashSet`].
///
/// [`drain_filter`]: HashSet::drain_filter
///
/// # Examples
///
/// ```
/// #![feature(hash_drain_filter)]
///
/// use std::collections::HashSet;
///
/// let mut a: HashSet<u32> = vec![1, 2, 3].into_iter().collect();
///
/// let mut drain_filtered = a.drain_filter(|v| v % 2 == 0);
/// ```
#[unstable(feature = "hash_drain_filter", issue = "59618")]
pub struct DrainFilter<'a, K, F>
where
    F: FnMut(&K) -> bool,
{
    base: base::DrainFilter<'a, K, F>,
}

/// A lazy iterator producing elements in the intersection of `HashSet`s.
///
/// This `struct` is created by the [`intersection`] method on [`HashSet`].
/// See its documentation for more.
///
/// [`intersection`]: HashSet::intersection
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
///
/// let a: HashSet<u32> = vec![1, 2, 3].into_iter().collect();
/// let b: HashSet<_> = [4, 2, 3, 4].iter().cloned().collect();
///
/// let mut intersection = a.intersection(&b);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Intersection<'a, T: 'a, S: 'a> {
    // iterator of the first set
    iter: Iter<'a, T>,
    // the second set
    other: &'a HashSet<T, S>,
}

/// A lazy iterator producing elements in the difference of `HashSet`s.
///
/// This `struct` is created by the [`difference`] method on [`HashSet`].
/// See its documentation for more.
///
/// [`difference`]: HashSet::difference
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
///
/// let a: HashSet<u32> = vec![1, 2, 3].into_iter().collect();
/// let b: HashSet<_> = [4, 2, 3, 4].iter().cloned().collect();
///
/// let mut difference = a.difference(&b);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Difference<'a, T: 'a, S: 'a> {
    // iterator of the first set
    iter: Iter<'a, T>,
    // the second set
    other: &'a HashSet<T, S>,
}

/// A lazy iterator producing elements in the symmetric difference of `HashSet`s.
///
/// This `struct` is created by the [`symmetric_difference`] method on
/// [`HashSet`]. See its documentation for more.
///
/// [`symmetric_difference`]: HashSet::symmetric_difference
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
///
/// let a: HashSet<u32> = vec![1, 2, 3].into_iter().collect();
/// let b: HashSet<_> = [4, 2, 3, 4].iter().cloned().collect();
///
/// let mut intersection = a.symmetric_difference(&b);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SymmetricDifference<'a, T: 'a, S: 'a> {
    iter: Chain<Difference<'a, T, S>, Difference<'a, T, S>>,
}

/// A lazy iterator producing elements in the union of `HashSet`s.
///
/// This `struct` is created by the [`union`] method on [`HashSet`].
/// See its documentation for more.
///
/// [`union`]: HashSet::union
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
///
/// let a: HashSet<u32> = vec![1, 2, 3].into_iter().collect();
/// let b: HashSet<_> = [4, 2, 3, 4].iter().cloned().collect();
///
/// let mut union_iter = a.union(&b);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Union<'a, T: 'a, S: 'a> {
    iter: Chain<Iter<'a, T>, Difference<'a, T, S>>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, S> IntoIterator for &'a HashSet<T, S> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    #[cfg_attr(not(bootstrap), rustc_lint_query_instability)]
    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S> IntoIterator for HashSet<T, S> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    /// Creates a consuming iterator, that is, one that moves each value out
    /// of the set in arbitrary order. The set cannot be used after calling
    /// this.
    ///
    /// # Examples
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
    /// for x in &v {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    #[cfg_attr(not(bootstrap), rustc_lint_query_instability)]
    fn into_iter(self) -> IntoIter<T> {
        IntoIter { base: self.base.into_iter() }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K> Clone for Iter<'_, K> {
    #[inline]
    fn clone(&self) -> Self {
        Iter { base: self.base.clone() }
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K> Iterator for Iter<'a, K> {
    type Item = &'a K;

    #[inline]
    fn next(&mut self) -> Option<&'a K> {
        self.base.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.base.size_hint()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<K> ExactSizeIterator for Iter<'_, K> {
    #[inline]
    fn len(&self) -> usize {
        self.base.len()
    }
}
#[stable(feature = "fused", since = "1.26.0")]
impl<K> FusedIterator for Iter<'_, K> {}

#[stable(feature = "std_debug", since = "1.16.0")]
impl<K: fmt::Debug> fmt::Debug for Iter<'_, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K> Iterator for IntoIter<K> {
    type Item = K;

    #[inline]
    fn next(&mut self) -> Option<K> {
        self.base.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.base.size_hint()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<K> ExactSizeIterator for IntoIter<K> {
    #[inline]
    fn len(&self) -> usize {
        self.base.len()
    }
}
#[stable(feature = "fused", since = "1.26.0")]
impl<K> FusedIterator for IntoIter<K> {}

#[stable(feature = "std_debug", since = "1.16.0")]
impl<K: fmt::Debug> fmt::Debug for IntoIter<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.base, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K> Iterator for Drain<'a, K> {
    type Item = K;

    #[inline]
    fn next(&mut self) -> Option<K> {
        self.base.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.base.size_hint()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<K> ExactSizeIterator for Drain<'_, K> {
    #[inline]
    fn len(&self) -> usize {
        self.base.len()
    }
}
#[stable(feature = "fused", since = "1.26.0")]
impl<K> FusedIterator for Drain<'_, K> {}

#[stable(feature = "std_debug", since = "1.16.0")]
impl<K: fmt::Debug> fmt::Debug for Drain<'_, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.base, f)
    }
}

#[unstable(feature = "hash_drain_filter", issue = "59618")]
impl<K, F> Iterator for DrainFilter<'_, K, F>
where
    F: FnMut(&K) -> bool,
{
    type Item = K;

    #[inline]
    fn next(&mut self) -> Option<K> {
        self.base.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.base.size_hint()
    }
}

#[unstable(feature = "hash_drain_filter", issue = "59618")]
impl<K, F> FusedIterator for DrainFilter<'_, K, F> where F: FnMut(&K) -> bool {}

#[unstable(feature = "hash_drain_filter", issue = "59618")]
impl<'a, K, F> fmt::Debug for DrainFilter<'a, K, F>
where
    F: FnMut(&K) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DrainFilter").finish_non_exhaustive()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S> Clone for Intersection<'_, T, S> {
    #[inline]
    fn clone(&self) -> Self {
        Intersection { iter: self.iter.clone(), ..*self }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, S> Iterator for Intersection<'a, T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        loop {
            let elt = self.iter.next()?;
            if self.other.contains(elt) {
                return Some(elt);
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper)
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl<T, S> fmt::Debug for Intersection<'_, T, S>
where
    T: fmt::Debug + Eq + Hash,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, S> FusedIterator for Intersection<'_, T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S> Clone for Difference<'_, T, S> {
    #[inline]
    fn clone(&self) -> Self {
        Difference { iter: self.iter.clone(), ..*self }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, S> Iterator for Difference<'a, T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        loop {
            let elt = self.iter.next()?;
            if !self.other.contains(elt) {
                return Some(elt);
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper)
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, S> FusedIterator for Difference<'_, T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl<T, S> fmt::Debug for Difference<'_, T, S>
where
    T: fmt::Debug + Eq + Hash,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S> Clone for SymmetricDifference<'_, T, S> {
    #[inline]
    fn clone(&self) -> Self {
        SymmetricDifference { iter: self.iter.clone() }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, S> Iterator for SymmetricDifference<'a, T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        self.iter.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, S> FusedIterator for SymmetricDifference<'_, T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl<T, S> fmt::Debug for SymmetricDifference<'_, T, S>
where
    T: fmt::Debug + Eq + Hash,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, S> Clone for Union<'_, T, S> {
    #[inline]
    fn clone(&self) -> Self {
        Union { iter: self.iter.clone() }
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, S> FusedIterator for Union<'_, T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl<T, S> fmt::Debug for Union<'_, T, S>
where
    T: fmt::Debug + Eq + Hash,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, S> Iterator for Union<'a, T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        self.iter.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[allow(dead_code)]
fn assert_covariance() {
    fn set<'new>(v: HashSet<&'static str>) -> HashSet<&'new str> {
        v
    }
    fn iter<'a, 'new>(v: Iter<'a, &'static str>) -> Iter<'a, &'new str> {
        v
    }
    fn into_iter<'new>(v: IntoIter<&'static str>) -> IntoIter<&'new str> {
        v
    }
    fn difference<'a, 'new>(
        v: Difference<'a, &'static str, RandomState>,
    ) -> Difference<'a, &'new str, RandomState> {
        v
    }
    fn symmetric_difference<'a, 'new>(
        v: SymmetricDifference<'a, &'static str, RandomState>,
    ) -> SymmetricDifference<'a, &'new str, RandomState> {
        v
    }
    fn intersection<'a, 'new>(
        v: Intersection<'a, &'static str, RandomState>,
    ) -> Intersection<'a, &'new str, RandomState> {
        v
    }
    fn union<'a, 'new>(
        v: Union<'a, &'static str, RandomState>,
    ) -> Union<'a, &'new str, RandomState> {
        v
    }
    fn drain<'new>(d: Drain<'static, &'static str>) -> Drain<'new, &'new str> {
        d
    }
}
