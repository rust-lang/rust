use std::fmt;
use std::hash::Hash;

use super::map::SsoHashMap;

/// Small-storage-optimized implementation of a set.
///
/// Stores elements in a small array up to a certain length
/// and switches to `HashSet` when that length is exceeded.
//
// FIXME: Implements subset of HashSet API.
//
// Missing HashSet API:
//   all hasher-related
//   try_reserve
//   shrink_to (unstable)
//   drain_filter (unstable)
//   replace
//   get_or_insert/get_or_insert_owned/get_or_insert_with (unstable)
//   difference/symmetric_difference/intersection/union
//   is_disjoint/is_subset/is_superset
//   PartialEq/Eq (requires SsoHashMap implementation)
//   BitOr/BitAnd/BitXor/Sub
#[derive(Clone)]
pub struct SsoHashSet<T> {
    map: SsoHashMap<T, ()>,
}

/// Adapter function used to return
/// result if SsoHashMap functions into
/// result SsoHashSet should return.
#[inline(always)]
fn entry_to_key<K, V>((k, _v): (K, V)) -> K {
    k
}

impl<T> SsoHashSet<T> {
    /// Creates an empty `SsoHashSet`.
    #[inline]
    pub fn new() -> Self {
        Self { map: SsoHashMap::new() }
    }

    /// Creates an empty `SsoHashSet` with the specified capacity.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self { map: SsoHashMap::with_capacity(cap) }
    }

    /// Clears the set, removing all values.
    #[inline]
    pub fn clear(&mut self) {
        self.map.clear()
    }

    /// Returns the number of elements the set can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.map.capacity()
    }

    /// Returns the number of elements in the set.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the set contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// An iterator visiting all elements in arbitrary order.
    /// The iterator element type is `&'a T`.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.into_iter()
    }

    /// Clears the set, returning all elements in an iterator.
    #[inline]
    pub fn drain(&mut self) -> impl Iterator<Item = T> {
        self.map.drain().map(entry_to_key)
    }
}

impl<T: Eq + Hash> SsoHashSet<T> {
    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the `SsoHashSet`. The collection may reserve more space to avoid
    /// frequent reallocations.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.map.reserve(additional)
    }

    /// Shrinks the capacity of the set as much as possible. It will drop
    /// down as much as possible while maintaining the internal rules
    /// and possibly leaving some space in accordance with the resize policy.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.map.shrink_to_fit()
    }

    /// Retains only the elements specified by the predicate.
    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.map.retain(|k, _v| f(k))
    }

    /// Removes and returns the value in the set, if any, that is equal to the given one.
    #[inline]
    pub fn take(&mut self, value: &T) -> Option<T> {
        self.map.remove_entry(value).map(entry_to_key)
    }

    /// Returns a reference to the value in the set, if any, that is equal to the given value.
    #[inline]
    pub fn get(&self, value: &T) -> Option<&T> {
        self.map.get_key_value(value).map(entry_to_key)
    }

    /// Adds a value to the set.
    ///
    /// Returns whether the value was newly inserted. That is:
    ///
    /// - If the set did not previously contain this value, `true` is returned.
    /// - If the set already contained this value, `false` is returned.
    #[inline]
    pub fn insert(&mut self, elem: T) -> bool {
        self.map.insert(elem, ()).is_none()
    }

    /// Removes a value from the set. Returns whether the value was
    /// present in the set.
    #[inline]
    pub fn remove(&mut self, value: &T) -> bool {
        self.map.remove(value).is_some()
    }

    /// Returns `true` if the set contains a value.
    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        self.map.contains_key(value)
    }
}

impl<T: Eq + Hash> FromIterator<T> for SsoHashSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> SsoHashSet<T> {
        let mut set: SsoHashSet<T> = Default::default();
        set.extend(iter);
        set
    }
}

impl<T> Default for SsoHashSet<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Eq + Hash> Extend<T> for SsoHashSet<T> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for val in iter.into_iter() {
            self.insert(val);
        }
    }

    #[inline]
    fn extend_one(&mut self, item: T) {
        self.insert(item);
    }

    #[inline]
    fn extend_reserve(&mut self, additional: usize) {
        self.map.extend_reserve(additional)
    }
}

impl<'a, T> Extend<&'a T> for SsoHashSet<T>
where
    T: 'a + Eq + Hash + Copy,
{
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }

    #[inline]
    fn extend_one(&mut self, &item: &'a T) {
        self.insert(item);
    }

    #[inline]
    fn extend_reserve(&mut self, additional: usize) {
        Extend::<T>::extend_reserve(self, additional)
    }
}

impl<T> IntoIterator for SsoHashSet<T> {
    type IntoIter = std::iter::Map<<SsoHashMap<T, ()> as IntoIterator>::IntoIter, fn((T, ())) -> T>;
    type Item = <Self::IntoIter as Iterator>::Item;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter().map(entry_to_key)
    }
}

impl<'a, T> IntoIterator for &'a SsoHashSet<T> {
    type IntoIter = std::iter::Map<
        <&'a SsoHashMap<T, ()> as IntoIterator>::IntoIter,
        fn((&'a T, &'a ())) -> &'a T,
    >;
    type Item = <Self::IntoIter as Iterator>::Item;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.map.iter().map(entry_to_key)
    }
}

impl<T> fmt::Debug for SsoHashSet<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}
