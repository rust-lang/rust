use super::either_iter::EitherIter;
use crate::fx::FxHashMap;
use arrayvec::ArrayVec;
use std::fmt;
use std::hash::Hash;
use std::iter::FromIterator;
use std::ops::Index;

// For pointer-sized arguments arrays
// are faster than set/map for up to 64
// arguments.
//
// On the other hand such a big array
// hurts cache performance, makes passing
// sso structures around very expensive.
//
// Biggest performance benefit is gained
// for reasonably small arrays that stay
// small in vast majority of cases.
//
// '8' is chosen as a sane default, to be
// reevaluated later.
const SSO_ARRAY_SIZE: usize = 8;

/// Small-storage-optimized implementation of a map.
///
/// Stores elements in a small array up to a certain length
/// and switches to `HashMap` when that length is exceeded.
//
// FIXME: Implements subset of HashMap API.
//
// Missing HashMap API:
//   all hasher-related
//   try_reserve
//   shrink_to (unstable)
//   drain_filter (unstable)
//   into_keys/into_values (unstable)
//   all raw_entry-related
//   PartialEq/Eq (requires sorting the array)
//   Entry::or_insert_with_key
//   Vacant/Occupied entries and related
//
// FIXME: In HashMap most methods accepting key reference
// accept reference to generic `Q` where `K: Borrow<Q>`.
//
// However, using this approach in `HashMap::get` apparently
// breaks inlining and noticeably reduces performance.
//
// Performance *should* be the same given that borrow is
// a NOP in most cases, but in practice that's not the case.
//
// Further investigation is required.
//
// Affected methods:
//   SsoHashMap::get
//   SsoHashMap::get_mut
//   SsoHashMap::get_entry
//   SsoHashMap::get_key_value
//   SsoHashMap::contains_key
//   SsoHashMap::remove
//   SsoHashMap::remove_entry
//   Index::index
//   SsoHashSet::take
//   SsoHashSet::get
//   SsoHashSet::remove
//   SsoHashSet::contains

#[derive(Clone)]
pub enum SsoHashMap<K, V> {
    Array(ArrayVec<(K, V), SSO_ARRAY_SIZE>),
    Map(FxHashMap<K, V>),
}

impl<K, V> SsoHashMap<K, V> {
    /// Creates an empty `SsoHashMap`.
    #[inline]
    pub fn new() -> Self {
        SsoHashMap::Array(ArrayVec::new())
    }

    /// Creates an empty `SsoHashMap` with the specified capacity.
    pub fn with_capacity(cap: usize) -> Self {
        if cap <= SSO_ARRAY_SIZE {
            Self::new()
        } else {
            SsoHashMap::Map(FxHashMap::with_capacity_and_hasher(cap, Default::default()))
        }
    }

    /// Clears the map, removing all key-value pairs. Keeps the allocated memory
    /// for reuse.
    pub fn clear(&mut self) {
        match self {
            SsoHashMap::Array(array) => array.clear(),
            SsoHashMap::Map(map) => map.clear(),
        }
    }

    /// Returns the number of elements the map can hold without reallocating.
    pub fn capacity(&self) -> usize {
        match self {
            SsoHashMap::Array(_) => SSO_ARRAY_SIZE,
            SsoHashMap::Map(map) => map.capacity(),
        }
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        match self {
            SsoHashMap::Array(array) => array.len(),
            SsoHashMap::Map(map) => map.len(),
        }
    }

    /// Returns `true` if the map contains no elements.
    pub fn is_empty(&self) -> bool {
        match self {
            SsoHashMap::Array(array) => array.is_empty(),
            SsoHashMap::Map(map) => map.is_empty(),
        }
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    /// The iterator element type is `(&'a K, &'a V)`.
    #[inline]
    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    /// An iterator visiting all key-value pairs in arbitrary order,
    /// with mutable references to the values.
    /// The iterator element type is `(&'a K, &'a mut V)`.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&'_ K, &'_ mut V)> {
        self.into_iter()
    }

    /// An iterator visiting all keys in arbitrary order.
    /// The iterator element type is `&'a K`.
    pub fn keys(&self) -> impl Iterator<Item = &'_ K> {
        match self {
            SsoHashMap::Array(array) => EitherIter::Left(array.iter().map(|(k, _v)| k)),
            SsoHashMap::Map(map) => EitherIter::Right(map.keys()),
        }
    }

    /// An iterator visiting all values in arbitrary order.
    /// The iterator element type is `&'a V`.
    pub fn values(&self) -> impl Iterator<Item = &'_ V> {
        match self {
            SsoHashMap::Array(array) => EitherIter::Left(array.iter().map(|(_k, v)| v)),
            SsoHashMap::Map(map) => EitherIter::Right(map.values()),
        }
    }

    /// An iterator visiting all values mutably in arbitrary order.
    /// The iterator element type is `&'a mut V`.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &'_ mut V> {
        match self {
            SsoHashMap::Array(array) => EitherIter::Left(array.iter_mut().map(|(_k, v)| v)),
            SsoHashMap::Map(map) => EitherIter::Right(map.values_mut()),
        }
    }

    /// Clears the map, returning all key-value pairs as an iterator. Keeps the
    /// allocated memory for reuse.
    pub fn drain(&mut self) -> impl Iterator<Item = (K, V)> + '_ {
        match self {
            SsoHashMap::Array(array) => EitherIter::Left(array.drain(..)),
            SsoHashMap::Map(map) => EitherIter::Right(map.drain()),
        }
    }
}

impl<K: Eq + Hash, V> SsoHashMap<K, V> {
    /// Changes underlying storage from array to hashmap
    /// if array is full.
    fn migrate_if_full(&mut self) {
        if let SsoHashMap::Array(array) = self {
            if array.is_full() {
                *self = SsoHashMap::Map(array.drain(..).collect());
            }
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the `SsoHashMap`. The collection may reserve more space to avoid
    /// frequent reallocations.
    pub fn reserve(&mut self, additional: usize) {
        match self {
            SsoHashMap::Array(array) => {
                if SSO_ARRAY_SIZE < (array.len() + additional) {
                    let mut map: FxHashMap<K, V> = array.drain(..).collect();
                    map.reserve(additional);
                    *self = SsoHashMap::Map(map);
                }
            }
            SsoHashMap::Map(map) => map.reserve(additional),
        }
    }

    /// Shrinks the capacity of the map as much as possible. It will drop
    /// down as much as possible while maintaining the internal rules
    /// and possibly leaving some space in accordance with the resize policy.
    pub fn shrink_to_fit(&mut self) {
        if let SsoHashMap::Map(map) = self {
            if map.len() <= SSO_ARRAY_SIZE {
                *self = SsoHashMap::Array(map.drain().collect());
            } else {
                map.shrink_to_fit();
            }
        }
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        match self {
            SsoHashMap::Array(array) => array.retain(|(k, v)| f(k, v)),
            SsoHashMap::Map(map) => map.retain(f),
        }
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, [`None`] is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned. The key is not updated, though; this matters for
    /// types that can be `==` without being identical. See the [module-level
    /// documentation] for more.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self {
            SsoHashMap::Array(array) => {
                for (k, v) in array.iter_mut() {
                    if *k == key {
                        let old_value = std::mem::replace(v, value);
                        return Some(old_value);
                    }
                }
                if let Err(error) = array.try_push((key, value)) {
                    let mut map: FxHashMap<K, V> = array.drain(..).collect();
                    let (key, value) = error.element();
                    map.insert(key, value);
                    *self = SsoHashMap::Map(map);
                }
                None
            }
            SsoHashMap::Map(map) => map.insert(key, value),
        }
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        match self {
            SsoHashMap::Array(array) => {
                array.iter().position(|(k, _v)| k == key).map(|index| array.swap_remove(index).1)
            }
            SsoHashMap::Map(map) => map.remove(key),
        }
    }

    /// Removes a key from the map, returning the stored key and value if the
    /// key was previously in the map.
    pub fn remove_entry(&mut self, key: &K) -> Option<(K, V)> {
        match self {
            SsoHashMap::Array(array) => {
                array.iter().position(|(k, _v)| k == key).map(|index| array.swap_remove(index))
            }
            SsoHashMap::Map(map) => map.remove_entry(key),
        }
    }

    /// Returns a reference to the value corresponding to the key.
    pub fn get(&self, key: &K) -> Option<&V> {
        match self {
            SsoHashMap::Array(array) => {
                for (k, v) in array {
                    if k == key {
                        return Some(v);
                    }
                }
                None
            }
            SsoHashMap::Map(map) => map.get(key),
        }
    }

    /// Returns a mutable reference to the value corresponding to the key.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        match self {
            SsoHashMap::Array(array) => {
                for (k, v) in array {
                    if k == key {
                        return Some(v);
                    }
                }
                None
            }
            SsoHashMap::Map(map) => map.get_mut(key),
        }
    }

    /// Returns the key-value pair corresponding to the supplied key.
    pub fn get_key_value(&self, key: &K) -> Option<(&K, &V)> {
        match self {
            SsoHashMap::Array(array) => {
                for (k, v) in array {
                    if k == key {
                        return Some((k, v));
                    }
                }
                None
            }
            SsoHashMap::Map(map) => map.get_key_value(key),
        }
    }

    /// Returns `true` if the map contains a value for the specified key.
    pub fn contains_key(&self, key: &K) -> bool {
        match self {
            SsoHashMap::Array(array) => array.iter().any(|(k, _v)| k == key),
            SsoHashMap::Map(map) => map.contains_key(key),
        }
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    #[inline]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        Entry { ssomap: self, key }
    }
}

impl<K, V> Default for SsoHashMap<K, V> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash, V> FromIterator<(K, V)> for SsoHashMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> SsoHashMap<K, V> {
        let mut map: SsoHashMap<K, V> = Default::default();
        map.extend(iter);
        map
    }
}

impl<K: Eq + Hash, V> Extend<(K, V)> for SsoHashMap<K, V> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (K, V)>,
    {
        for (key, value) in iter.into_iter() {
            self.insert(key, value);
        }
    }

    #[inline]
    fn extend_one(&mut self, (k, v): (K, V)) {
        self.insert(k, v);
    }

    fn extend_reserve(&mut self, additional: usize) {
        match self {
            SsoHashMap::Array(array) => {
                if SSO_ARRAY_SIZE < (array.len() + additional) {
                    let mut map: FxHashMap<K, V> = array.drain(..).collect();
                    map.extend_reserve(additional);
                    *self = SsoHashMap::Map(map);
                }
            }
            SsoHashMap::Map(map) => map.extend_reserve(additional),
        }
    }
}

impl<'a, K, V> Extend<(&'a K, &'a V)> for SsoHashMap<K, V>
where
    K: Eq + Hash + Copy,
    V: Copy,
{
    fn extend<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: T) {
        self.extend(iter.into_iter().map(|(k, v)| (*k, *v)))
    }

    #[inline]
    fn extend_one(&mut self, (&k, &v): (&'a K, &'a V)) {
        self.insert(k, v);
    }

    #[inline]
    fn extend_reserve(&mut self, additional: usize) {
        Extend::<(K, V)>::extend_reserve(self, additional)
    }
}

impl<K, V> IntoIterator for SsoHashMap<K, V> {
    type IntoIter = EitherIter<
        <ArrayVec<(K, V), 8> as IntoIterator>::IntoIter,
        <FxHashMap<K, V> as IntoIterator>::IntoIter,
    >;
    type Item = <Self::IntoIter as Iterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            SsoHashMap::Array(array) => EitherIter::Left(array.into_iter()),
            SsoHashMap::Map(map) => EitherIter::Right(map.into_iter()),
        }
    }
}

/// adapts Item of array reference iterator to Item of hashmap reference iterator.
#[inline(always)]
fn adapt_array_ref_it<K, V>(pair: &(K, V)) -> (&K, &V) {
    let (a, b) = pair;
    (a, b)
}

/// adapts Item of array mut reference iterator to Item of hashmap mut reference iterator.
#[inline(always)]
fn adapt_array_mut_it<K, V>(pair: &mut (K, V)) -> (&K, &mut V) {
    let (a, b) = pair;
    (a, b)
}

impl<'a, K, V> IntoIterator for &'a SsoHashMap<K, V> {
    type IntoIter = EitherIter<
        std::iter::Map<
            <&'a ArrayVec<(K, V), 8> as IntoIterator>::IntoIter,
            fn(&'a (K, V)) -> (&'a K, &'a V),
        >,
        <&'a FxHashMap<K, V> as IntoIterator>::IntoIter,
    >;
    type Item = <Self::IntoIter as Iterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            SsoHashMap::Array(array) => EitherIter::Left(array.into_iter().map(adapt_array_ref_it)),
            SsoHashMap::Map(map) => EitherIter::Right(map.iter()),
        }
    }
}

impl<'a, K, V> IntoIterator for &'a mut SsoHashMap<K, V> {
    type IntoIter = EitherIter<
        std::iter::Map<
            <&'a mut ArrayVec<(K, V), 8> as IntoIterator>::IntoIter,
            fn(&'a mut (K, V)) -> (&'a K, &'a mut V),
        >,
        <&'a mut FxHashMap<K, V> as IntoIterator>::IntoIter,
    >;
    type Item = <Self::IntoIter as Iterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            SsoHashMap::Array(array) => EitherIter::Left(array.into_iter().map(adapt_array_mut_it)),
            SsoHashMap::Map(map) => EitherIter::Right(map.iter_mut()),
        }
    }
}

impl<K, V> fmt::Debug for SsoHashMap<K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<'a, K, V> Index<&'a K> for SsoHashMap<K, V>
where
    K: Eq + Hash,
{
    type Output = V;

    #[inline]
    fn index(&self, key: &K) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

/// A view into a single entry in a map.
pub struct Entry<'a, K, V> {
    ssomap: &'a mut SsoHashMap<K, V>,
    key: K,
}

impl<'a, K: Eq + Hash, V> Entry<'a, K, V> {
    /// Provides in-place mutable access to an occupied entry before any
    /// potential inserts into the map.
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        if let Some(value) = self.ssomap.get_mut(&self.key) {
            f(value);
        }
        self
    }

    /// Ensures a value is in the entry by inserting the default if empty, and returns
    /// a mutable reference to the value in the entry.
    #[inline]
    pub fn or_insert(self, value: V) -> &'a mut V {
        self.or_insert_with(|| value)
    }

    /// Ensures a value is in the entry by inserting the result of the default function if empty,
    /// and returns a mutable reference to the value in the entry.
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        self.ssomap.migrate_if_full();
        match self.ssomap {
            SsoHashMap::Array(array) => {
                let key_ref = &self.key;
                let found_index = array.iter().position(|(k, _v)| k == key_ref);
                let index = if let Some(index) = found_index {
                    index
                } else {
                    let index = array.len();
                    array.try_push((self.key, default())).unwrap();
                    index
                };
                &mut array[index].1
            }
            SsoHashMap::Map(map) => map.entry(self.key).or_insert_with(default),
        }
    }

    /// Returns a reference to this entry's key.
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }
}

impl<'a, K: Eq + Hash, V: Default> Entry<'a, K, V> {
    /// Ensures a value is in the entry by inserting the default value if empty,
    /// and returns a mutable reference to the value in the entry.
    #[inline]
    pub fn or_default(self) -> &'a mut V {
        self.or_insert_with(Default::default)
    }
}
