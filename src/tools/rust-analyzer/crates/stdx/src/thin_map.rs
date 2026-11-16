//! A map that is optimized for having few elements: when the length is low, it is stored as a `ThinVec`.
#![deny(unsafe_code)]

mod set;
#[cfg(test)]
mod tests;

use std::{
    borrow::Borrow,
    hash::{BuildHasher, Hash},
    mem,
};

use thin_vec::ThinVec;

pub use self::set::{ThinHashSet, ThinSet};
pub use self::unsafe_encapsulation::ThinMap;

#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "salsa", derive(salsa::SalsaValue))]
#[repr(align(2))]
struct Align<T>(T);

type ThinReprItem<K, V> = Align<(K, V)>;
type ThinReprVec<K, V> = ThinVec<ThinReprItem<K, V>>;

enum ReprRef<'a, K, V, M> {
    Thin(&'a [ThinReprItem<K, V>]),
    Map(&'a M),
}

enum ReprMut<'a, K, V, M> {
    Thin(&'a mut ThinReprVec<K, V>),
    Map(&'a mut M),
}

#[cfg_attr(feature = "salsa", derive(salsa::SalsaValue))]
enum Repr<K, V, M> {
    Thin(ThinReprVec<K, V>),
    Map(Box<Align<M>>),
}

pub enum MapEntry<'a, K, V, M: Map<K, V> + 'a> {
    Occupied(M::OccupiedEntry<'a>),
    Vacant(M::VacantEntry<'a>),
}

// We use generic parameters and not associated types because this allows us to keep `ThinMap` covariant over `M`
// (associated types are invariant).
pub trait Map<K, V>: Default {
    fn with_capacity(capacity: usize) -> Self;
    fn reserve(&mut self, additional: usize);
    fn len(&self) -> usize;
    fn shrink_to_fit(&mut self);

    type Iter<'a>: ExactSizeIterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
        Self: 'a;
    type IterMut<'a>: ExactSizeIterator<Item = (&'a K, &'a mut V)>
    where
        K: 'a,
        V: 'a,
        Self: 'a;
    type IntoIter: ExactSizeIterator<Item = (K, V)>;

    fn iter(&self) -> Self::Iter<'_>;
    fn iter_mut(&mut self) -> Self::IterMut<'_>;
    fn into_iter(self) -> Self::IntoIter;

    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq;
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq;
    fn get_disjoint_mut<Q, const N: usize>(&mut self, ks: [&Q; N]) -> [Option<&mut V>; N]
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq;

    fn insert(&mut self, key: K, value: V) -> Option<V>;
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq;

    type OccupiedEntry<'a>
    where
        Self: 'a;
    type VacantEntry<'a>
    where
        Self: 'a;

    fn entry(&mut self, key: K) -> MapEntry<'_, K, V, Self>;
    fn vacant_entry_insert<'a>(entry: Self::VacantEntry<'a>, value: V) -> &'a mut V
    where
        Self: 'a;
    fn occupied_entry_get<'a, 'b>(entry: &'a Self::OccupiedEntry<'b>) -> &'a V
    where
        Self: 'b;
    fn occupied_entry_get_mut<'a, 'b>(entry: &'a mut Self::OccupiedEntry<'b>) -> &'a mut V
    where
        Self: 'b;
    fn occupied_entry_into_mut<'a>(entry: Self::OccupiedEntry<'a>) -> &'a mut V
    where
        Self: 'a;
    fn occupied_entry_insert<'a>(entry: &mut Self::OccupiedEntry<'a>, value: V) -> V
    where
        Self: 'a;
}

#[expect(clippy::disallowed_types)]
impl<K: Hash + Eq, V, S: BuildHasher + Default> Map<K, V> for std::collections::HashMap<K, V, S> {
    #[inline]
    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, S::default())
    }
    #[inline]
    fn reserve(&mut self, additional: usize) {
        self.reserve(additional);
    }
    #[inline]
    fn len(&self) -> usize {
        self.len()
    }
    #[inline]
    fn shrink_to_fit(&mut self) {
        self.shrink_to_fit();
    }

    type Iter<'a>
        = std::collections::hash_map::Iter<'a, K, V>
    where
        Self: 'a;
    type IterMut<'a>
        = std::collections::hash_map::IterMut<'a, K, V>
    where
        Self: 'a;
    type IntoIter = std::collections::hash_map::IntoIter<K, V>;

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }
    #[inline]
    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self.iter_mut()
    }
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIterator::into_iter(self)
    }

    #[inline]
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.get(key)
    }
    #[inline]
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.get_mut(key)
    }
    #[inline]
    fn get_disjoint_mut<Q, const N: usize>(&mut self, ks: [&Q; N]) -> [Option<&mut V>; N]
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.get_disjoint_mut(ks)
    }

    #[inline]
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.insert(key, value)
    }
    #[inline]
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.remove(key)
    }

    type OccupiedEntry<'a>
        = std::collections::hash_map::OccupiedEntry<'a, K, V>
    where
        Self: 'a;
    type VacantEntry<'a>
        = std::collections::hash_map::VacantEntry<'a, K, V>
    where
        Self: 'a;

    #[inline]
    fn entry(&mut self, key: K) -> MapEntry<'_, K, V, Self> {
        match self.entry(key) {
            std::collections::hash_map::Entry::Occupied(it) => MapEntry::Occupied(it),
            std::collections::hash_map::Entry::Vacant(it) => MapEntry::Vacant(it),
        }
    }
    #[inline]
    fn vacant_entry_insert<'a>(entry: Self::VacantEntry<'a>, value: V) -> &'a mut V
    where
        Self: 'a,
    {
        entry.insert(value)
    }
    #[inline]
    fn occupied_entry_get<'a, 'b>(entry: &'a Self::OccupiedEntry<'b>) -> &'a V
    where
        Self: 'b,
    {
        entry.get()
    }
    #[inline]
    fn occupied_entry_get_mut<'a, 'b>(entry: &'a mut Self::OccupiedEntry<'b>) -> &'a mut V
    where
        Self: 'b,
    {
        entry.get_mut()
    }
    #[inline]
    fn occupied_entry_into_mut<'a>(entry: Self::OccupiedEntry<'a>) -> &'a mut V
    where
        Self: 'a,
    {
        entry.into_mut()
    }
    #[inline]
    fn occupied_entry_insert<'a>(entry: &mut Self::OccupiedEntry<'a>, value: V) -> V
    where
        Self: 'a,
    {
        entry.insert(value)
    }
}

#[expect(clippy::disallowed_types)]
pub type ThinHashMap<K, V, S> = ThinMap<K, V, std::collections::HashMap<K, V, S>>;

/// The maximum len to still use `ThinVec`.
const LEN_CUTOFF: usize = 10;

impl<K: Hash + Eq, V, M: Map<K, V>> ThinMap<K, V, M> {
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let repr = if capacity > LEN_CUTOFF {
            Repr::Map(Box::new(Align(M::with_capacity(capacity))))
        } else {
            Repr::Thin(ThinVec::with_capacity(capacity))
        };
        Self::from_repr(repr)
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        match self.repr_mut() {
            ReprMut::Thin(repr) => {
                let expected_len = repr.len() + additional;
                if expected_len > repr.capacity() || expected_len > LEN_CUTOFF {
                    self.reserve_cold(expected_len);
                }
            }
            ReprMut::Map(repr) => repr.reserve(additional),
        }
    }

    #[cold]
    fn reserve_cold(&mut self, expected_len: usize) {
        let ReprMut::Thin(repr) = self.repr_mut() else { unreachable!() };
        if expected_len <= LEN_CUTOFF {
            repr.reserve(expected_len - repr.len());
        } else {
            let mut map = M::with_capacity(expected_len);
            for Align((key, value)) in mem::take(repr) {
                map.insert(key, value);
            }
            *self = Self::from_repr(Repr::Map(Box::new(Align(map))));
        }
    }

    pub fn shrink_to_fit(&mut self) {
        let mut repr = mem::take(self).into_repr();
        match repr {
            Repr::Thin(ref mut repr) => repr.shrink_to_fit(),
            Repr::Map(mut map) => {
                repr = if map.0.len() > LEN_CUTOFF {
                    map.0.shrink_to_fit();
                    Repr::Map(map)
                } else {
                    Repr::Thin(map.0.into_iter().map(Align).collect())
                };
            }
        }
        *self = Self::from_repr(repr);
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self.repr_ref() {
            ReprRef::Thin(repr) => repr.len(),
            ReprRef::Map(repr) => repr.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V, M> {
        let repr = match self.repr_ref() {
            ReprRef::Thin(repr) => IterRepr::Thin(repr.iter()),
            ReprRef::Map(repr) => IterRepr::Map(repr.iter()),
        };
        Iter { repr }
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V, M> {
        let repr = match self.repr_mut() {
            ReprMut::Thin(repr) => IterMutRepr::Thin(repr.iter_mut()),
            ReprMut::Map(repr) => IterMutRepr::Map(repr.iter_mut()),
        };
        IterMut { repr }
    }

    #[inline]
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &K> {
        self.iter().map(|(key, _value)| key)
    }

    #[inline]
    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> {
        self.iter().map(|(_key, value)| value)
    }

    #[inline]
    pub fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut V> {
        self.iter_mut().map(|(_key, value)| value)
    }

    #[inline]
    pub fn into_keys(self) -> impl ExactSizeIterator<Item = K> {
        self.into_iter().map(|(key, _value)| key)
    }

    #[inline]
    pub fn into_values(self) -> impl ExactSizeIterator<Item = V> {
        self.into_iter().map(|(_key, value)| value)
    }

    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.get(key).is_some()
    }

    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        match self.repr_ref() {
            ReprRef::Thin(repr) => {
                repr.iter().find_map(|Align((k, v))| (k.borrow() == key).then_some(v))
            }
            ReprRef::Map(repr) => repr.get(key),
        }
    }

    #[inline]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        match self.repr_mut() {
            ReprMut::Thin(repr) => {
                repr.iter_mut().find_map(|Align((k, v))| ((*k).borrow() == key).then_some(v))
            }
            ReprMut::Map(repr) => repr.get_mut(key),
        }
    }

    pub fn get_disjoint_mut<Q, const N: usize>(&mut self, ks: [&Q; N]) -> [Option<&mut V>; N]
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        match self.repr_mut() {
            ReprMut::Thin(repr) => {
                let mut result = [const { None }; N];
                for Align((key, value)) in repr {
                    for (&k, v) in std::iter::zip(&ks, &mut result) {
                        if (*key).borrow() == k {
                            *v = Some(value);
                            break;
                        }
                    }
                }
                result
            }
            ReprMut::Map(repr) => repr.get_disjoint_mut(ks),
        }
    }

    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.repr_mut() {
            ReprMut::Thin(repr) => {
                let existing = repr.iter_mut().find(|Align((k, _v))| *k == key);
                match existing {
                    Some(Align((_k, v))) => Some(mem::replace(v, value)),
                    None => {
                        if repr.len() < LEN_CUTOFF {
                            repr.push(Align((key, value)));
                        } else {
                            self.insert_cold(key, value);
                        }
                        None
                    }
                }
            }
            ReprMut::Map(repr) => repr.insert(key, value),
        }
    }

    #[inline]
    #[expect(unsafe_code)]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, M> {
        let this_ptr = self as *mut Self;
        match self.repr_mut() {
            ReprMut::Thin(repr) => {
                let existing = repr.iter().position(|Align((k, _v))| *k == key);
                match existing {
                    Some(existing) => Entry::Occupied(OccupiedEntry {
                        repr: OccupiedEntryRepr::Thin { slice: repr, index: existing },
                    }),
                    None => {
                        if repr.len() < LEN_CUTOFF {
                            Entry::Vacant(VacantEntry {
                                repr: VacantEntryRepr::ThinHasSpace(key, repr),
                            })
                        } else {
                            // SAFETY: Polonius workaround.
                            Entry::Vacant(VacantEntry {
                                repr: VacantEntryRepr::ThinNoSpace(key, unsafe { &mut *this_ptr }),
                            })
                        }
                    }
                }
            }
            ReprMut::Map(repr) => match repr.entry(key) {
                MapEntry::Occupied(it) => {
                    Entry::Occupied(OccupiedEntry { repr: OccupiedEntryRepr::Map(it) })
                }
                MapEntry::Vacant(it) => {
                    Entry::Vacant(VacantEntry { repr: VacantEntryRepr::Map(it) })
                }
            },
        }
    }

    #[cold]
    fn insert_cold(&mut self, new_key: K, new_value: V) -> &mut V {
        let old_repr = match self.repr_mut() {
            ReprMut::Thin(repr) => mem::take(repr),
            ReprMut::Map(_) => unreachable!(),
        };
        let map = M::with_capacity(old_repr.len() + 1);
        *self = Self::from_repr(Repr::Map(Box::new(Align(map))));
        let map = match self.repr_mut() {
            ReprMut::Map(map) => map,
            ReprMut::Thin(_) => unreachable!(),
        };
        for Align((old_key, old_value)) in old_repr {
            map.insert(old_key, old_value);
        }
        match M::entry(map, new_key) {
            MapEntry::Vacant(entry) => M::vacant_entry_insert(entry, new_value),
            MapEntry::Occupied(_) => panic!("non-existing key somehow compares equal?!"),
        }
    }

    #[inline]
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        match self.repr_mut() {
            ReprMut::Thin(repr) => {
                let index = repr.iter().position(|Align((k, _v))| k.borrow() == key);
                index.map(|index| repr.swap_remove(index).0.1)
            }
            ReprMut::Map(repr) => repr.remove(key),
        }
    }
}

enum OccupiedEntryRepr<'a, K, V, M: Map<K, V> + 'a> {
    Thin { slice: &'a mut [Align<(K, V)>], index: usize },
    Map(M::OccupiedEntry<'a>),
}

enum VacantEntryRepr<'a, K, V, M: Map<K, V>> {
    ThinHasSpace(K, &'a mut ThinReprVec<K, V>),
    ThinNoSpace(K, &'a mut ThinMap<K, V, M>),
    Map(M::VacantEntry<'a>),
}

pub struct OccupiedEntry<'a, K, V, M: Map<K, V>> {
    repr: OccupiedEntryRepr<'a, K, V, M>,
}

pub struct VacantEntry<'a, K, V, M: Map<K, V>> {
    repr: VacantEntryRepr<'a, K, V, M>,
}

pub enum Entry<'a, K, V, M: Map<K, V>> {
    Occupied(OccupiedEntry<'a, K, V, M>),
    Vacant(VacantEntry<'a, K, V, M>),
}

impl<'a, K: Hash + Eq, V, M: Map<K, V>> VacantEntry<'a, K, V, M> {
    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        match self.repr {
            VacantEntryRepr::ThinHasSpace(key, repr) => {
                repr.push(Align((key, value)));
                &mut repr.last_mut().unwrap().0.1
            }
            VacantEntryRepr::ThinNoSpace(key, repr) => repr.insert_cold(key, value),
            VacantEntryRepr::Map(repr) => M::vacant_entry_insert(repr, value),
        }
    }
}

impl<'a, K: Hash + Eq, V, M: Map<K, V>> OccupiedEntry<'a, K, V, M> {
    #[inline]
    pub fn get(&self) -> &V {
        match &self.repr {
            OccupiedEntryRepr::Thin { slice, index } => &slice[*index].0.1,
            OccupiedEntryRepr::Map(repr) => M::occupied_entry_get(repr),
        }
    }

    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        match &mut self.repr {
            OccupiedEntryRepr::Thin { slice, index } => &mut slice[*index].0.1,
            OccupiedEntryRepr::Map(repr) => M::occupied_entry_get_mut(repr),
        }
    }

    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        match self.repr {
            OccupiedEntryRepr::Thin { slice, index } => &mut slice[index].0.1,
            OccupiedEntryRepr::Map(repr) => M::occupied_entry_into_mut(repr),
        }
    }

    #[inline]
    pub fn insert(&mut self, value: V) -> V {
        match &mut self.repr {
            OccupiedEntryRepr::Thin { slice, index } => mem::replace(&mut slice[*index].0.1, value),
            OccupiedEntryRepr::Map(repr) => M::occupied_entry_insert(repr, value),
        }
    }
}

impl<Q, K, V, M> std::ops::Index<&Q> for ThinMap<K, V, M>
where
    Q: ?Sized + Hash + Eq,
    K: Borrow<Q> + Hash + Eq,
    M: Map<K, V>,
{
    type Output = V;

    #[inline]
    #[track_caller]
    fn index(&self, index: &Q) -> &Self::Output {
        self.get(index).expect("missing key")
    }
}

impl<'a, K: Hash + Eq, V, M: Map<K, V>> Entry<'a, K, V, M> {
    #[inline]
    pub fn or_insert_with(self, default: impl FnOnce() -> V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => {
                drop(default);
                entry.into_mut()
            }
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }

    #[inline]
    pub fn or_insert(self, default: V) -> &'a mut V {
        self.or_insert_with(move || default)
    }

    #[inline]
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }
}

impl<K, V, M: Map<K, V>> Default for ThinMap<K, V, M> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, M> std::fmt::Debug for ThinMap<K, V, M>
where
    M: Map<K, V>,
    K: Hash + Eq + std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self).finish()
    }
}

impl<K, V, M> Clone for ThinMap<K, V, M>
where
    M: Map<K, V> + Clone,
    K: Clone,
    V: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        let repr = match self.repr_ref() {
            ReprRef::Thin(repr) => Repr::Thin(ThinVec::from(repr)),
            ReprRef::Map(repr) => Repr::Map(Box::new(Align(repr.clone()))),
        };
        Self::from_repr(repr)
    }
}

impl<K, V, M> PartialEq for ThinMap<K, V, M>
where
    M: Map<K, V> + PartialEq,
    K: Hash + Eq,
    V: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self.repr_ref(), other.repr_ref()) {
            (ReprRef::Map(this), ReprRef::Map(other)) => {
                this.len() == other.len()
                    && other.iter().all(|(key, value)| this.get(key) == Some(value))
            }
            (ReprRef::Map(map), ReprRef::Thin(thin)) | (ReprRef::Thin(thin), ReprRef::Map(map)) => {
                thin.len() == map.len()
                    && thin.iter().all(|Align((key, value))| map.get(key) == Some(value))
            }
            (ReprRef::Thin(this), ReprRef::Thin(other)) => {
                this.len() == other.len() && this.iter().all(|this_item| other.contains(this_item))
            }
        }
    }
}

impl<K, V, M> Eq for ThinMap<K, V, M>
where
    M: Map<K, V> + Eq,
    K: Hash + Eq,
    V: Eq,
{
}

impl<'a, K: Hash + Eq, V, M: Map<K, V>> IntoIterator for &'a ThinMap<K, V, M> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V, M>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Hash + Eq, V, M: Map<K, V>> IntoIterator for &'a mut ThinMap<K, V, M> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V, M>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K, V, M: Map<K, V>> IntoIterator for ThinMap<K, V, M> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, M>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let repr = match self.into_repr() {
            Repr::Thin(repr) => IntoIterRepr::Thin(repr.into_iter()),
            Repr::Map(repr) => IntoIterRepr::Map(repr.0.into_iter()),
        };
        IntoIter { repr }
    }
}

enum IterRepr<'a, K, V, M: Map<K, V> + 'a> {
    Thin(std::slice::Iter<'a, Align<(K, V)>>),
    Map(M::Iter<'a>),
}

enum IterMutRepr<'a, K, V, M: Map<K, V> + 'a> {
    Thin(std::slice::IterMut<'a, Align<(K, V)>>),
    Map(M::IterMut<'a>),
}

enum IntoIterRepr<K, V, M: Map<K, V>> {
    Thin(thin_vec::IntoIter<Align<(K, V)>>),
    Map(M::IntoIter),
}

macro_rules! impl_iter_methods {
    ( $repr:ident, $map:expr ) => {
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            match &mut self.repr {
                $repr::Thin(repr) => repr.next().map($map),
                $repr::Map(repr) => repr.next(),
            }
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = match &self.repr {
                $repr::Thin(repr) => repr.len(),
                $repr::Map(repr) => repr.len(),
            };
            (len, Some(len))
        }

        #[inline]
        fn fold<B, F>(self, init: B, mut f: F) -> B
        where
            Self: Sized,
            F: FnMut(B, Self::Item) -> B,
        {
            match self.repr {
                $repr::Thin(repr) => repr.fold(init, move |init, item| f(init, $map(item))),
                $repr::Map(repr) => repr.fold(init, f),
            }
        }

        #[inline]
        fn for_each<F>(self, mut f: F)
        where
            Self: Sized,
            F: FnMut(Self::Item),
        {
            match self.repr {
                $repr::Thin(repr) => repr.for_each(move |item| f($map(item))),
                $repr::Map(repr) => repr.for_each(f),
            }
        }

        #[inline]
        fn all<F>(&mut self, mut f: F) -> bool
        where
            Self: Sized,
            F: FnMut(Self::Item) -> bool,
        {
            match &mut self.repr {
                $repr::Thin(repr) => repr.all(move |item| f($map(item))),
                $repr::Map(repr) => repr.all(f),
            }
        }

        #[inline]
        fn any<F>(&mut self, mut f: F) -> bool
        where
            Self: Sized,
            F: FnMut(Self::Item) -> bool,
        {
            match &mut self.repr {
                $repr::Thin(repr) => repr.any(move |item| f($map(item))),
                $repr::Map(repr) => repr.any(f),
            }
        }
    };
}

pub struct Iter<'a, K, V, M: Map<K, V>> {
    repr: IterRepr<'a, K, V, M>,
}

impl<'a, K, V, M: Map<K, V>> Iterator for Iter<'a, K, V, M> {
    type Item = (&'a K, &'a V);

    impl_iter_methods!(IterRepr, |item: &'a Align<(K, V)>| (&item.0.0, &item.0.1));
}

impl<K, V, M: Map<K, V>> ExactSizeIterator for Iter<'_, K, V, M> {}

pub struct IterMut<'a, K, V, M: Map<K, V>> {
    repr: IterMutRepr<'a, K, V, M>,
}

impl<'a, K, V, M: Map<K, V>> Iterator for IterMut<'a, K, V, M> {
    type Item = (&'a K, &'a mut V);

    impl_iter_methods!(IterMutRepr, |item: &'a mut Align<(K, V)>| (&item.0.0, &mut item.0.1));
}

impl<K, V, M: Map<K, V>> ExactSizeIterator for IterMut<'_, K, V, M> {}

pub struct IntoIter<K, V, M: Map<K, V>> {
    repr: IntoIterRepr<K, V, M>,
}

impl<K, V, M: Map<K, V>> Iterator for IntoIter<K, V, M> {
    type Item = (K, V);

    impl_iter_methods!(IntoIterRepr, |item: Align<(K, V)>| item.0);
}

impl<K, V, M: Map<K, V>> ExactSizeIterator for IntoIter<K, V, M> {}

impl<K: Hash + Eq, V, M: Map<K, V>> FromIterator<(K, V)> for ThinMap<K, V, M> {
    #[inline]
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut result = Self::new();
        result.extend(iter);
        result
    }
}

impl<K: Hash + Eq, V, M: Map<K, V>> Extend<(K, V)> for ThinMap<K, V, M> {
    #[inline]
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        iter.for_each(|(key, value)| {
            self.insert(key, value);
        });
    }
}

#[expect(unsafe_code)]
mod unsafe_encapsulation {
    use std::{
        marker::PhantomData,
        mem::{self, ManuallyDrop},
        ptr::NonNull,
    };

    use thin_vec::ThinVec;

    use super::{Align, Repr, ReprMut, ReprRef, ThinReprVec};

    pub struct ThinMap<K, V, M> {
        // INVARIANT: Holds a transmuted `ThinReprVec` or a `Box<Align<M>>` with its LSB set.
        ptr: NonNull<()>,
        _repr: PhantomData<Repr<K, V, M>>,
    }

    // SAFETY: We essentially own `Repr<K, V, M>`.
    unsafe impl<K, V, M> Send for ThinMap<K, V, M> where Repr<K, V, M>: Send {}

    unsafe impl<K, V, M> Sync for ThinMap<K, V, M> where Repr<K, V, M>: Sync {}

    #[cfg(feature = "salsa")]
    unsafe impl<K, V, M> salsa::SalsaValue for ThinMap<K, V, M> where Repr<K, V, M>: salsa::SalsaValue {}

    impl<K, V, M> ThinMap<K, V, M> {
        const TAG: usize = 0b1;

        #[inline]
        pub(super) fn repr_ref<'a>(&'a self) -> ReprRef<'a, K, V, M> {
            let tag = self.ptr.addr().get() & Self::TAG;
            // SAFETY: Per our invariant. `ThinVec` is layout-equivalent to a pointer.
            // FIXME: Technically `ThinVec`'s layout isn't guaranteed unless we enable the `gecko-ffi` feature. Should we enable it?
            unsafe {
                match tag {
                    0 => {
                        let thin_vec =
                            mem::transmute::<&'a NonNull<()>, &'a ThinReprVec<K, V>>(&self.ptr);
                        ReprRef::Thin(thin_vec)
                    }
                    1 => ReprRef::Map(
                        &(*self.ptr.as_ptr().map_addr(|addr| addr & !Self::TAG).cast::<Align<M>>())
                            .0,
                    ),
                    _ => unreachable!(),
                }
            }
        }

        #[inline]
        pub(super) fn repr_mut<'a>(&'a mut self) -> ReprMut<'a, K, V, M> {
            let tag = self.ptr.addr().get() & Self::TAG;
            // SAFETY: Per our invariant. `ThinVec` is layout-equivalent to a pointer.
            unsafe {
                match tag {
                    0 => {
                        let thin_vec = mem::transmute::<
                            &'a mut NonNull<()>,
                            &'a mut ThinReprVec<K, V>,
                        >(&mut self.ptr);
                        ReprMut::Thin(thin_vec)
                    }
                    1 => ReprMut::Map(
                        &mut (*self
                            .ptr
                            .as_ptr()
                            .map_addr(|addr| addr & !Self::TAG)
                            .cast::<Align<M>>())
                        .0,
                    ),
                    _ => unreachable!(),
                }
            }
        }

        #[inline]
        pub(super) fn into_repr(self) -> Repr<K, V, M> {
            let this = ManuallyDrop::new(self);
            let tag = this.ptr.addr().get() & Self::TAG;
            // SAFETY: Per our invariant. `ThinVec` is layout-equivalent to a pointer.
            unsafe {
                match tag {
                    0 => Repr::Thin(mem::transmute::<NonNull<()>, ThinReprVec<K, V>>(this.ptr)),
                    1 => Repr::Map(Box::from_raw(
                        this.ptr.as_ptr().map_addr(|addr| addr & !Self::TAG).cast::<Align<M>>(),
                    )),
                    _ => unreachable!(),
                }
            }
        }

        #[inline]
        pub(super) fn from_repr(repr: Repr<K, V, M>) -> Self {
            let ptr = match repr {
                // SAFETY: `ThinVec` is layout-equivalent to a pointer.
                Repr::Thin(repr) => unsafe {
                    mem::transmute::<ThinReprVec<K, V>, NonNull<()>>(repr)
                },
                // SAFETY: `Box::into_raw()` never returns NULL.
                // FIXME: Switch to `Box::into_non_null()` once stable.
                Repr::Map(repr) => unsafe {
                    NonNull::new_unchecked(
                        Box::into_raw(repr).cast::<()>().map_addr(|addr| addr | 1),
                    )
                },
            };
            // INVARIANT: We keep it properly.
            Self { ptr, _repr: PhantomData }
        }

        #[inline]
        pub const fn new() -> Self {
            // INVARIANT: We keep it properly.
            let ptr = unsafe { mem::transmute::<ThinReprVec<K, V>, NonNull<()>>(ThinVec::new()) };
            Self { ptr, _repr: PhantomData }
        }
    }

    impl<K, V, M> Drop for ThinMap<K, V, M> {
        #[inline]
        fn drop(&mut self) {
            // INVARIANT: This is a copy of `self` only to drop it.
            drop(ThinMap::<K, V, M> { ptr: self.ptr, _repr: PhantomData }.into_repr());
        }
    }
}
