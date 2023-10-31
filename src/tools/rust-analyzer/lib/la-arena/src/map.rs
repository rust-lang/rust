use std::iter::Enumerate;
use std::marker::PhantomData;

use crate::Idx;

/// A map from arena indexes to some other type.
/// Space requirement is O(highest index).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArenaMap<IDX, V> {
    v: Vec<Option<V>>,
    _ty: PhantomData<IDX>,
}

impl<T, V> ArenaMap<Idx<T>, V> {
    /// Creates a new empty map.
    pub const fn new() -> Self {
        Self { v: Vec::new(), _ty: PhantomData }
    }

    /// Create a new empty map with specific capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self { v: Vec::with_capacity(capacity), _ty: PhantomData }
    }

    /// Reserves capacity for at least additional more elements to be inserted in the map.
    pub fn reserve(&mut self, additional: usize) {
        self.v.reserve(additional);
    }

    /// Clears the map, removing all elements.
    pub fn clear(&mut self) {
        self.v.clear();
    }

    /// Shrinks the capacity of the map as much as possible.
    pub fn shrink_to_fit(&mut self) {
        let min_len = self.v.iter().rposition(|slot| slot.is_some()).map_or(0, |i| i + 1);
        self.v.truncate(min_len);
        self.v.shrink_to_fit();
    }

    /// Returns whether the map contains a value for the specified index.
    pub fn contains_idx(&self, idx: Idx<T>) -> bool {
        matches!(self.v.get(Self::to_idx(idx)), Some(Some(_)))
    }

    /// Removes an index from the map, returning the value at the index if the index was previously in the map.
    pub fn remove(&mut self, idx: Idx<T>) -> Option<V> {
        self.v.get_mut(Self::to_idx(idx))?.take()
    }

    /// Inserts a value associated with a given arena index into the map.
    ///
    /// If the map did not have this index present, None is returned.
    /// Otherwise, the value is updated, and the old value is returned.
    pub fn insert(&mut self, idx: Idx<T>, t: V) -> Option<V> {
        let idx = Self::to_idx(idx);

        self.v.resize_with((idx + 1).max(self.v.len()), || None);
        self.v[idx].replace(t)
    }

    /// Returns a reference to the value associated with the provided index
    /// if it is present.
    pub fn get(&self, idx: Idx<T>) -> Option<&V> {
        self.v.get(Self::to_idx(idx)).and_then(|it| it.as_ref())
    }

    /// Returns a mutable reference to the value associated with the provided index
    /// if it is present.
    pub fn get_mut(&mut self, idx: Idx<T>) -> Option<&mut V> {
        self.v.get_mut(Self::to_idx(idx)).and_then(|it| it.as_mut())
    }

    /// Returns an iterator over the values in the map.
    pub fn values(&self) -> impl Iterator<Item = &V> + DoubleEndedIterator {
        self.v.iter().filter_map(|o| o.as_ref())
    }

    /// Returns an iterator over mutable references to the values in the map.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> + DoubleEndedIterator {
        self.v.iter_mut().filter_map(|o| o.as_mut())
    }

    /// Returns an iterator over the arena indexes and values in the map.
    pub fn iter(&self) -> impl Iterator<Item = (Idx<T>, &V)> + DoubleEndedIterator {
        self.v.iter().enumerate().filter_map(|(idx, o)| Some((Self::from_idx(idx), o.as_ref()?)))
    }

    /// Returns an iterator over the arena indexes and values in the map.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Idx<T>, &mut V)> {
        self.v
            .iter_mut()
            .enumerate()
            .filter_map(|(idx, o)| Some((Self::from_idx(idx), o.as_mut()?)))
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    pub fn entry(&mut self, idx: Idx<T>) -> Entry<'_, Idx<T>, V> {
        let idx = Self::to_idx(idx);
        self.v.resize_with((idx + 1).max(self.v.len()), || None);
        match &mut self.v[idx] {
            slot @ Some(_) => Entry::Occupied(OccupiedEntry { slot, _ty: PhantomData }),
            slot @ None => Entry::Vacant(VacantEntry { slot, _ty: PhantomData }),
        }
    }

    fn to_idx(idx: Idx<T>) -> usize {
        u32::from(idx.into_raw()) as usize
    }

    fn from_idx(idx: usize) -> Idx<T> {
        Idx::from_raw((idx as u32).into())
    }
}

impl<T, V> std::ops::Index<Idx<V>> for ArenaMap<Idx<V>, T> {
    type Output = T;
    fn index(&self, idx: Idx<V>) -> &T {
        self.v[Self::to_idx(idx)].as_ref().unwrap()
    }
}

impl<T, V> std::ops::IndexMut<Idx<V>> for ArenaMap<Idx<V>, T> {
    fn index_mut(&mut self, idx: Idx<V>) -> &mut T {
        self.v[Self::to_idx(idx)].as_mut().unwrap()
    }
}

impl<T, V> Default for ArenaMap<Idx<V>, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, V> Extend<(Idx<V>, T)> for ArenaMap<Idx<V>, T> {
    fn extend<I: IntoIterator<Item = (Idx<V>, T)>>(&mut self, iter: I) {
        iter.into_iter().for_each(move |(k, v)| {
            self.insert(k, v);
        });
    }
}

impl<T, V> FromIterator<(Idx<V>, T)> for ArenaMap<Idx<V>, T> {
    fn from_iter<I: IntoIterator<Item = (Idx<V>, T)>>(iter: I) -> Self {
        let mut this = Self::new();
        this.extend(iter);
        this
    }
}

pub struct ArenaMapIter<IDX, V> {
    iter: Enumerate<std::vec::IntoIter<Option<V>>>,
    _ty: PhantomData<IDX>,
}

impl<T, V> IntoIterator for ArenaMap<Idx<T>, V> {
    type Item = (Idx<T>, V);

    type IntoIter = ArenaMapIter<Idx<T>, V>;

    fn into_iter(self) -> Self::IntoIter {
        let iter = self.v.into_iter().enumerate();
        Self::IntoIter { iter, _ty: PhantomData }
    }
}

impl<T, V> ArenaMapIter<Idx<T>, V> {
    fn mapper((idx, o): (usize, Option<V>)) -> Option<(Idx<T>, V)> {
        Some((ArenaMap::<Idx<T>, V>::from_idx(idx), o?))
    }
}

impl<T, V> Iterator for ArenaMapIter<Idx<T>, V> {
    type Item = (Idx<T>, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        for next in self.iter.by_ref() {
            match Self::mapper(next) {
                Some(r) => return Some(r),
                None => continue,
            }
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T, V> DoubleEndedIterator for ArenaMapIter<Idx<T>, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        while let Some(next_back) = self.iter.next_back() {
            match Self::mapper(next_back) {
                Some(r) => return Some(r),
                None => continue,
            }
        }

        None
    }
}

/// A view into a single entry in a map, which may either be vacant or occupied.
///
/// This `enum` is constructed from the [`entry`] method on [`ArenaMap`].
///
/// [`entry`]: ArenaMap::entry
pub enum Entry<'a, IDX, V> {
    /// A vacant entry.
    Vacant(VacantEntry<'a, IDX, V>),
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, IDX, V>),
}

impl<'a, IDX, V> Entry<'a, IDX, V> {
    /// Ensures a value is in the entry by inserting the default if empty, and returns a mutable reference to
    /// the value in the entry.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Self::Vacant(ent) => ent.insert(default),
            Self::Occupied(ent) => ent.into_mut(),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default function if empty, and returns
    /// a mutable reference to the value in the entry.
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Self::Vacant(ent) => ent.insert(default()),
            Self::Occupied(ent) => ent.into_mut(),
        }
    }

    /// Provides in-place mutable access to an occupied entry before any potential inserts into the map.
    pub fn and_modify<F: FnOnce(&mut V)>(mut self, f: F) -> Self {
        if let Self::Occupied(ent) = &mut self {
            f(ent.get_mut());
        }
        self
    }
}

impl<'a, IDX, V> Entry<'a, IDX, V>
where
    V: Default,
{
    /// Ensures a value is in the entry by inserting the default value if empty, and returns a mutable reference
    /// to the value in the entry.
    pub fn or_default(self) -> &'a mut V {
        self.or_insert_with(Default::default)
    }
}

/// A view into an vacant entry in a [`ArenaMap`]. It is part of the [`Entry`] enum.
pub struct VacantEntry<'a, IDX, V> {
    slot: &'a mut Option<V>,
    _ty: PhantomData<IDX>,
}

impl<'a, IDX, V> VacantEntry<'a, IDX, V> {
    /// Sets the value of the entry with the `VacantEntry`’s key, and returns a mutable reference to it.
    pub fn insert(self, value: V) -> &'a mut V {
        self.slot.insert(value)
    }
}

/// A view into an occupied entry in a [`ArenaMap`]. It is part of the [`Entry`] enum.
pub struct OccupiedEntry<'a, IDX, V> {
    slot: &'a mut Option<V>,
    _ty: PhantomData<IDX>,
}

impl<'a, IDX, V> OccupiedEntry<'a, IDX, V> {
    /// Gets a reference to the value in the entry.
    pub fn get(&self) -> &V {
        self.slot.as_ref().expect("Occupied")
    }

    /// Gets a mutable reference to the value in the entry.
    pub fn get_mut(&mut self) -> &mut V {
        self.slot.as_mut().expect("Occupied")
    }

    /// Converts the entry into a mutable reference to its value.
    pub fn into_mut(self) -> &'a mut V {
        self.slot.as_mut().expect("Occupied")
    }

    /// Sets the value of the entry with the `OccupiedEntry`’s key, and returns the entry’s old value.
    pub fn insert(&mut self, value: V) -> V {
        self.slot.replace(value).expect("Occupied")
    }

    /// Takes the value of the entry out of the map, and returns it.
    pub fn remove(self) -> V {
        self.slot.take().expect("Occupied")
    }
}
