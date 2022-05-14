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
    /// Inserts a value associated with a given arena index into the map.
    pub fn insert(&mut self, idx: Idx<T>, t: V) {
        let idx = Self::to_idx(idx);

        self.v.resize_with((idx + 1).max(self.v.len()), || None);
        self.v[idx] = Some(t);
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
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.v.iter().filter_map(|o| o.as_ref())
    }

    /// Returns an iterator over mutable references to the values in the map.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.v.iter_mut().filter_map(|o| o.as_mut())
    }

    /// Returns an iterator over the arena indexes and values in the map.
    pub fn iter(&self) -> impl Iterator<Item = (Idx<T>, &V)> {
        self.v.iter().enumerate().filter_map(|(idx, o)| Some((Self::from_idx(idx), o.as_ref()?)))
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
        ArenaMap { v: Vec::new(), _ty: PhantomData }
    }
}
