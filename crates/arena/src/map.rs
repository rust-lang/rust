//! A map from arena IDs to some other type. Space requirement is O(highest ID).

use std::marker::PhantomData;

use crate::Idx;

/// A map from arena IDs to some other type. Space requirement is O(highest ID).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArenaMap<ID, V> {
    v: Vec<Option<V>>,
    _ty: PhantomData<ID>,
}

impl<T, V> ArenaMap<Idx<T>, V> {
    pub fn insert(&mut self, id: Idx<T>, t: V) {
        let idx = Self::to_idx(id);

        self.v.resize_with((idx + 1).max(self.v.len()), || None);
        self.v[idx] = Some(t);
    }

    pub fn get(&self, id: Idx<T>) -> Option<&V> {
        self.v.get(Self::to_idx(id)).and_then(|it| it.as_ref())
    }

    pub fn get_mut(&mut self, id: Idx<T>) -> Option<&mut V> {
        self.v.get_mut(Self::to_idx(id)).and_then(|it| it.as_mut())
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.v.iter().filter_map(|o| o.as_ref())
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.v.iter_mut().filter_map(|o| o.as_mut())
    }

    pub fn iter(&self) -> impl Iterator<Item = (Idx<T>, &V)> {
        self.v.iter().enumerate().filter_map(|(idx, o)| Some((Self::from_idx(idx), o.as_ref()?)))
    }

    fn to_idx(id: Idx<T>) -> usize {
        u32::from(id.into_raw()) as usize
    }

    fn from_idx(idx: usize) -> Idx<T> {
        Idx::from_raw((idx as u32).into())
    }
}

impl<T, V> std::ops::Index<Idx<V>> for ArenaMap<Idx<V>, T> {
    type Output = T;
    fn index(&self, id: Idx<V>) -> &T {
        self.v[Self::to_idx(id)].as_ref().unwrap()
    }
}

impl<T, V> Default for ArenaMap<Idx<V>, T> {
    fn default() -> Self {
        ArenaMap { v: Vec::new(), _ty: PhantomData }
    }
}
