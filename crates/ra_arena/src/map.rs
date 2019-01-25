//! A map from arena IDs to some other type. Space requirement is O(highest ID).

use std::marker::PhantomData;

use super::ArenaId;

/// A map from arena IDs to some other type. Space requirement is O(highest ID).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArenaMap<ID, T> {
    v: Vec<Option<T>>,
    _ty: PhantomData<ID>,
}

impl<ID: ArenaId, T> ArenaMap<ID, T> {
    pub fn insert(&mut self, id: ID, t: T) {
        let idx = Self::to_idx(id);
        if self.v.capacity() <= idx {
            self.v.reserve(idx + 1 - self.v.capacity());
        }
        if self.v.len() <= idx {
            while self.v.len() <= idx {
                self.v.push(None);
            }
        }
        self.v[idx] = Some(t);
    }

    pub fn get(&self, id: ID) -> Option<&T> {
        self.v.get(Self::to_idx(id)).and_then(|it| it.as_ref())
    }

    pub fn get_mut(&mut self, id: ID) -> Option<&mut T> {
        self.v.get_mut(Self::to_idx(id)).and_then(|it| it.as_mut())
    }

    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.v.iter().filter_map(|o| o.as_ref())
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.v.iter_mut().filter_map(|o| o.as_mut())
    }

    pub fn iter(&self) -> impl Iterator<Item = (ID, &T)> {
        self.v
            .iter()
            .enumerate()
            .filter_map(|(idx, o)| Some((Self::from_idx(idx), o.as_ref()?)))
    }

    fn to_idx(id: ID) -> usize {
        u32::from(id.into_raw()) as usize
    }

    fn from_idx(idx: usize) -> ID {
        ID::from_raw((idx as u32).into())
    }
}

impl<ID: ArenaId, T> std::ops::Index<ID> for ArenaMap<ID, T> {
    type Output = T;
    fn index(&self, id: ID) -> &T {
        self.v[Self::to_idx(id)].as_ref().unwrap()
    }
}

impl<ID, T> Default for ArenaMap<ID, T> {
    fn default() -> Self {
        ArenaMap {
            v: Vec::new(),
            _ty: PhantomData,
        }
    }
}
