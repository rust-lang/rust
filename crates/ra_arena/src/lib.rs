//! Yet another index-based arena.

use std::{
    fmt,
    iter::FromIterator,
    marker::PhantomData,
    ops::{Index, IndexMut},
};

pub mod map;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RawId(u32);

impl From<RawId> for u32 {
    fn from(raw: RawId) -> u32 {
        raw.0
    }
}

impl From<u32> for RawId {
    fn from(id: u32) -> RawId {
        RawId(id)
    }
}

impl fmt::Debug for RawId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Display for RawId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Arena<ID: ArenaId, T> {
    data: Vec<T>,
    _ty: PhantomData<ID>,
}

impl<ID: ArenaId, T: fmt::Debug> fmt::Debug for Arena<ID, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("Arena").field("len", &self.len()).field("data", &self.data).finish()
    }
}

#[macro_export]
macro_rules! impl_arena_id {
    ($name:ident) => {
        impl $crate::ArenaId for $name {
            fn from_raw(raw: $crate::RawId) -> Self {
                $name(raw)
            }
            fn into_raw(self) -> $crate::RawId {
                self.0
            }
        }
    };
}

pub trait ArenaId {
    fn from_raw(raw: RawId) -> Self;
    fn into_raw(self) -> RawId;
}

impl<ID: ArenaId, T> Arena<ID, T> {
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn alloc(&mut self, value: T) -> ID {
        let id = RawId(self.data.len() as u32);
        self.data.push(value);
        ID::from_raw(id)
    }
    pub fn iter(&self) -> impl Iterator<Item = (ID, &T)> + ExactSizeIterator {
        self.data.iter().enumerate().map(|(idx, value)| (ID::from_raw(RawId(idx as u32)), value))
    }
}

impl<ID: ArenaId, T> Default for Arena<ID, T> {
    fn default() -> Arena<ID, T> {
        Arena { data: Vec::new(), _ty: PhantomData }
    }
}

impl<ID: ArenaId, T> Index<ID> for Arena<ID, T> {
    type Output = T;
    fn index(&self, idx: ID) -> &T {
        let idx = idx.into_raw().0 as usize;
        &self.data[idx]
    }
}

impl<ID: ArenaId, T> IndexMut<ID> for Arena<ID, T> {
    fn index_mut(&mut self, idx: ID) -> &mut T {
        let idx = idx.into_raw().0 as usize;
        &mut self.data[idx]
    }
}

impl<ID: ArenaId, T> FromIterator<T> for Arena<ID, T> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Arena { data: Vec::from_iter(iter), _ty: PhantomData }
    }
}
