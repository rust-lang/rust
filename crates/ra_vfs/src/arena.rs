use std::{
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Index, IndexMut},
};

#[derive(Clone, Debug)]
pub(crate) struct Arena<ID: ArenaId, T> {
    data: Vec<T>,
    _ty: PhantomData<ID>,
}

pub(crate) trait ArenaId {
    fn from_u32(id: u32) -> Self;
    fn to_u32(self) -> u32;
}

impl<ID: ArenaId, T> Arena<ID, T> {
    pub fn alloc(&mut self, value: T) -> ID {
        let id = self.data.len() as u32;
        self.data.push(value);
        ID::from_u32(id)
    }
}

impl<ID: ArenaId, T> Default for Arena<ID, T> {
    fn default() -> Arena<ID, T> {
        Arena {
            data: Vec::new(),
            _ty: PhantomData,
        }
    }
}

impl<ID: ArenaId, T> Index<ID> for Arena<ID, T> {
    type Output = T;
    fn index(&self, idx: ID) -> &T {
        let idx = idx.to_u32() as usize;
        &self.data[idx]
    }
}

impl<ID: ArenaId, T> IndexMut<ID> for Arena<ID, T> {
    fn index_mut(&mut self, idx: ID) -> &mut T {
        let idx = idx.to_u32() as usize;
        &mut self.data[idx]
    }
}
