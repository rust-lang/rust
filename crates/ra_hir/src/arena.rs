//! A simple id-based arena, similar to https://github.com/fitzgen/id-arena.
//! We use our own version for more compact id's and to allow inherent impls
//! on Ids.

use std::{
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
};

pub struct Id<T> {
    idx: u32,
    _ty: PhantomData<fn() -> T>,
}

impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Id").field(&self.idx).finish()
    }
}
impl<T> Copy for Id<T> {}
impl<T> Clone for Id<T> {
    fn clone(&self) -> Id<T> {
        *self
    }
}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Id<T>) -> bool {
        self.idx == other.idx
    }
}

impl<T> Eq for Id<T> {}

impl<T> Hash for Id<T> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.idx.hash(h);
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ArenaBehavior<T> {
    _ty: PhantomData<T>,
}

impl<T> id_arena::ArenaBehavior for ArenaBehavior<T> {
    type Id = Id<T>;
    fn new_arena_id() -> usize {
        0
    }
    fn new_id(_arena_id: usize, index: usize) -> Id<T> {
        Id {
            idx: index as u32,
            _ty: PhantomData,
        }
    }
    fn index(id: Id<T>) -> usize {
        id.idx as usize
    }
    fn arena_id(_id: Id<T>) -> usize {
        0
    }
}

pub(crate) type Arena<T> = id_arena::Arena<T, ArenaBehavior<T>>;
