//! A simple id-based arena, similar to https://github.com/fitzgen/id-arena.
//! We use our own version for more compact id's and to allow inherent impls
//! on Ids.

use std::{
    fmt,
    ops::{Index, IndexMut},
    hash::{Hash, Hasher},
    marker::PhantomData,
};

pub(crate) struct Id<T> {
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
pub(crate) struct Arena<T> {
    data: Vec<T>,
}

impl<T> Default for Arena<T> {
    fn default() -> Arena<T> {
        Arena { data: Vec::new() }
    }
}

impl<T> Arena<T> {
    pub(crate) fn push(&mut self, value: T) -> Id<T> {
        let id = self.data.len() as u32;
        self.data.push(value);
        Id {
            idx: id as u32,
            _ty: PhantomData,
        }
    }

    pub(crate) fn keys<'a>(&'a self) -> impl Iterator<Item = Id<T>> + 'a {
        (0..(self.data.len() as u32)).into_iter().map(|idx| Id {
            idx,
            _ty: PhantomData,
        })
    }

    pub(crate) fn items<'a>(&'a self) -> impl Iterator<Item = (Id<T>, &T)> + 'a {
        self.data.iter().enumerate().map(|(idx, item)| {
            let idx = idx as u32;
            (
                Id {
                    idx,
                    _ty: PhantomData,
                },
                item,
            )
        })
    }
}

impl<T> Index<Id<T>> for Arena<T> {
    type Output = T;
    fn index(&self, id: Id<T>) -> &T {
        &self.data[id.idx as usize]
    }
}

impl<T> IndexMut<Id<T>> for Arena<T> {
    fn index_mut(&mut self, id: Id<T>) -> &mut T {
        &mut self.data[id.idx as usize]
    }
}
