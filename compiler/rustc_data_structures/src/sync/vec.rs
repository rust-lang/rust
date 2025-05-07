use std::marker::PhantomData;

use rustc_index::Idx;

#[derive(Default)]
pub struct AppendOnlyIndexVec<I: Idx, T: Copy> {
    vec: elsa::sync::LockFreeFrozenVec<T>,
    _marker: PhantomData<fn(&I)>,
}

impl<I: Idx, T: Copy> AppendOnlyIndexVec<I, T> {
    pub fn new() -> Self {
        Self { vec: elsa::sync::LockFreeFrozenVec::new(), _marker: PhantomData }
    }

    pub fn push(&self, val: T) -> I {
        let i = self.vec.push(val);
        I::new(i)
    }

    pub fn get(&self, i: I) -> Option<T> {
        let i = i.index();
        self.vec.get(i)
    }
}

#[derive(Default)]
pub struct AppendOnlyVec<T: Copy> {
    vec: parking_lot::RwLock<Vec<T>>,
}

impl<T: Copy> AppendOnlyVec<T> {
    pub fn new() -> Self {
        Self { vec: Default::default() }
    }

    pub fn push(&self, val: T) -> usize {
        let mut v = self.vec.write();
        let n = v.len();
        v.push(val);
        n
    }

    pub fn get(&self, i: usize) -> Option<T> {
        self.vec.read().get(i).copied()
    }

    pub fn iter_enumerated(&self) -> impl Iterator<Item = (usize, T)> {
        (0..)
            .map(|i| (i, self.get(i)))
            .take_while(|(_, o)| o.is_some())
            .filter_map(|(i, o)| Some((i, o?)))
    }

    pub fn iter(&self) -> impl Iterator<Item = T> {
        (0..).map(|i| self.get(i)).take_while(|o| o.is_some()).flatten()
    }
}

impl<T: Copy + PartialEq> AppendOnlyVec<T> {
    pub fn contains(&self, val: T) -> bool {
        self.iter_enumerated().any(|(_, v)| v == val)
    }
}

impl<A: Copy> FromIterator<A> for AppendOnlyVec<A> {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let this = Self::new();
        for val in iter {
            this.push(val);
        }
        this
    }
}
