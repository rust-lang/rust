use std::hash::{Hash, Hasher};
use std::ops::Deref;

use smallvec::SmallVec;

use crate::dep_graph::DepNodeIndex;

#[derive(Default, Debug)]
pub(crate) struct EdgesVec {
    max: u32,
    edges: SmallVec<[DepNodeIndex; EdgesVec::INLINE_CAPACITY]>,
}

impl Hash for EdgesVec {
    #[inline]
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        Hash::hash(&self.edges, hasher)
    }
}

impl EdgesVec {
    pub(crate) const INLINE_CAPACITY: usize = 8;

    #[inline]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub(crate) fn push(&mut self, edge: DepNodeIndex) {
        self.max = self.max.max(edge.as_u32());
        self.edges.push(edge);
    }

    #[inline]
    pub(crate) fn max_index(&self) -> u32 {
        self.max
    }
}

impl Deref for EdgesVec {
    type Target = [DepNodeIndex];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.edges.as_slice()
    }
}

impl FromIterator<DepNodeIndex> for EdgesVec {
    #[inline]
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = DepNodeIndex>,
    {
        let mut vec = EdgesVec::new();
        for index in iter {
            vec.push(index)
        }
        vec
    }
}

impl Extend<DepNodeIndex> for EdgesVec {
    #[inline]
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = DepNodeIndex>,
    {
        for elem in iter {
            self.push(elem);
        }
    }
}
