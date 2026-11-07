use std::{borrow::Borrow, hash::Hash};

use crate::thin_map::{Map, ThinMap};

#[expect(clippy::disallowed_types)]
pub type ThinHashSet<T, S> = ThinSet<T, std::collections::HashMap<T, (), S>>;

pub struct ThinSet<T, M: Map<T, ()>> {
    map: ThinMap<T, (), M>,
}

impl<T: Hash + Eq, M: Map<T, ()>> ThinSet<T, M> {
    #[inline]
    pub const fn new() -> Self {
        Self { map: ThinMap::new() }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self { map: ThinMap::with_capacity(capacity) }
    }

    pub fn shrink_to_fit(&mut self) {
        self.map.shrink_to_fit();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    #[inline]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> {
        self.map.keys()
    }

    #[expect(clippy::should_implement_trait, reason = "cannot do that without ATPIT")]
    #[inline]
    pub fn into_iter(self) -> impl ExactSizeIterator<Item = T> {
        self.map.into_keys()
    }

    #[inline]
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.map.contains_key(key)
    }

    #[inline]
    pub fn insert(&mut self, item: T) -> bool {
        self.map.insert(item, ()).is_none()
    }

    #[inline]
    pub fn remove<Q>(&mut self, item: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.map.remove(item).is_some()
    }
}

impl<T: Hash + Eq, M: Map<T, ()>> Default for ThinSet<T, M> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, M> std::fmt::Debug for ThinSet<T, M>
where
    M: Map<T, ()>,
    T: Hash + Eq + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T, M> Clone for ThinSet<T, M>
where
    M: Map<T, ()> + Clone,
    T: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { map: self.map.clone() }
    }
}

impl<T, M> PartialEq for ThinSet<T, M>
where
    M: Map<T, ()> + PartialEq,
    T: Hash + Eq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map
    }
}

impl<T, M> Eq for ThinSet<T, M>
where
    M: Map<T, ()> + Eq,
    T: Hash + Eq,
{
}

impl<T: Hash + Eq, M: Map<T, ()>> FromIterator<T> for ThinSet<T, M> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self { map: iter.into_iter().map(|item| (item, ())).collect() }
    }
}

impl<T: Hash + Eq, M: Map<T, ()>> Extend<T> for ThinSet<T, M> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.map.extend(iter.into_iter().map(|item| (item, ())));
    }
}
