pub use rustc_hash::FxHashSet;
use std::borrow::Borrow;
use std::fmt;
use std::hash::Hash;

use crate::stable_hasher::{stable_hash_reduce, HashStable, StableHasher, ToStableHashKey};

/// A deterministic wrapper around FxHashSet that does not provide iteration support.
///
/// It supports insert, remove, get functions from FxHashSet.
/// It also allows to convert hashset to a sorted vector with the method `into_sorted_vector()`.
#[derive(Clone, Encodable, Decodable)]
pub struct StableSet<T>
where
    T: Eq + Hash,
{
    base: FxHashSet<T>,
}

impl<T> Default for StableSet<T>
where
    T: Eq + Hash,
{
    #[inline]
    fn default() -> StableSet<T> {
        StableSet::new()
    }
}

impl<T> fmt::Debug for StableSet<T>
where
    T: Eq + Hash + fmt::Debug,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.base)
    }
}

impl<T> PartialEq<StableSet<T>> for StableSet<T>
where
    T: Eq + Hash,
{
    #[inline]
    fn eq(&self, other: &StableSet<T>) -> bool {
        self.base == other.base
    }
}

impl<T> Eq for StableSet<T> where T: Eq + Hash {}

impl<T: Hash + Eq> StableSet<T> {
    #[inline]
    pub fn new() -> StableSet<T> {
        StableSet { base: FxHashSet::default() }
    }

    #[inline]
    pub fn into_sorted_vector<HCX>(self, hcx: &HCX) -> Vec<T>
    where
        T: ToStableHashKey<HCX>,
    {
        let mut vector = self.base.into_iter().collect::<Vec<_>>();
        vector.sort_by_cached_key(|x| x.to_stable_hash_key(hcx));
        vector
    }

    #[inline]
    pub fn sorted_vector<HCX>(&self, hcx: &HCX) -> Vec<&T>
    where
        T: ToStableHashKey<HCX>,
    {
        let mut vector = self.base.iter().collect::<Vec<_>>();
        vector.sort_by_cached_key(|x| x.to_stable_hash_key(hcx));
        vector
    }

    #[inline]
    pub fn get<Q: ?Sized>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.base.get(value)
    }

    #[inline]
    pub fn insert(&mut self, value: T) -> bool {
        self.base.insert(value)
    }

    #[inline]
    pub fn remove<Q: ?Sized>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.base.remove(value)
    }

    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        self.base.contains(value)
    }
}

impl<T, HCX> HashStable<HCX> for StableSet<T>
where
    T: ToStableHashKey<HCX> + Eq + Hash,
{
    #[inline]
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut StableHasher) {
        stable_hash_reduce(hcx, hasher, self.base.iter(), self.base.len(), |hasher, hcx, key| {
            let key = key.to_stable_hash_key(hcx);
            key.hash_stable(hcx, hasher);
        });
    }
}

impl<T> FromIterator<T> for StableSet<T>
where
    T: Eq + Hash,
{
    #[inline]
    fn from_iter<Collection: IntoIterator<Item = T>>(iter: Collection) -> Self {
        Self { base: iter.into_iter().collect() }
    }
}

impl<T> Extend<T> for StableSet<T>
where
    T: Eq + Hash,
{
    #[inline]
    fn extend<Iter: IntoIterator<Item = T>>(&mut self, iter: Iter) {
        self.base.extend(iter)
    }
}
