pub use rustc_hash::FxHashMap;
use std::borrow::Borrow;
use std::collections::hash_map::Entry;
use std::fmt;
use std::hash::Hash;

/// A deterministic wrapper around FxHashMap that does not provide iteration support.
///
/// It supports insert, remove, get and get_mut functions from FxHashMap.
/// It also allows to convert hashmap to a sorted vector with the method `into_sorted_vector()`.
#[derive(Clone)]
pub struct StableMap<K, V> {
    base: FxHashMap<K, V>,
}

impl<K, V> Default for StableMap<K, V>
where
    K: Eq + Hash,
{
    fn default() -> StableMap<K, V> {
        StableMap::new()
    }
}

impl<K, V> fmt::Debug for StableMap<K, V>
where
    K: Eq + Hash + fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.base)
    }
}

impl<K, V> PartialEq for StableMap<K, V>
where
    K: Eq + Hash,
    V: PartialEq,
{
    fn eq(&self, other: &StableMap<K, V>) -> bool {
        self.base == other.base
    }
}

impl<K, V> Eq for StableMap<K, V>
where
    K: Eq + Hash,
    V: Eq,
{
}

impl<K, V> StableMap<K, V>
where
    K: Eq + Hash,
{
    pub fn new() -> StableMap<K, V> {
        StableMap { base: FxHashMap::default() }
    }

    pub fn into_sorted_vector(self) -> Vec<(K, V)>
    where
        K: Ord + Copy,
    {
        let mut vector = self.base.into_iter().collect::<Vec<_>>();
        vector.sort_unstable_by_key(|pair| pair.0);
        vector
    }

    pub fn entry(&mut self, k: K) -> Entry<'_, K, V> {
        self.base.entry(k)
    }

    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.base.get(k)
    }

    pub fn get_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.base.get_mut(k)
    }

    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.base.insert(k, v)
    }

    pub fn remove<Q: ?Sized>(&mut self, k: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.base.remove(k)
    }
}
