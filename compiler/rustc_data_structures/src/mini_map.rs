use crate::fx::FxHashMap;
use arrayvec::ArrayVec;

use std::hash::Hash;

/// Small-storage-optimized implementation of a map
/// made specifically for caching results.
///
/// Stores elements in a small array up to a certain length
/// and switches to `HashMap` when that length is exceeded.
pub enum MiniMap<K, V> {
    Array(ArrayVec<[(K, V); 8]>),
    Map(FxHashMap<K, V>),
}

impl<K: Eq + Hash, V> MiniMap<K, V> {
    /// Creates an empty `MiniMap`.
    pub fn new() -> Self {
        MiniMap::Array(ArrayVec::new())
    }

    /// Inserts or updates value in the map.
    pub fn insert(&mut self, key: K, value: V) {
        match self {
            MiniMap::Array(array) => {
                for pair in array.iter_mut() {
                    if pair.0 == key {
                        pair.1 = value;
                        return;
                    }
                }
                if let Err(error) = array.try_push((key, value)) {
                    let mut map: FxHashMap<K, V> = array.drain(..).collect();
                    let (key, value) = error.element();
                    map.insert(key, value);
                    *self = MiniMap::Map(map);
                }
            }
            MiniMap::Map(map) => {
                map.insert(key, value);
            }
        }
    }

    /// Return value by key if any.
    pub fn get(&self, key: &K) -> Option<&V> {
        match self {
            MiniMap::Array(array) => {
                for pair in array {
                    if pair.0 == *key {
                        return Some(&pair.1);
                    }
                }
                return None;
            }
            MiniMap::Map(map) => {
                return map.get(key);
            }
        }
    }
}
