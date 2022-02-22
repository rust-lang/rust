//! Cache for candidate selection.

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lock;

use std::hash::Hash;

#[derive(Clone)]
pub struct Cache<Key, Value> {
    hashmap: Lock<FxHashMap<Key, Value>>,
}

impl<Key, Value> Default for Cache<Key, Value> {
    fn default() -> Self {
        Self { hashmap: Default::default() }
    }
}

impl<Key, Value> Cache<Key, Value> {
    /// Actually frees the underlying memory in contrast to what stdlib containers do on `clear`
    pub fn clear(&self) {
        *self.hashmap.borrow_mut() = Default::default();
    }
}

impl<Key: Eq + Hash, Value: Clone> Cache<Key, Value> {
    pub fn get(&self, key: &Key) -> Option<Value> {
        Some(self.hashmap.borrow().get(key)?.clone())
    }

    pub fn insert(&self, key: Key, value: Value) {
        self.hashmap.borrow_mut().insert(key, value);
    }
}
