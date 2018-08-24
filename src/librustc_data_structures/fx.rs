use std::collections::{HashMap, HashSet};
use std::default::Default;
use std::hash::Hash;

pub use rustc_hash::FxHashMap;
pub use rustc_hash::FxHashSet;
pub use rustc_hash::FxHasher;

#[allow(non_snake_case)]
pub fn FxHashMap<K: Hash + Eq, V>() -> FxHashMap<K, V> {
    HashMap::default()
}

#[allow(non_snake_case)]
pub fn FxHashSet<V: Hash + Eq>() -> FxHashSet<V> {
    HashSet::default()
}

