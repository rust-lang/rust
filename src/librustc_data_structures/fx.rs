use std::hash::BuildHasherDefault;

pub use rustc_hash::{FxHasher, FxHashMap, FxHashSet};

pub type FxIndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasherDefault<FxHasher>>;
pub type FxIndexSet<V> = indexmap::IndexSet<V, BuildHasherDefault<FxHasher>>;
