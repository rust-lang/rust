use std::hash::BuildHasherDefault;
use std::collections::hash_map::HashMap;
use std::collections::hash_set::HashSet;
pub use ahash::AHasher as FxHasher;

/// Type alias for a hashmap using the `fx` hash algorithm.
pub type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// Type alias for a hashmap using the `fx` hash algorithm.
pub type FxHashSet<V> = HashSet<V, BuildHasherDefault<FxHasher>>;
