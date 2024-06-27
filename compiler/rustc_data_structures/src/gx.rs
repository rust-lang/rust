use std::hash::BuildHasherDefault;

#[cfg(not(all(target_feature = "sse2", target_feature = "aes")))]
pub use rustc_hash::{FxHashMap as GxHashMap, FxHashSet as GxHashSet, FxHasher as GxHasher};

#[cfg(all(target_feature = "sse2", target_feature = "aes"))]
pub use gxhash::GxHasher;

pub type StdEntry<'a, K, V> = std::collections::hash_map::Entry<'a, K, V>;

#[cfg(all(target_feature = "sse2", target_feature = "aes"))]
pub type GxHashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<GxHasher>>;
#[cfg(all(target_feature = "sse2", target_feature = "aes"))]
pub type GxHashSet<T> = std::collections::HashSet<T, BuildHasherDefault<GxHasher>>;

pub type GxIndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasherDefault<GxHasher>>;
pub type GxIndexSet<V> = indexmap::IndexSet<V, BuildHasherDefault<GxHasher>>;
pub type IndexEntry<'a, K, V> = indexmap::map::Entry<'a, K, V>;
pub type IndexOccupiedEntry<'a, K, V> = indexmap::map::OccupiedEntry<'a, K, V>;

#[macro_export]
macro_rules! define_id_collections {
    ($map_name:ident, $set_name:ident, $entry_name:ident, $key:ty) => {
        pub type $map_name<T> = $crate::unord::UnordMap<$key, T>;
        pub type $set_name = $crate::unord::UnordSet<$key>;
        pub type $entry_name<'a, T> = $crate::gx::StdEntry<'a, $key, T>;
    };
}

#[macro_export]
macro_rules! define_stable_id_collections {
    ($map_name:ident, $set_name:ident, $entry_name:ident, $key:ty) => {
        pub type $map_name<T> = $crate::gx::GxIndexMap<$key, T>;
        pub type $set_name = $crate::gx::GxIndexSet<$key>;
        pub type $entry_name<'a, T> = $crate::gx::IndexEntry<'a, $key, T>;
    };
}
