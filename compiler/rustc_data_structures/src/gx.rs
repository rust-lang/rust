use std::{
    collections::{HashMap, HashSet},
    hash::BuildHasherDefault,
};

// pub use rustc_hash::{GxHashMap, GxHashSet, GxHasher};

pub use gxhash::GxHasher;

pub type StdEntry<'a, K, V> = std::collections::hash_map::Entry<'a, K, V>;

pub type GxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<GxHasher>>;
pub type GxHashSet<T> = HashSet<T, BuildHasherDefault<GxHasher>>;

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
