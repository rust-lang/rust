use std::hash::BuildHasherDefault;

pub use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet, FxHasher};

pub type StdEntry<'a, K, V> = std::collections::hash_map::Entry<'a, K, V>;

pub type FxIndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasherDefault<FxHasher>>;
pub type FxIndexSet<V> = indexmap::IndexSet<V, BuildHasherDefault<FxHasher>>;
pub type IndexEntry<'a, K, V> = indexmap::map::Entry<'a, K, V>;
pub type IndexOccupiedEntry<'a, K, V> = indexmap::map::OccupiedEntry<'a, K, V>;

pub use indexmap::set::MutableValues;

#[macro_export]
macro_rules! define_id_collections {
    ($map_name:ident, $set_name:ident, $entry_name:ident, $key:ty) => {
        pub type $map_name<T> = $crate::unord::UnordMap<$key, T>;
        pub type $set_name = $crate::unord::UnordSet<$key>;
        pub type $entry_name<'a, T> = $crate::fx::StdEntry<'a, $key, T>;
    };
}

#[macro_export]
macro_rules! define_stable_id_collections {
    ($map_name:ident, $set_name:ident, $entry_name:ident, $key:ty) => {
        pub type $map_name<T> = $crate::fx::FxIndexMap<$key, T>;
        pub type $set_name = $crate::fx::FxIndexSet<$key>;
        pub type $entry_name<'a, T> = $crate::fx::IndexEntry<'a, $key, T>;
    };
}

pub mod default {
    use super::{FxBuildHasher, FxHashMap, FxHashSet};

    // FIXME: These two functions will become unnecessary after
    // <https://github.com/rust-lang/rustc-hash/pull/63> lands and we start using the corresponding
    // `rustc-hash` version. After that we can use `Default::default()` instead.
    pub const fn fx_hash_map<K, V>() -> FxHashMap<K, V> {
        FxHashMap::with_hasher(FxBuildHasher)
    }

    pub const fn fx_hash_set<V>() -> FxHashSet<V> {
        FxHashSet::with_hasher(FxBuildHasher)
    }
}
