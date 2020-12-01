use std::hash::BuildHasherDefault;

pub use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

pub type FxIndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasherDefault<FxHasher>>;
pub type FxIndexSet<V> = indexmap::IndexSet<V, BuildHasherDefault<FxHasher>>;

#[macro_export]
macro_rules! define_id_collections {
    ($map_name:ident, $set_name:ident, $key:ty) => {
        pub type $map_name<T> = $crate::fx::FxHashMap<$key, T>;
        pub type $set_name = $crate::fx::FxHashSet<$key>;
    };
}
