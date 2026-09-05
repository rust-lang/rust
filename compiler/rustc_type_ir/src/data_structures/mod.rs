pub use ena::unify::{NoError, UnifyKey, UnifyValue};
use rustc_hash::FxBuildHasher;

// `hashbrown`'s hash map and set rather than the standard library's, matching
// `rustc_data_structures::fx`. [LLM-generated]
pub type HashMap<K, V> = hashbrown::HashMap<K, V, FxBuildHasher>;
pub type HashSet<T> = hashbrown::HashSet<T, FxBuildHasher>;
pub use hashbrown::hash_map;

pub type IndexMap<K, V> = indexmap::IndexMap<K, V, FxBuildHasher>;
pub type IndexSet<V> = indexmap::IndexSet<V, FxBuildHasher>;

mod delayed_map;

#[cfg(feature = "nightly")]
mod impl_ {
    pub use rustc_data_structures::sso::{SsoHashMap, SsoHashSet};
}

#[cfg(not(feature = "nightly"))]
mod impl_ {
    pub use std::collections::{HashMap as SsoHashMap, HashSet as SsoHashSet};
}

pub use delayed_map::{DelayedMap, DelayedSet};
pub use impl_::*;
