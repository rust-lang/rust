pub use ena::unify::{NoError, UnifyKey, UnifyValue};
use rustc_hash::FxBuildHasher;
pub use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

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
