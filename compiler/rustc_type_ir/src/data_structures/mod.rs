use std::hash::BuildHasherDefault;

use rustc_hash::FxHasher;
pub use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

pub type IndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasherDefault<FxHasher>>;
pub type IndexSet<V> = indexmap::IndexSet<V, BuildHasherDefault<FxHasher>>;

mod delayed_map;

#[cfg(feature = "nightly")]
mod impl_ {
    pub use rustc_data_structures::sso::{SsoHashMap, SsoHashSet};
    pub use rustc_data_structures::stack::ensure_sufficient_stack;
    pub use rustc_data_structures::sync::Lrc;
}

#[cfg(not(feature = "nightly"))]
mod impl_ {
    pub use std::collections::{HashMap as SsoHashMap, HashSet as SsoHashSet};
    pub use std::sync::Arc as Lrc;

    #[inline]
    pub fn ensure_sufficient_stack<R>(f: impl FnOnce() -> R) -> R {
        f()
    }
}

pub use delayed_map::{DelayedMap, DelayedSet};
pub use impl_::*;
