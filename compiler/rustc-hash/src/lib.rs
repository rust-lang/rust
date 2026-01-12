pub mod hash;

use std::hash::BuildHasher;

pub use hash::Hasher as FxHasher;

#[derive(Copy, Clone, Debug, Default)]
pub struct FxBuildHasher;

impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;

    fn build_hasher(&self) -> Self::Hasher {
        hash::FixedState::new(0).build_hasher()
    }
}

#[allow(rustc::default_hash_types)]
pub type FxHashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;
#[allow(rustc::default_hash_types)]
pub type FxHashSet<V> = std::collections::HashSet<V, FxBuildHasher>;
