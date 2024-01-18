use std::collections::{HashMap, HashSet};

use crate::FxHasher;

/// Type alias for a hashmap using the `fx` hash algorithm with [`FxSeededState`].
pub type FxHashMapSeed<K, V> = HashMap<K, V, FxSeededState>;

/// Type alias for a hashmap using the `fx` hash algorithm with [`FxSeededState`].
pub type FxHashSetSeed<V> = HashSet<V, FxSeededState>;

/// [`FxSetState`] is an alternative state for `HashMap` types, allowing to use [`FxHasher`] with a set seed.
///
/// ```
/// # use std::collections::HashMap;
/// use rustc_hash::FxSeededState;
///
/// let mut map = HashMap::with_hasher(FxSeededState::with_seed(12));
/// map.insert(15, 610);
/// assert_eq!(map[&15], 610);
/// ```
pub struct FxSeededState {
    seed: usize,
}

impl FxSeededState {
    /// Constructs a new `FxSeededState` that is initialized with a `seed`.
    pub fn with_seed(seed: usize) -> FxSeededState {
        Self { seed }
    }
}

impl core::hash::BuildHasher for FxSeededState {
    type Hasher = FxHasher;

    fn build_hasher(&self) -> Self::Hasher {
        FxHasher::with_seed(self.seed)
    }
}

#[cfg(test)]
mod tests {
    use core::hash::BuildHasher;

    use crate::{FxHashMapSeed, FxSeededState};

    #[test]
    fn different_states_are_different() {
        let a = FxHashMapSeed::<&str, u32>::with_hasher(FxSeededState::with_seed(1));
        let b = FxHashMapSeed::<&str, u32>::with_hasher(FxSeededState::with_seed(2));

        assert_ne!(
            a.hasher().build_hasher().hash,
            b.hasher().build_hasher().hash
        );
    }
}
