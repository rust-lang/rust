//! A pass that checks to make sure private fields and methods aren't used
//! outside their scopes. This pass will also generate a set of exported items
//! which are available for use externally when compiled as a library.

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_macros::HashStable;
use rustc_query_system::ich::{NodeIdHashingMode, StableHashingContext};
use rustc_span::def_id::LocalDefId;
use std::hash::Hash;

/// Represents the levels of accessibility an item can have.
///
/// The variants are sorted in ascending order of accessibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, HashStable)]
pub enum AccessLevel {
    /// Superset of `AccessLevel::Reachable` used to mark impl Trait items.
    ReachableFromImplTrait,
    /// Exported items + items participating in various kinds of public interfaces,
    /// but not directly nameable. For example, if function `fn f() -> T {...}` is
    /// public, then type `T` is reachable. Its values can be obtained by other crates
    /// even if the type itself is not nameable.
    Reachable,
    /// Public items + items accessible to other crates with the help of `pub use` re-exports.
    Exported,
    /// Items accessible to other crates directly, without the help of re-exports.
    Public,
}

/// Holds a map of accessibility levels for reachable HIR nodes.
#[derive(Debug, Clone)]
pub struct AccessLevels<Id = LocalDefId> {
    pub map: FxHashMap<Id, AccessLevel>,
}

impl<Id: Hash + Eq> AccessLevels<Id> {
    /// See `AccessLevel::Reachable`.
    pub fn is_reachable(&self, id: Id) -> bool {
        self.map.get(&id) >= Some(&AccessLevel::Reachable)
    }

    /// See `AccessLevel::Exported`.
    pub fn is_exported(&self, id: Id) -> bool {
        self.map.get(&id) >= Some(&AccessLevel::Exported)
    }

    /// See `AccessLevel::Public`.
    pub fn is_public(&self, id: Id) -> bool {
        self.map.get(&id) >= Some(&AccessLevel::Public)
    }
}

impl<Id> Default for AccessLevels<Id> {
    fn default() -> Self {
        AccessLevels { map: Default::default() }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for AccessLevels {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            let AccessLevels { ref map } = *self;
            map.hash_stable(hcx, hasher);
        });
    }
}
