//! A pass that checks to make sure private fields and methods aren't used
//! outside their scopes. This pass will also generate a set of exported items
//! which are available for use externally when compiled as a library.

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::HirId;
use rustc_macros::HashStable;
use std::fmt;
use std::hash::Hash;

// Accessibility levels, sorted in ascending order
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, HashStable)]
pub enum AccessLevel {
    /// Superset of `AccessLevel::Reachable` used to mark impl Trait items.
    ReachableFromImplTrait,
    /// Exported items + items participating in various kinds of public interfaces,
    /// but not directly nameable. For example, if function `fn f() -> T {...}` is
    /// public, then type `T` is reachable. Its values can be obtained by other crates
    /// even if the type itself is not nameable.
    Reachable,
    /// Public items + items accessible to other crates with help of `pub use` re-exports
    Exported,
    /// Items accessible to other crates directly, without help of re-exports
    Public,
}

// Accessibility levels for reachable HIR nodes
#[derive(Clone)]
pub struct AccessLevels<Id = HirId> {
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

impl<Id: Hash + Eq> Default for AccessLevels<Id> {
    fn default() -> Self {
        AccessLevels { map: Default::default() }
    }
}

impl<Id: Hash + Eq + fmt::Debug> fmt::Debug for AccessLevels<Id> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.map, f)
    }
}
