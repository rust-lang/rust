//! A pass that checks to make sure private fields and methods aren't used
//! outside their scopes. This pass will also generate a set of exported items
//! which are available for use externally when compiled as a library.
use crate::ty::{DefIdTree, Visibility};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_macros::HashStable;
use rustc_query_system::ich::StableHashingContext;
use rustc_span::def_id::{DefId, LocalDefId};
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

impl AccessLevel {
    pub fn all_levels() -> [AccessLevel; 4] {
        [
            AccessLevel::Public,
            AccessLevel::Exported,
            AccessLevel::Reachable,
            AccessLevel::ReachableFromImplTrait,
        ]
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, HashStable)]
pub struct EffectiveVisibility {
    public: Visibility,
    exported: Visibility,
    reachable: Visibility,
    reachable_from_impl_trait: Visibility,
}

impl EffectiveVisibility {
    pub fn get(&self, tag: AccessLevel) -> &Visibility {
        match tag {
            AccessLevel::Public => &self.public,
            AccessLevel::Exported => &self.exported,
            AccessLevel::Reachable => &self.reachable,
            AccessLevel::ReachableFromImplTrait => &self.reachable_from_impl_trait,
        }
    }

    fn get_mut(&mut self, tag: AccessLevel) -> &mut Visibility {
        match tag {
            AccessLevel::Public => &mut self.public,
            AccessLevel::Exported => &mut self.exported,
            AccessLevel::Reachable => &mut self.reachable,
            AccessLevel::ReachableFromImplTrait => &mut self.reachable_from_impl_trait,
        }
    }

    pub fn is_public_at_level(&self, tag: AccessLevel) -> bool {
        self.get(tag).is_public()
    }

    pub fn from_vis(vis: Visibility) -> EffectiveVisibility {
        EffectiveVisibility {
            public: vis,
            exported: vis,
            reachable: vis,
            reachable_from_impl_trait: vis,
        }
    }
}

/// Holds a map of accessibility levels for reachable HIR nodes.
#[derive(Debug, Clone)]
pub struct AccessLevels<Id = LocalDefId> {
    map: FxHashMap<Id, EffectiveVisibility>,
}

impl<Id: Hash + Eq + Copy> AccessLevels<Id> {
    pub fn is_public_at_level(&self, id: Id, tag: AccessLevel) -> bool {
        self.get_effective_vis(id)
            .map_or(false, |effective_vis| effective_vis.is_public_at_level(tag))
    }

    /// See `AccessLevel::Reachable`.
    pub fn is_reachable(&self, id: Id) -> bool {
        self.is_public_at_level(id, AccessLevel::Reachable)
    }

    /// See `AccessLevel::Exported`.
    pub fn is_exported(&self, id: Id) -> bool {
        self.is_public_at_level(id, AccessLevel::Exported)
    }

    /// See `AccessLevel::Public`.
    pub fn is_public(&self, id: Id) -> bool {
        self.is_public_at_level(id, AccessLevel::Public)
    }

    pub fn get_access_level(&self, id: Id) -> Option<AccessLevel> {
        self.get_effective_vis(id).and_then(|effective_vis| {
            for level in AccessLevel::all_levels() {
                if effective_vis.is_public_at_level(level) {
                    return Some(level);
                }
            }
            None
        })
    }

    pub fn get_effective_vis(&self, id: Id) -> Option<&EffectiveVisibility> {
        self.map.get(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Id, &EffectiveVisibility)> {
        self.map.iter()
    }

    pub fn map_id<OutId: Hash + Eq + Copy>(&self, f: impl Fn(Id) -> OutId) -> AccessLevels<OutId> {
        AccessLevels { map: self.map.iter().map(|(k, v)| (f(*k), *v)).collect() }
    }

    pub fn set_access_level(
        &mut self,
        id: Id,
        default_vis: impl FnOnce() -> Visibility,
        tag: AccessLevel,
    ) {
        let mut effective_vis = self
            .get_effective_vis(id)
            .copied()
            .unwrap_or_else(|| EffectiveVisibility::from_vis(default_vis()));
        for level in AccessLevel::all_levels() {
            if level <= tag {
                *effective_vis.get_mut(level) = Visibility::Public;
            }
        }
        self.map.insert(id, effective_vis);
    }
}

impl<Id: Hash + Eq + Copy + Into<DefId>> AccessLevels<Id> {
    // `parent_id` is not necessarily a parent in source code tree,
    // it is the node from which the maximum effective visibility is inherited.
    pub fn update(
        &mut self,
        id: Id,
        nominal_vis: Visibility,
        default_vis: impl FnOnce() -> Visibility,
        parent_id: Id,
        tag: AccessLevel,
        tree: impl DefIdTree,
    ) -> bool {
        let mut changed = false;
        let mut current_effective_vis = self.get_effective_vis(id).copied().unwrap_or_else(|| {
            if id.into().is_crate_root() {
                EffectiveVisibility::from_vis(Visibility::Public)
            } else {
                EffectiveVisibility::from_vis(default_vis())
            }
        });
        if let Some(inherited_effective_vis) = self.get_effective_vis(parent_id) {
            let mut inherited_effective_vis_at_prev_level = *inherited_effective_vis.get(tag);
            let mut calculated_effective_vis = inherited_effective_vis_at_prev_level;
            for level in AccessLevel::all_levels() {
                if tag >= level {
                    let inherited_effective_vis_at_level = *inherited_effective_vis.get(level);
                    let current_effective_vis_at_level = current_effective_vis.get_mut(level);
                    // effective visibility for id shouldn't be recalculated if
                    // inherited from parent_id effective visibility isn't changed at next level
                    if !(inherited_effective_vis_at_prev_level == inherited_effective_vis_at_level
                        && tag != level)
                    {
                        calculated_effective_vis =
                            if nominal_vis.is_at_least(inherited_effective_vis_at_level, tree) {
                                inherited_effective_vis_at_level
                            } else {
                                nominal_vis
                            };
                    }
                    // effective visibility can't be decreased at next update call for the
                    // same id
                    if *current_effective_vis_at_level != calculated_effective_vis
                        && calculated_effective_vis
                            .is_at_least(*current_effective_vis_at_level, tree)
                    {
                        changed = true;
                        *current_effective_vis_at_level = calculated_effective_vis;
                    }
                    inherited_effective_vis_at_prev_level = inherited_effective_vis_at_level;
                }
            }
        }
        self.map.insert(id, current_effective_vis);
        changed
    }
}

impl<Id> Default for AccessLevels<Id> {
    fn default() -> Self {
        AccessLevels { map: Default::default() }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for AccessLevels {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let AccessLevels { ref map } = *self;
        map.hash_stable(hcx, hasher);
    }
}
