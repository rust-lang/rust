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

/// Represents the levels of effective visibility an item can have.
///
/// The variants are sorted in ascending order of directness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, HashStable)]
pub enum Level {
    /// Superset of `Reachable` including items leaked through return position `impl Trait`.
    ReachableThroughImplTrait,
    /// Item is either reexported, or leaked through any kind of interface.
    /// For example, if function `fn f() -> T {...}` is directly public, then type `T` is publicly
    /// reachable and its values can be obtained by other crates even if the type itself is not
    /// nameable.
    Reachable,
    /// Item is accessible either directly, or with help of `use` reexports.
    Reexported,
    /// Item is directly accessible, without help of reexports.
    Direct,
}

impl Level {
    pub fn all_levels() -> [Level; 4] {
        [Level::Direct, Level::Reexported, Level::Reachable, Level::ReachableThroughImplTrait]
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, HashStable)]
pub struct EffectiveVisibility {
    direct: Visibility,
    reexported: Visibility,
    reachable: Visibility,
    reachable_through_impl_trait: Visibility,
}

impl EffectiveVisibility {
    pub fn at_level(&self, level: Level) -> &Visibility {
        match level {
            Level::Direct => &self.direct,
            Level::Reexported => &self.reexported,
            Level::Reachable => &self.reachable,
            Level::ReachableThroughImplTrait => &self.reachable_through_impl_trait,
        }
    }

    fn at_level_mut(&mut self, level: Level) -> &mut Visibility {
        match level {
            Level::Direct => &mut self.direct,
            Level::Reexported => &mut self.reexported,
            Level::Reachable => &mut self.reachable,
            Level::ReachableThroughImplTrait => &mut self.reachable_through_impl_trait,
        }
    }

    pub fn is_public_at_level(&self, level: Level) -> bool {
        self.at_level(level).is_public()
    }

    pub fn from_vis(vis: Visibility) -> EffectiveVisibility {
        EffectiveVisibility {
            direct: vis,
            reexported: vis,
            reachable: vis,
            reachable_through_impl_trait: vis,
        }
    }
}

/// Holds a map of effective visibilities for reachable HIR nodes.
#[derive(Debug, Clone)]
pub struct EffectiveVisibilities<Id = LocalDefId> {
    map: FxHashMap<Id, EffectiveVisibility>,
}

impl<Id: Hash + Eq + Copy> EffectiveVisibilities<Id> {
    pub fn is_public_at_level(&self, id: Id, level: Level) -> bool {
        self.effective_vis(id)
            .map_or(false, |effective_vis| effective_vis.is_public_at_level(level))
    }

    /// See `Level::Reachable`.
    pub fn is_reachable(&self, id: Id) -> bool {
        self.is_public_at_level(id, Level::Reachable)
    }

    /// See `Level::Reexported`.
    pub fn is_exported(&self, id: Id) -> bool {
        self.is_public_at_level(id, Level::Reexported)
    }

    /// See `Level::Direct`.
    pub fn is_directly_public(&self, id: Id) -> bool {
        self.is_public_at_level(id, Level::Direct)
    }

    pub fn public_at_level(&self, id: Id) -> Option<Level> {
        self.effective_vis(id).and_then(|effective_vis| {
            for level in Level::all_levels() {
                if effective_vis.is_public_at_level(level) {
                    return Some(level);
                }
            }
            None
        })
    }

    pub fn effective_vis(&self, id: Id) -> Option<&EffectiveVisibility> {
        self.map.get(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Id, &EffectiveVisibility)> {
        self.map.iter()
    }

    pub fn map_id<OutId: Hash + Eq + Copy>(
        &self,
        f: impl Fn(Id) -> OutId,
    ) -> EffectiveVisibilities<OutId> {
        EffectiveVisibilities { map: self.map.iter().map(|(k, v)| (f(*k), *v)).collect() }
    }

    pub fn set_public_at_level(
        &mut self,
        id: Id,
        default_vis: impl FnOnce() -> Visibility,
        level: Level,
    ) {
        let mut effective_vis = self
            .effective_vis(id)
            .copied()
            .unwrap_or_else(|| EffectiveVisibility::from_vis(default_vis()));
        for l in Level::all_levels() {
            if l <= level {
                *effective_vis.at_level_mut(l) = Visibility::Public;
            }
        }
        self.map.insert(id, effective_vis);
    }
}

impl<Id: Hash + Eq + Copy + Into<DefId>> EffectiveVisibilities<Id> {
    // `parent_id` is not necessarily a parent in source code tree,
    // it is the node from which the maximum effective visibility is inherited.
    pub fn update(
        &mut self,
        id: Id,
        nominal_vis: Visibility,
        default_vis: impl FnOnce() -> Visibility,
        parent_id: Id,
        level: Level,
        tree: impl DefIdTree,
    ) -> bool {
        let mut changed = false;
        let mut current_effective_vis = self.effective_vis(id).copied().unwrap_or_else(|| {
            if id.into().is_crate_root() {
                EffectiveVisibility::from_vis(Visibility::Public)
            } else {
                EffectiveVisibility::from_vis(default_vis())
            }
        });
        if let Some(inherited_effective_vis) = self.effective_vis(parent_id) {
            let mut inherited_effective_vis_at_prev_level =
                *inherited_effective_vis.at_level(level);
            let mut calculated_effective_vis = inherited_effective_vis_at_prev_level;
            for l in Level::all_levels() {
                if level >= l {
                    let inherited_effective_vis_at_level = *inherited_effective_vis.at_level(l);
                    let current_effective_vis_at_level = current_effective_vis.at_level_mut(l);
                    // effective visibility for id shouldn't be recalculated if
                    // inherited from parent_id effective visibility isn't changed at next level
                    if !(inherited_effective_vis_at_prev_level == inherited_effective_vis_at_level
                        && level != l)
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

impl<Id> Default for EffectiveVisibilities<Id> {
    fn default() -> Self {
        EffectiveVisibilities { map: Default::default() }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for EffectiveVisibilities {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let EffectiveVisibilities { ref map } = *self;
        map.hash_stable(hcx, hasher);
    }
}
