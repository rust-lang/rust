//! A pass that checks to make sure private fields and methods aren't used
//! outside their scopes. This pass will also generate a set of exported items
//! which are available for use externally when compiled as a library.
use crate::ty::{DefIdTree, TyCtxt, Visibility};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_macros::HashStable;
use rustc_query_system::ich::StableHashingContext;
use rustc_span::def_id::LocalDefId;
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
#[derive(Clone, Debug)]
pub struct EffectiveVisibilities<Id = LocalDefId> {
    map: FxHashMap<Id, EffectiveVisibility>,
}

impl EffectiveVisibilities {
    pub fn is_public_at_level(&self, id: LocalDefId, level: Level) -> bool {
        self.effective_vis(id)
            .map_or(false, |effective_vis| effective_vis.is_public_at_level(level))
    }

    /// See `Level::Reachable`.
    pub fn is_reachable(&self, id: LocalDefId) -> bool {
        self.is_public_at_level(id, Level::Reachable)
    }

    /// See `Level::Reexported`.
    pub fn is_exported(&self, id: LocalDefId) -> bool {
        self.is_public_at_level(id, Level::Reexported)
    }

    /// See `Level::Direct`.
    pub fn is_directly_public(&self, id: LocalDefId) -> bool {
        self.is_public_at_level(id, Level::Direct)
    }

    pub fn public_at_level(&self, id: LocalDefId) -> Option<Level> {
        self.effective_vis(id).and_then(|effective_vis| {
            Level::all_levels().into_iter().find(|&level| effective_vis.is_public_at_level(level))
        })
    }

    // FIXME: Share code with `fn update`.
    pub fn update_eff_vis(
        &mut self,
        def_id: LocalDefId,
        eff_vis: &EffectiveVisibility,
        tree: impl DefIdTree,
    ) {
        use std::collections::hash_map::Entry;
        match self.map.entry(def_id) {
            Entry::Occupied(mut occupied) => {
                let old_eff_vis = occupied.get_mut();
                for l in Level::all_levels() {
                    let vis_at_level = eff_vis.at_level(l);
                    let old_vis_at_level = old_eff_vis.at_level_mut(l);
                    if vis_at_level != old_vis_at_level
                        && vis_at_level.is_at_least(*old_vis_at_level, tree)
                    {
                        *old_vis_at_level = *vis_at_level
                    }
                }
                old_eff_vis
            }
            Entry::Vacant(vacant) => vacant.insert(*eff_vis),
        };
    }

    pub fn set_public_at_level(
        &mut self,
        id: LocalDefId,
        lazy_private_vis: impl FnOnce() -> Visibility,
        level: Level,
    ) {
        let mut effective_vis = self
            .effective_vis(id)
            .copied()
            .unwrap_or_else(|| EffectiveVisibility::from_vis(lazy_private_vis()));
        for l in Level::all_levels() {
            if l <= level {
                *effective_vis.at_level_mut(l) = Visibility::Public;
            }
        }
        self.map.insert(id, effective_vis);
    }

    pub fn check_invariants(&self, tcx: TyCtxt<'_>, early: bool) {
        if !cfg!(debug_assertions) {
            return;
        }
        for (&def_id, ev) in &self.map {
            // More direct visibility levels can never go farther than less direct ones,
            // neither of effective visibilities can go farther than nominal visibility,
            // and all effective visibilities are larger or equal than private visibility.
            let private_vis = Visibility::Restricted(tcx.parent_module_from_def_id(def_id));
            let span = tcx.def_span(def_id.to_def_id());
            if !ev.direct.is_at_least(private_vis, tcx) {
                span_bug!(span, "private {:?} > direct {:?}", private_vis, ev.direct);
            }
            if !ev.reexported.is_at_least(ev.direct, tcx) {
                span_bug!(span, "direct {:?} > reexported {:?}", ev.direct, ev.reexported);
            }
            if !ev.reachable.is_at_least(ev.reexported, tcx) {
                span_bug!(span, "reexported {:?} > reachable {:?}", ev.reexported, ev.reachable);
            }
            if !ev.reachable_through_impl_trait.is_at_least(ev.reachable, tcx) {
                span_bug!(
                    span,
                    "reachable {:?} > reachable_through_impl_trait {:?}",
                    ev.reachable,
                    ev.reachable_through_impl_trait
                );
            }
            let nominal_vis = tcx.visibility(def_id);
            // FIXME: `rustc_privacy` is not yet updated for the new logic and can set
            // effective visibilities that are larger than the nominal one.
            if !nominal_vis.is_at_least(ev.reachable_through_impl_trait, tcx) && early {
                span_bug!(
                    span,
                    "{:?}: reachable_through_impl_trait {:?} > nominal {:?}",
                    def_id,
                    ev.reachable_through_impl_trait,
                    nominal_vis
                );
            }
        }
    }
}

impl<Id: Eq + Hash> EffectiveVisibilities<Id> {
    pub fn iter(&self) -> impl Iterator<Item = (&Id, &EffectiveVisibility)> {
        self.map.iter()
    }

    pub fn effective_vis(&self, id: Id) -> Option<&EffectiveVisibility> {
        self.map.get(&id)
    }

    // FIXME: Share code with `fn update`.
    pub fn effective_vis_or_private(
        &mut self,
        id: Id,
        lazy_private_vis: impl FnOnce() -> Visibility,
    ) -> &EffectiveVisibility {
        self.map.entry(id).or_insert_with(|| EffectiveVisibility::from_vis(lazy_private_vis()))
    }

    pub fn update(
        &mut self,
        id: Id,
        nominal_vis: Visibility,
        lazy_private_vis: impl FnOnce() -> Visibility,
        inherited_effective_vis: EffectiveVisibility,
        level: Level,
        tree: impl DefIdTree,
    ) -> bool {
        let mut changed = false;
        let mut current_effective_vis = self
            .map
            .get(&id)
            .copied()
            .unwrap_or_else(|| EffectiveVisibility::from_vis(lazy_private_vis()));

        let mut inherited_effective_vis_at_prev_level = *inherited_effective_vis.at_level(level);
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
                    && calculated_effective_vis.is_at_least(*current_effective_vis_at_level, tree)
                {
                    changed = true;
                    *current_effective_vis_at_level = calculated_effective_vis;
                }
                inherited_effective_vis_at_prev_level = inherited_effective_vis_at_level;
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
