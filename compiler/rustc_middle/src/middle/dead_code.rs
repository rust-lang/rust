use std::collections::hash_map::Entry;

use rustc_data_structures::fx::FxIndexSet;
use rustc_hir::def_id::{DefId, LocalDefId, LocalDefIdMap};
use rustc_macros::StableHash;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, StableHash)]
pub enum Liveness {
    AllowExpect,
    Public,
    Live,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LivenessState {
    Unknown,
    Known(Liveness),
}

impl LivenessState {
    pub fn max(self, other: Liveness) -> Liveness {
        match self {
            LivenessState::Unknown => other,
            LivenessState::Known(known) => known.max(other),
        }
    }
}

#[derive(Default, Clone, Debug, StableHash)]
pub struct LiveSymbols {
    live: LocalDefIdMap<Liveness>,
}

impl LiveSymbols {
    /// Insert the given Liveness if one of the following conditions holds
    /// 1. the entry was not previously present
    /// 2. the new liveness is higher than the previous one
    /// Returns true if the entry was newly inserted or updated.
    pub fn insert(&mut self, def_id: LocalDefId, liveness: Liveness) -> bool {
        match self.live.entry(def_id) {
            Entry::Vacant(entry) => {
                entry.insert(liveness);
                true
            }
            Entry::Occupied(mut entry) => {
                if liveness > *entry.get() {
                    entry.insert(liveness);
                    true
                } else {
                    false
                }
            }
        }
    }

    pub fn get(&self, def_id: LocalDefId) -> LivenessState {
        match self.live.get(&def_id) {
            None => LivenessState::Unknown,
            Some(liveness) => LivenessState::Known(*liveness),
        }
    }

    pub fn extend(&mut self, iter: impl Iterator<Item = (LocalDefId, Liveness)>) {
        for (id, liveness) in iter {
            self.insert(id, liveness);
        }
    }

    pub fn is_live(&self, def_id: LocalDefId) -> bool {
        matches!(self.get(def_id), LivenessState::Known(Liveness::Live))
    }

    pub fn at_least_as_live(&self, def_id: LocalDefId, min_liveness: Liveness) -> bool {
        match self.get(def_id) {
            LivenessState::Unknown => false,
            LivenessState::Known(liveness) => liveness >= min_liveness,
        }
    }

    pub fn at_most_as_live(&self, def_id: LocalDefId, max_liveness: Liveness) -> bool {
        match self.get(def_id) {
            LivenessState::Unknown => true,
            LivenessState::Known(liveness) => liveness <= max_liveness,
        }
    }
}

impl FromIterator<(LocalDefId, Liveness)> for LiveSymbols {
    fn from_iter<I: IntoIterator<Item = (LocalDefId, Liveness)>>(iter: I) -> Self {
        let mut live = LiveSymbols::default();
        for (id, liveness) in iter.into_iter() {
            live.insert(id, liveness);
        }
        live
    }
}

/// Dead-code liveness data
#[derive(Clone, Debug, StableHash)]
pub struct DeadCodeLivenessSummary {
    pub live_symbols: LiveSymbols,
    pub ignored_derived_traits: LocalDefIdMap<FxIndexSet<DefId>>,
}
