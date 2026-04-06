use rustc_data_structures::fx::FxIndexSet;
use rustc_hir::def_id::{DefId, LocalDefIdMap, LocalDefIdSet};
use rustc_macros::HashStable;

/// A single snapshot of dead-code liveness analysis state.
#[derive(Clone, Debug, HashStable)]
pub struct DeadCodeLivenessSnapshot {
    pub live_symbols: LocalDefIdSet,
    /// Maps each ADT to derived traits (for example `Debug` and `Clone`) that should be ignored
    /// when checking for dead code diagnostics.
    pub ignored_derived_traits: LocalDefIdMap<FxIndexSet<DefId>>,
}

/// Dead-code liveness data across primary and deferred seeding.
///
/// `pre_deferred_seeding` is computed after the initial analysis pass but before deferred seeds
/// are added.
/// `final_result` is the liveness snapshot after all seeds have been processed.
#[derive(Clone, Debug, HashStable)]
pub struct DeadCodeLivenessSummary {
    pub pre_deferred_seeding: DeadCodeLivenessSnapshot,
    pub final_result: DeadCodeLivenessSnapshot,
}
