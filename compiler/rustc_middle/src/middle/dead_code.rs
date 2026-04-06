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

/// Dead-code liveness data for both analysis phases.
///
/// `pre_deferred_seeding` is computed before reachable-public and `#[allow(dead_code)]` seeding,
/// and is used for lint `unused_pub_items_in_binary`.
/// `final_result` is the final liveness snapshot used for lint `dead_code`.
#[derive(Clone, Debug, HashStable)]
pub struct DeadCodeLivenessSummary {
    pub pre_deferred_seeding: DeadCodeLivenessSnapshot,
    pub final_result: DeadCodeLivenessSnapshot,
}
