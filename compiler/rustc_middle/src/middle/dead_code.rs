use rustc_hir::def_id::{LocalDefId, LocalDefIdMap, LocalDefIdSet};
use rustc_macros::StableHash;
use rustc_span::Span;

/// A single snapshot of dead-code liveness analysis state.
#[derive(Clone, Debug, StableHash)]
pub struct DeadCodeLivenessSnapshot {
    pub live: LocalDefIdSet,
    /// This maps defs that were only used from items annotated with
    /// `#[rustc_trivial_field_reads]` to those items
    pub sites_by_defid: LocalDefIdMap<Vec<LocalDefId>>,
    /// For each of the items annotated with `#[rustc_trivial_field_reads]`,
    /// we store the attr span here, so we can use it for diagnostics if needed
    pub sites: LocalDefIdMap<Span>,
}

impl DeadCodeLivenessSnapshot {
    pub fn is_live(&self, id: &LocalDefId) -> bool {
        self.live.contains(id)
    }

    pub fn trivial_usage_sites(&self, def_id: &LocalDefId) -> impl Iterator<Item = LocalDefId> {
        self.sites_by_defid.get(def_id).into_iter().flatten().copied()
    }

    pub fn trivial_usage_site_span_by_id(&self, site_id: LocalDefId) -> Span {
        *self.sites.get(&site_id).unwrap()
    }
}

/// Dead-code liveness data for both analysis phases.
///
/// `pre_deferred_seeding` is computed before reachable-public and `#[allow(dead_code)]` seeding,
/// and is used for lint `dead_code_pub_in_binary`.
/// `final_result` is the final liveness snapshot used for lint `dead_code`.
#[derive(Clone, Debug, StableHash)]
pub struct DeadCodeLivenessSummary {
    pub pre_deferred_seeding: DeadCodeLivenessSnapshot,
    pub final_result: DeadCodeLivenessSnapshot,
}
