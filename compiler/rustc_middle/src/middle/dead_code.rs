use rustc_hir::HirId;
use rustc_hir::def_id::{LocalDefId, LocalDefIdMap, LocalDefIdSet};
use rustc_macros::StableHash;
use rustc_span::Span;

#[derive(Clone, Debug, StableHash, PartialEq, Eq, Hash)]
pub enum LiveKind {
    Trivial,
    NonTrivial,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, StableHash, PartialOrd, Ord)]
pub struct TrivialUsageSiteId(usize);

#[derive(Debug, Clone, StableHash)]
pub struct TrivialUsageSite {
    pub attr_span: Span,
    pub hir_id: HirId,
}

#[derive(Clone, Debug, StableHash, Default)]
pub struct TrivialUsageSites(Vec<TrivialUsageSite>);

/// A single snapshot of dead-code liveness analysis state.
#[derive(Clone, Debug, StableHash)]
pub struct DeadCodeLivenessSnapshot {
    pub live: LocalDefIdSet,
    /// This maps defs that were only used from items annotated with
    /// `#[rustc_trivial_field_reads]` to those items
    pub sites_by_defid: LocalDefIdMap<Vec<TrivialUsageSiteId>>,
    /// List of all items annotated with `#[rustc_trivial_field_reads]`
    pub sites: TrivialUsageSites,
}

impl TrivialUsageSites {
    pub fn add_trivial_usage_site(&mut self, site: TrivialUsageSite) -> TrivialUsageSiteId {
        self.0.push(site);
        TrivialUsageSiteId(self.0.len() - 1)
    }
}

impl DeadCodeLivenessSnapshot {
    pub fn is_live(&self, id: &LocalDefId) -> bool {
        self.live.contains(id)
    }

    pub fn trivial_usage_sites(
        &self,
        def_id: &LocalDefId,
    ) -> impl Iterator<Item = &TrivialUsageSite> {
        self.sites_by_defid
            .get(def_id)
            .into_iter()
            .flatten()
            .map(|site_id| &self.sites.0[site_id.0])
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
