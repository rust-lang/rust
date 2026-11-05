use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::HirId;
use rustc_hir::def_id::{LocalDefId, LocalDefIdMap, LocalDefIdSet};
use rustc_macros::StableHash;
use rustc_span::Span;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, StableHash, PartialOrd, Ord)]
pub struct TrivialUsageSiteId(usize);

#[derive(Debug, Clone, StableHash)]
pub struct TrivialUsageSite {
    pub attr_span: Span,
    pub hir_id: HirId,
}

/// Represents trivial usage sites, i.e. hir nodes annotated with
/// `rustc_trivial_field_reads`. Uses a map internally to deduplicate
/// items which are encountered multiple times during dead code analysis.
#[derive(Clone, Debug, StableHash, Default)]
pub struct TrivialUsageSites(FxIndexMap<HirId, Span>);

/// A single snapshot of dead-code liveness analysis state.
#[derive(Clone, Debug, StableHash)]
pub struct DeadCodeLivenessSnapshot {
    pub live: LocalDefIdSet,
    /// This maps defs that were only used from items annotated with
    /// `#[rustc_trivial_field_reads]` to those items
    pub sites_by_defid: LocalDefIdMap<Vec<TrivialUsageSiteId>>,
    /// All items annotated with `#[rustc_trivial_field_reads]`
    pub sites: TrivialUsageSites,
}

impl TrivialUsageSites {
    pub fn add_trivial_usage_site(&mut self, site: TrivialUsageSite) -> TrivialUsageSiteId {
        let (index, _) = self.0.insert_full(site.hir_id, site.attr_span);
        TrivialUsageSiteId(index)
    }
}

impl DeadCodeLivenessSnapshot {
    pub fn is_live(&self, id: &LocalDefId) -> bool {
        self.live.contains(id)
    }

    pub fn trivial_usage_sites(
        &self,
        def_id: &LocalDefId,
    ) -> impl Iterator<Item = TrivialUsageSiteId> {
        self.sites_by_defid.get(def_id).into_iter().flatten().copied()
    }

    pub fn trivial_usage_site_by_id(&self, site_id: TrivialUsageSiteId) -> TrivialUsageSite {
        let (hir_id, attr_span) = self.sites.0.get_index(site_id.0).unwrap();
        TrivialUsageSite { hir_id: *hir_id, attr_span: *attr_span }
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
