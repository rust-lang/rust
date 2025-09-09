use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::unord::UnordMap;
use rustc_macros::{HashStable, TyDecodable, TyEncodable};
use rustc_query_system::ich::StableHashingContext;
use rustc_span::{Span, Symbol};

#[derive(Clone, Debug, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub struct RemovedLibFeatureInfo {
    /// Reason (short free-form text).
    pub reason: Symbol,
    /// `since = "..."` version string.
    pub since: Symbol,
    /// Span where the `#[unstable_removed]` attribute was declared (crate root).
    pub span: Span,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[derive(HashStable, TyEncodable, TyDecodable)]
pub enum FeatureStability {
    AcceptedSince(Symbol),
    Unstable { old_name: Option<Symbol> },
    Removed { since: Symbol, reason: Symbol },
}

#[derive(Clone, Debug, Default)]
pub struct LibFeatures {
    /// `#[stable(feature = "...", since = "...")]`
    pub stable: UnordMap<Symbol, Span>,
    /// `#[unstable(feature = "...", issue = "...")]`
    pub unstable: UnordMap<Symbol, Span>,
    /// `#![unstable_removed(...)]`
    pub removed: UnordMap<Symbol, RemovedLibFeatureInfo>,
    /// Internal stability tracking
    pub stability: UnordMap<Symbol, (FeatureStability, Span)>,
}

impl LibFeatures {
    pub fn to_sorted_vec(&self) -> Vec<(Symbol, FeatureStability)> {
        self.stability
            .to_sorted_stable_ord()
            .iter()
            .map(|&(&sym, &(stab, _))| (sym, stab))
            .collect()
    }

    pub fn get_removed(&self, name: &Symbol) -> Option<&RemovedLibFeatureInfo> {
        self.removed.get(name)
    }
}

impl HashStable<StableHashingContext<'_>> for LibFeatures {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'_>, hasher: &mut StableHasher) {
        // Hash each UnordMap deterministically
        self.stable.to_sorted_stable_ord().hash_stable(hcx, hasher);
        self.unstable.to_sorted_stable_ord().hash_stable(hcx, hasher);
        self.removed.to_sorted_stable_ord().hash_stable(hcx, hasher);
        self.stability.to_sorted_stable_ord().hash_stable(hcx, hasher);
    }
}
