use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::{DefId, LocalDefId, LocalDefIdMap};

use crate::ty::SimplifiedType;

/// A map for the local crate mapping each type to a vector of its
/// inherent impls. This is not meant to be used outside of coherence;
/// rather, you should request the vector for a specific type via
/// `tcx.inherent_impls(def_id)` so as to minimize your dependencies
/// (constructing this map requires touching the entire crate).
#[derive(Clone, Debug, Default, HashStable)]
pub struct CrateInherentImpls {
    pub inherent_impls: LocalDefIdMap<Vec<DefId>>,
    pub incoherent_impls: FxHashMap<SimplifiedType, Vec<LocalDefId>>,
}
