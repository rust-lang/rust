//! For the NLL computation, we need to compute liveness, but only for those
//! local variables whose types contain regions. The others are not of interest
//! to us. This file defines a new index type (LiveVar) that indexes into
//! a list of "variables whose type contain regions". It also defines a map from
//! Local to LiveVar and vice versa -- this map can be given to the
//! liveness code so that it only operates over variables with regions in their
//! types, instead of all variables.

use crate::borrow_check::nll::ToRegionVid;
use crate::borrow_check::nll::facts::{AllFacts, AllFactsExt};
use crate::util::liveness::LiveVariableMap;
use rustc::mir::{Local, Mir};
use rustc::ty::{RegionVid, TyCtxt};
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};

/// Map between Local and LiveVar indices: the purpose of this
/// map is to define the subset of local variables for which we need
/// to do a liveness computation. We only need to compute whether a
/// variable `X` is live if that variable contains some region `R` in
/// its type where `R` is not known to outlive a free region (i.e.,
/// where `R` may be valid for just a subset of the fn body).
crate struct NllLivenessMap {
    /// For each local variable, contains `Some(i)` if liveness is
    /// needed for this variable.
    pub from_local: IndexVec<Local, Option<LiveVar>>,

    /// For each `LiveVar`, maps back to the original `Local` index.
    pub to_local: IndexVec<LiveVar, Local>,
}

impl LiveVariableMap for NllLivenessMap {
    fn from_local(&self, local: Local) -> Option<Self::LiveVar> {
        self.from_local[local]
    }

    type LiveVar = LiveVar;

    fn from_live_var(&self, local: Self::LiveVar) -> Local {
        self.to_local[local]
    }

    fn num_variables(&self) -> usize {
        self.to_local.len()
    }
}

impl NllLivenessMap {
    crate fn compute(
        tcx: TyCtxt<'_, '_, 'tcx>,
        free_regions: &FxHashSet<RegionVid>,
        mir: &Mir<'tcx>,
    ) -> Self {
        let mut to_local = IndexVec::default();
        let facts_enabled = AllFacts::enabled(tcx);
        let from_local: IndexVec<Local, Option<_>> = mir.local_decls
            .iter_enumerated()
            .map(|(local, local_decl)| {
                if tcx.all_free_regions_meet(&local_decl.ty, |r| {
                    free_regions.contains(&r.to_region_vid())
                }) && !facts_enabled {
                    // If all the regions in the type are free regions
                    // (or there are no regions), then we don't need
                    // to track liveness for this variable.
                    None
                } else {
                    Some(to_local.push(local))
                }
            })
            .collect();

        debug!("{} total variables", mir.local_decls.len());
        debug!("{} variables need liveness", to_local.len());
        debug!("{} regions outlive free regions", free_regions.len());

        Self {
            from_local,
            to_local,
        }
    }

    /// True if there are no local variables that need liveness computation.
    crate fn is_empty(&self) -> bool {
        self.to_local.is_empty()
    }
}

/// Index given to each local variable for which we need to
/// compute liveness information. For many locals, we are able to
/// skip liveness information: for example, those variables whose
/// types contain no regions.
newtype_index! {
    pub struct LiveVar { .. }
}
