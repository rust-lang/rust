use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, OutlivesClause, TyCtxt};

use super::utils::*;

#[derive(Debug)]
pub(crate) struct ExplicitClausesMap<'tcx> {
    map: FxIndexMap<DefId, ty::EarlyBinder<'tcx, RequiredClauses<'tcx>>>,
}

impl<'tcx> ExplicitClausesMap<'tcx> {
    pub(crate) fn new() -> ExplicitClausesMap<'tcx> {
        ExplicitClausesMap { map: FxIndexMap::default() }
    }

    pub(crate) fn explicit_clauses_of(
        &mut self,
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
    ) -> &ty::EarlyBinder<'tcx, RequiredClauses<'tcx>> {
        self.map.entry(def_id).or_insert_with(|| {
            let gen_clauses = if def_id.is_local() {
                tcx.explicit_clauses_of(def_id)
            } else {
                tcx.clauses_of(def_id)
            };
            let mut required_clauses = RequiredClauses::default();

            // Process clauses and convert to `RequiredClauses` entry, see below.
            for &(clause, span) in gen_clauses.clauses {
                match clause.kind().skip_binder() {
                    ty::ClauseKind::TypeOutlives(OutlivesClause(ty, reg)) => {
                        insert_outlives_clause(tcx, ty.into(), reg, span, &mut required_clauses)
                    }

                    ty::ClauseKind::RegionOutlives(OutlivesClause(reg1, reg2)) => {
                        insert_outlives_clause(tcx, reg1.into(), reg2, span, &mut required_clauses)
                    }
                    ty::ClauseKind::Trait(_)
                    | ty::ClauseKind::Projection(_)
                    | ty::ClauseKind::ConstArgHasType(_, _)
                    | ty::ClauseKind::WellFormed(_)
                    | ty::ClauseKind::ConstEvaluatable(_)
                    | ty::ClauseKind::UnstableFeature(_)
                    | ty::ClauseKind::HostEffect(..)
                    | ty::ClauseKind::CoroutineWitnessRegionConstraints(..) => {}
                }
            }

            ty::EarlyBinder::bind_iter(required_clauses)
        })
    }
}
