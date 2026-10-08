use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, OutlivesClause, TyCtxt};

use super::utils::*;

#[derive(Default, Debug)]
pub(crate) struct GlobalExplicitOutlivesClauses<'tcx> {
    map: GlobalOutlivesClauses<'tcx>,
}

impl<'tcx> GlobalExplicitOutlivesClauses<'tcx> {
    pub(crate) fn explicit_outlives_clauses_of(
        &mut self,
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
    ) -> &ty::EarlyBinder<'tcx, OutlivesClauses<'tcx>> {
        self.map.entry(def_id).or_insert_with(|| {
            let clauses = if def_id.is_local() {
                tcx.explicit_clauses_of(def_id)
            } else {
                tcx.clauses_of(def_id)
            };
            let mut outlives_clauses = OutlivesClauses::default();

            for &(clause, span) in clauses.clauses {
                match clause.kind().skip_binder() {
                    ty::ClauseKind::TypeOutlives(OutlivesClause(ty, reg)) => {
                        insert_outlives_clause(tcx, ty.into(), reg, span, &mut outlives_clauses)
                    }

                    ty::ClauseKind::RegionOutlives(OutlivesClause(reg1, reg2)) => {
                        insert_outlives_clause(tcx, reg1.into(), reg2, span, &mut outlives_clauses)
                    }
                    ty::ClauseKind::Trait(_)
                    | ty::ClauseKind::Projection(_)
                    | ty::ClauseKind::ConstArgHasType(_, _)
                    | ty::ClauseKind::WellFormed(_)
                    | ty::ClauseKind::ConstEvaluatable(_)
                    | ty::ClauseKind::UnstableFeature(_)
                    | ty::ClauseKind::HostEffect(..) => {}
                }
            }

            ty::EarlyBinder::bind_iter(outlives_clauses)
        })
    }
}
