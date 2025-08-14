use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, OutlivesPredicate, TyCtxt};

use super::utils::*;

#[derive(Debug)]
pub(crate) struct ExplicitPredicatesMap<'tcx> {
    map: FxIndexMap<DefId, ty::EarlyBinder<'tcx, RequiredPredicates<'tcx>>>,
}

impl<'tcx> ExplicitPredicatesMap<'tcx> {
    pub(crate) fn new() -> ExplicitPredicatesMap<'tcx> {
        ExplicitPredicatesMap { map: FxIndexMap::default() }
    }

    pub(crate) fn explicit_predicates_of(
        &mut self,
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
    ) -> &ty::EarlyBinder<'tcx, RequiredPredicates<'tcx>> {
        self.map.entry(def_id).or_insert_with(|| {
            let predicates = if def_id.is_local() {
                tcx.explicit_predicates_of(def_id)
            } else {
                tcx.predicates_of(def_id)
            };
            let mut required_predicates = RequiredPredicates::default();

            // process predicates and convert to `RequiredPredicates` entry, see below
            for &(predicate, span) in predicates.predicates {
                match predicate.kind().skip_binder() {
                    ty::ClauseKind::TypeOutlives(OutlivesPredicate(ty, reg)) => {
                        insert_outlives_predicate(
                            tcx,
                            ty.into(),
                            reg,
                            span,
                            &mut required_predicates,
                        )
                    }

                    ty::ClauseKind::RegionOutlives(OutlivesPredicate(reg1, reg2)) => {
                        insert_outlives_predicate(
                            tcx,
                            reg1.into(),
                            reg2,
                            span,
                            &mut required_predicates,
                        )
                    }
                    ty::ClauseKind::Trait(_)
                    | ty::ClauseKind::Projection(_)
                    | ty::ClauseKind::ConstArgHasType(_, _)
                    | ty::ClauseKind::WellFormed(_)
                    | ty::ClauseKind::ConstEvaluatable(_)
                    | ty::ClauseKind::HostEffect(..) => {}
                }
            }

            ty::EarlyBinder::bind(required_predicates)
        })
    }
}
