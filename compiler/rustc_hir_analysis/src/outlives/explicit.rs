use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, OutlivesPredicate, TyCtxt};

use super::utils::*;

#[derive(Debug)]
pub struct ExplicitPredicatesMap<'tcx> {
    map: FxHashMap<DefId, ty::EarlyBinder<RequiredPredicates<'tcx>>>,
}

impl<'tcx> ExplicitPredicatesMap<'tcx> {
    pub fn new() -> ExplicitPredicatesMap<'tcx> {
        ExplicitPredicatesMap { map: FxHashMap::default() }
    }

    pub(crate) fn explicit_predicates_of(
        &mut self,
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
    ) -> &ty::EarlyBinder<RequiredPredicates<'tcx>> {
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
                    ty::PredicateKind::Clause(ty::Clause::TypeOutlives(OutlivesPredicate(
                        ty,
                        reg,
                    ))) => insert_outlives_predicate(
                        tcx,
                        ty.into(),
                        reg,
                        span,
                        &mut required_predicates,
                    ),

                    ty::PredicateKind::Clause(ty::Clause::RegionOutlives(OutlivesPredicate(
                        reg1,
                        reg2,
                    ))) => insert_outlives_predicate(
                        tcx,
                        reg1.into(),
                        reg2,
                        span,
                        &mut required_predicates,
                    ),

                    ty::PredicateKind::Clause(ty::Clause::Trait(..))
                    | ty::PredicateKind::Clause(ty::Clause::Projection(..))
                    | ty::PredicateKind::Clause(ty::Clause::ConstArgHasType(..))
                    | ty::PredicateKind::WellFormed(..)
                    | ty::PredicateKind::AliasRelate(..)
                    | ty::PredicateKind::ObjectSafe(..)
                    | ty::PredicateKind::ClosureKind(..)
                    | ty::PredicateKind::Subtype(..)
                    | ty::PredicateKind::Coerce(..)
                    | ty::PredicateKind::ConstEvaluatable(..)
                    | ty::PredicateKind::ConstEquate(..)
                    | ty::PredicateKind::Ambiguous
                    | ty::PredicateKind::TypeWellFormedFromEnv(..) => (),
                }
            }

            ty::EarlyBinder(required_predicates)
        })
    }
}
