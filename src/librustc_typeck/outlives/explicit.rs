use rustc::hir::def_id::DefId;
use rustc::ty::{self, OutlivesPredicate, TyCtxt};
use crate::util::nodemap::FxHashMap;

use super::utils::*;

#[derive(Debug)]
pub struct ExplicitPredicatesMap<'tcx> {
    map: FxHashMap<DefId, RequiredPredicates<'tcx>>,
}

impl<'tcx> ExplicitPredicatesMap<'tcx> {
    pub fn new() -> ExplicitPredicatesMap<'tcx> {
        ExplicitPredicatesMap {
            map: FxHashMap::default(),
        }
    }

    pub fn explicit_predicates_of(
        &mut self,
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
    ) -> &RequiredPredicates<'tcx> {
        self.map.entry(def_id).or_insert_with(|| {
            let predicates = if def_id.is_local() {
                tcx.explicit_predicates_of(def_id)
            } else {
                tcx.predicates_of(def_id)
            };
            let mut required_predicates = RequiredPredicates::default();

            // process predicates and convert to `RequiredPredicates` entry, see below
            for (pred, _) in predicates.predicates.iter() {
                match pred {
                    ty::Predicate::TypeOutlives(predicate) => {
                        let OutlivesPredicate(ref ty, ref reg) = predicate.skip_binder();
                        insert_outlives_predicate(tcx, (*ty).into(), reg, &mut required_predicates)
                    }

                    ty::Predicate::RegionOutlives(predicate) => {
                        let OutlivesPredicate(ref reg1, ref reg2) = predicate.skip_binder();
                        insert_outlives_predicate(
                            tcx,
                            (*reg1).into(),
                            reg2,
                            &mut required_predicates,
                        )
                    }

                    ty::Predicate::Trait(..)
                    | ty::Predicate::Projection(..)
                    | ty::Predicate::WellFormed(..)
                    | ty::Predicate::ObjectSafe(..)
                    | ty::Predicate::ClosureKind(..)
                    | ty::Predicate::Subtype(..)
                    | ty::Predicate::ConstEvaluatable(..) => (),
                }
            }

            required_predicates
        })
    }
}
