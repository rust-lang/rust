use smallvec::{smallvec, SmallVec};

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, TyCtxt};

pub struct Elaborator<'tcx> {
    tcx: TyCtxt<'tcx>,
    stack: SmallVec<[(DefId, SmallVec<[ty::BoundVariableKind; 8]>); 8]>,
    visited: FxHashSet<DefId>,
}

#[allow(dead_code)]
pub fn supertraits<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> Elaborator<'tcx> {
    Elaborator { tcx, stack: smallvec![(def_id, smallvec![])], visited: Default::default() }
}

impl<'tcx> Elaborator<'tcx> {
    fn elaborate(&mut self, def_id: DefId, bound_vars: &SmallVec<[ty::BoundVariableKind; 8]>) {
        let tcx = self.tcx;

        let predicates = tcx.super_predicates_of(def_id);
        let obligations = predicates.predicates.iter().filter_map(|&(pred, _)| {
            let bound_predicate = pred.kind();
            match bound_predicate.skip_binder() {
                ty::PredicateKind::Trait(data, _) => {
                    // The order here needs to match what we would get from `subst_supertrait`
                    let pred_bound_vars = bound_predicate.bound_vars();
                    let mut all_bound_vars = bound_vars.clone();
                    all_bound_vars.extend(pred_bound_vars.iter());
                    let super_def_id = data.trait_ref.def_id;
                    Some((super_def_id, all_bound_vars))
                }
                _ => None,
            }
        });

        let visited = &mut self.visited;
        let obligations = obligations.filter(|o| visited.insert(o.0));
        self.stack.extend(obligations);
    }
}

impl<'tcx> Iterator for Elaborator<'tcx> {
    type Item = (DefId, SmallVec<[ty::BoundVariableKind; 8]>);

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.stack.len(), None)
    }

    fn next(&mut self) -> Option<Self::Item> {
        match self.stack.pop() {
            Some((def_id, bound_vars)) => {
                self.elaborate(def_id, &bound_vars);
                Some((def_id, bound_vars))
            }
            None => None,
        }
    }
}
