use std::mem;

use rustc_data_structures::fx::FxHashMap;
use rustc_infer::{
    infer::InferCtxt,
    traits::{query::NoSolution, FulfillmentError, PredicateObligation, TraitEngine},
};
use rustc_middle::ty;

use super::{Certainty, EvalCtxt};

/// A trait engine using the new trait solver.
///
/// This is mostly identical to how `evaluate_all` works inside of the
/// solver, except that the requirements are slightly different.
///
/// Unlike `evaluate_all` it is possible to add new obligations later on
/// and we also have to track diagnostics information by using `Obligation`
/// instead of `Goal`.
///
/// It is also likely that we want to use slightly different datastructures
/// here as this will have to deal with far more root goals than `evaluate_all`.
pub struct FulfillmentCtxt<'tcx> {
    obligations: Vec<PredicateObligation<'tcx>>,
}

impl<'tcx> FulfillmentCtxt<'tcx> {
    pub fn new() -> FulfillmentCtxt<'tcx> {
        FulfillmentCtxt { obligations: Vec::new() }
    }
}

impl<'tcx> TraitEngine<'tcx> for FulfillmentCtxt<'tcx> {
    fn register_predicate_obligation(
        &mut self,
        _infcx: &InferCtxt<'tcx>,
        obligation: PredicateObligation<'tcx>,
    ) {
        self.obligations.push(obligation);
    }

    fn select_all_or_error(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<FulfillmentError<'tcx>> {
        let errors = self.select_where_possible(infcx);
        if !errors.is_empty() {
            return errors;
        }

        if self.obligations.is_empty() {
            Vec::new()
        } else {
            unimplemented!("ambiguous obligations")
        }
    }

    fn select_where_possible(&mut self, infcx: &InferCtxt<'tcx>) -> Vec<FulfillmentError<'tcx>> {
        let errors = Vec::new();
        for i in 0.. {
            if !infcx.tcx.recursion_limit().value_within_limit(i) {
                unimplemented!("overflow")
            }

            let mut has_changed = false;
            for o in mem::take(&mut self.obligations) {
                let mut cx = EvalCtxt::new(infcx.tcx);
                let (changed, certainty) = match cx.evaluate_goal(infcx, o.clone().into()) {
                    Ok(result) => result,
                    Err(NoSolution) => unimplemented!("error"),
                };

                has_changed |= changed;
                match certainty {
                    Certainty::Yes => {}
                    Certainty::Maybe(_) => self.obligations.push(o),
                }
            }

            if !has_changed {
                break;
            }
        }

        errors
    }

    fn pending_obligations(&self) -> Vec<PredicateObligation<'tcx>> {
        self.obligations.clone()
    }

    fn relationships(&mut self) -> &mut FxHashMap<ty::TyVid, ty::FoundRelationships> {
        unimplemented!("Should be moved out of `TraitEngine`")
    }
}
