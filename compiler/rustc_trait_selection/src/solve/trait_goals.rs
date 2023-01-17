//! Dealing with trait goals, i.e. `T: Trait<'a, U>`.

use std::iter;

use super::assembly::{self, Candidate, CandidateSource};
use super::infcx_ext::InferCtxtExt;
use super::{Certainty, EvalCtxt, Goal, QueryResult};
use rustc_hir::def_id::DefId;
use rustc_infer::traits::query::NoSolution;
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams};
use rustc_middle::ty::TraitPredicate;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::DUMMY_SP;

impl<'tcx> assembly::GoalKind<'tcx> for TraitPredicate<'tcx> {
    fn self_ty(self) -> Ty<'tcx> {
        self.self_ty()
    }

    fn with_self_ty(self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> Self {
        self.with_self_ty(tcx, self_ty)
    }

    fn trait_def_id(self, _: TyCtxt<'tcx>) -> DefId {
        self.def_id()
    }

    fn consider_impl_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, TraitPredicate<'tcx>>,
        impl_def_id: DefId,
    ) -> Result<Certainty, NoSolution> {
        let tcx = ecx.tcx();

        let impl_trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
        let drcx = DeepRejectCtxt { treat_obligation_params: TreatParams::AsPlaceholder };
        if iter::zip(goal.predicate.trait_ref.substs, impl_trait_ref.skip_binder().substs)
            .any(|(goal, imp)| !drcx.generic_args_may_unify(goal, imp))
        {
            return Err(NoSolution);
        }

        ecx.infcx.probe(|_| {
            let impl_substs = ecx.infcx.fresh_substs_for_item(DUMMY_SP, impl_def_id);
            let impl_trait_ref = impl_trait_ref.subst(tcx, impl_substs);

            let mut nested_goals =
                ecx.infcx.eq(goal.param_env, goal.predicate.trait_ref, impl_trait_ref)?;
            let where_clause_bounds = tcx
                .predicates_of(impl_def_id)
                .instantiate(tcx, impl_substs)
                .predicates
                .into_iter()
                .map(|pred| goal.with(tcx, pred));
            nested_goals.extend(where_clause_bounds);
            ecx.evaluate_all(nested_goals)
        })
    }

    fn consider_builtin_sized_candidate(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        _goal: Goal<'tcx, Self>,
    ) -> Result<Certainty, NoSolution> {
        unimplemented!();
    }

    fn consider_assumption(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        _goal: Goal<'tcx, Self>,
        assumption: ty::Predicate<'tcx>,
    ) -> Result<Certainty, NoSolution> {
        if let Some(_poly_trait_pred) = assumption.to_opt_poly_trait_pred() {
            unimplemented!()
        } else {
            Err(NoSolution)
        }
    }
}

impl<'tcx> EvalCtxt<'_, 'tcx> {
    pub(super) fn compute_trait_goal(
        &mut self,
        goal: Goal<'tcx, TraitPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        let candidates = self.assemble_and_evaluate_candidates(goal);
        self.merge_trait_candidates_discard_reservation_impls(candidates)
    }

    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn merge_trait_candidates_discard_reservation_impls(
        &mut self,
        mut candidates: Vec<Candidate<'tcx>>,
    ) -> QueryResult<'tcx> {
        match candidates.len() {
            0 => return Err(NoSolution),
            1 => return Ok(self.discard_reservation_impl(candidates.pop().unwrap()).result),
            _ => {}
        }

        if candidates.len() > 1 {
            let mut i = 0;
            'outer: while i < candidates.len() {
                for j in (0..candidates.len()).filter(|&j| i != j) {
                    if self.trait_candidate_should_be_dropped_in_favor_of(
                        &candidates[i],
                        &candidates[j],
                    ) {
                        debug!(candidate = ?candidates[i], "Dropping candidate #{}/{}", i, candidates.len());
                        candidates.swap_remove(i);
                        continue 'outer;
                    }
                }

                debug!(candidate = ?candidates[i], "Retaining candidate #{}/{}", i, candidates.len());
                // If there are *STILL* multiple candidates, give up
                // and report ambiguity.
                i += 1;
                if i > 1 {
                    debug!("multiple matches, ambig");
                    // FIXME: return overflow if all candidates overflow, otherwise return ambiguity.
                    unimplemented!();
                }
            }
        }

        Ok(self.discard_reservation_impl(candidates.pop().unwrap()).result)
    }

    fn trait_candidate_should_be_dropped_in_favor_of(
        &self,
        candidate: &Candidate<'tcx>,
        other: &Candidate<'tcx>,
    ) -> bool {
        // FIXME: implement this
        match (candidate.source, other.source) {
            (CandidateSource::Impl(_), _)
            | (CandidateSource::ParamEnv(_), _)
            | (CandidateSource::AliasBound(_), _)
            | (CandidateSource::BuiltinImpl, _) => unimplemented!(),
        }
    }

    fn discard_reservation_impl(&self, candidate: Candidate<'tcx>) -> Candidate<'tcx> {
        if let CandidateSource::Impl(def_id) = candidate.source {
            if let ty::ImplPolarity::Reservation = self.tcx().impl_polarity(def_id) {
                debug!("Selected reservation impl");
                // FIXME: reduce candidate to ambiguous
                // FIXME: replace `var_values` with identity, yeet external constraints.
                unimplemented!()
            }
        }

        candidate
    }
}
