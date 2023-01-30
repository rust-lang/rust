//! Dealing with trait goals, i.e. `T: Trait<'a, U>`.

use std::iter;

use super::assembly::{self, Candidate, CandidateSource};
use super::infcx_ext::InferCtxtExt;
use super::{Certainty, EvalCtxt, Goal, QueryResult};
use rustc_hir::def_id::DefId;
use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::query::NoSolution;
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams};
use rustc_middle::ty::{self, ToPredicate, Ty, TyCtxt};
use rustc_middle::ty::{TraitPredicate, TypeVisitable};
use rustc_span::DUMMY_SP;

pub mod structural_traits;

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
    ) -> QueryResult<'tcx> {
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
            ecx.evaluate_all_and_make_canonical_response(nested_goals)
        })
    }

    fn consider_assumption(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        assumption: ty::Predicate<'tcx>,
    ) -> QueryResult<'tcx> {
        if let Some(poly_trait_pred) = assumption.to_opt_poly_trait_pred()
            && poly_trait_pred.def_id() == goal.predicate.def_id()
        {
            // FIXME: Constness and polarity
            ecx.infcx.probe(|_| {
                let assumption_trait_pred =
                    ecx.infcx.instantiate_bound_vars_with_infer(poly_trait_pred);
                let nested_goals = ecx.infcx.eq(
                    goal.param_env,
                    goal.predicate.trait_ref,
                    assumption_trait_pred.trait_ref,
                )?;
                ecx.evaluate_all_and_make_canonical_response(nested_goals)
            })
        } else {
            Err(NoSolution)
        }
    }

    fn consider_auto_trait_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        ecx.probe_and_evaluate_goal_for_constituent_tys(
            goal,
            structural_traits::instantiate_constituent_tys_for_auto_trait,
        )
    }

    fn consider_trait_alias_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        let tcx = ecx.tcx();

        ecx.infcx.probe(|_| {
            let nested_obligations = tcx
                .predicates_of(goal.predicate.def_id())
                .instantiate(tcx, goal.predicate.trait_ref.substs);
            ecx.evaluate_all_and_make_canonical_response(
                nested_obligations.predicates.into_iter().map(|p| goal.with(tcx, p)).collect(),
            )
        })
    }

    fn consider_builtin_sized_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        ecx.probe_and_evaluate_goal_for_constituent_tys(
            goal,
            structural_traits::instantiate_constituent_tys_for_sized_trait,
        )
    }

    fn consider_builtin_copy_clone_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        ecx.probe_and_evaluate_goal_for_constituent_tys(
            goal,
            structural_traits::instantiate_constituent_tys_for_copy_clone_trait,
        )
    }

    fn consider_builtin_pointer_sized_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.self_ty().has_non_region_infer() {
            return ecx.make_canonical_response(Certainty::AMBIGUOUS);
        }

        let tcx = ecx.tcx();
        let self_ty = tcx.erase_regions(goal.predicate.self_ty());

        if let Ok(layout) = tcx.layout_of(goal.param_env.and(self_ty))
            &&  let usize_layout = tcx.layout_of(ty::ParamEnv::empty().and(tcx.types.usize)).unwrap().layout
            && layout.layout.size() == usize_layout.size()
            && layout.layout.align().abi == usize_layout.align().abi
        {
            // FIXME: We could make this faster by making a no-constraints response
            ecx.make_canonical_response(Certainty::Yes)
        } else {
            Err(NoSolution)
        }
    }

    fn consider_builtin_fn_trait_candidates(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        goal_kind: ty::ClosureKind,
    ) -> QueryResult<'tcx> {
        if let Some(tupled_inputs_and_output) =
            structural_traits::extract_tupled_inputs_and_output_from_callable(
                ecx.tcx(),
                goal.predicate.self_ty(),
                goal_kind,
            )?
        {
            let pred = tupled_inputs_and_output
                .map_bound(|(inputs, _)| {
                    ecx.tcx()
                        .mk_trait_ref(goal.predicate.def_id(), [goal.predicate.self_ty(), inputs])
                })
                .to_predicate(ecx.tcx());
            Self::consider_assumption(ecx, goal, pred)
        } else {
            ecx.make_canonical_response(Certainty::AMBIGUOUS)
        }
    }

    fn consider_builtin_tuple_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if let ty::Tuple(..) = goal.predicate.self_ty().kind() {
            ecx.make_canonical_response(Certainty::Yes)
        } else {
            Err(NoSolution)
        }
    }

    fn consider_builtin_pointee_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        _goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        ecx.make_canonical_response(Certainty::Yes)
    }

    fn consider_builtin_future_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        let ty::Generator(def_id, _, _) = *goal.predicate.self_ty().kind() else {
            return Err(NoSolution);
        };

        // Generators are not futures unless they come from `async` desugaring
        let tcx = ecx.tcx();
        if !tcx.generator_is_async(def_id) {
            return Err(NoSolution);
        }

        // Async generator unconditionally implement `Future`
        ecx.make_canonical_response(Certainty::Yes)
    }

    fn consider_builtin_generator_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        let self_ty = goal.predicate.self_ty();
        let ty::Generator(def_id, substs, _) = *self_ty.kind() else {
            return Err(NoSolution);
        };

        // `async`-desugared generators do not implement the generator trait
        let tcx = ecx.tcx();
        if tcx.generator_is_async(def_id) {
            return Err(NoSolution);
        }

        let generator = substs.as_generator();
        Self::consider_assumption(
            ecx,
            goal,
            ty::Binder::dummy(
                tcx.mk_trait_ref(goal.predicate.def_id(), [self_ty, generator.resume_ty()]),
            )
            .to_predicate(tcx),
        )
    }
}

impl<'tcx> EvalCtxt<'_, 'tcx> {
    /// Convenience function for traits that are structural, i.e. that only
    /// have nested subgoals that only change the self type. Unlike other
    /// evaluate-like helpers, this does a probe, so it doesn't need to be
    /// wrapped in one.
    fn probe_and_evaluate_goal_for_constituent_tys(
        &mut self,
        goal: Goal<'tcx, TraitPredicate<'tcx>>,
        constituent_tys: impl Fn(&InferCtxt<'tcx>, Ty<'tcx>) -> Result<Vec<Ty<'tcx>>, NoSolution>,
    ) -> QueryResult<'tcx> {
        self.infcx.probe(|_| {
            self.evaluate_all_and_make_canonical_response(
                constituent_tys(self.infcx, goal.predicate.self_ty())?
                    .into_iter()
                    .map(|ty| {
                        goal.with(
                            self.tcx(),
                            ty::Binder::dummy(goal.predicate.with_self_ty(self.tcx(), ty)),
                        )
                    })
                    .collect(),
            )
        })
    }

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
            | (CandidateSource::AliasBound, _)
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
