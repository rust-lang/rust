//! Dealing with trait goals, i.e. `T: Trait<'a, U>`.

use std::iter;

use super::assembly::{self, Candidate, CandidateSource};
use super::infcx_ext::InferCtxtExt;
use super::{EvalCtxt, Goal, QueryResult};
use rustc_hir::def_id::DefId;
use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::query::NoSolution;
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{ToPolyTraitRef, TraitPredicate};
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
        if let Some(poly_trait_pred) = assumption.to_opt_poly_trait_pred() {
            // FIXME: Constness and polarity
            ecx.infcx.probe(|_| {
                let nested_goals = ecx.infcx.sup(
                    goal.param_env,
                    ty::Binder::dummy(goal.predicate.trait_ref),
                    poly_trait_pred.to_poly_trait_ref(),
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
        ecx.infcx.probe(|_| {
            let constituent_tys =
                instantiate_constituent_tys_for_auto_trait(ecx.infcx, goal.predicate.self_ty())?;
            ecx.evaluate_goal_for_constituent_tys_and_make_canonical_response(goal, constituent_tys)
        })
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
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        _goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        unimplemented!();
    }
}

impl<'tcx> EvalCtxt<'_, 'tcx> {
    fn evaluate_goal_for_constituent_tys_and_make_canonical_response(
        &mut self,
        goal: Goal<'tcx, TraitPredicate<'tcx>>,
        constituent_tys: Vec<Ty<'tcx>>,
    ) -> QueryResult<'tcx> {
        self.evaluate_all_and_make_canonical_response(
            constituent_tys
                .into_iter()
                .map(|ty| {
                    goal.with(
                        self.tcx(),
                        ty::Binder::dummy(goal.predicate.with_self_ty(self.tcx(), ty)),
                    )
                })
                .collect(),
        )
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

// Calculates the constituent types of a type for `auto trait` purposes.
//
// For types with an "existential" binder, i.e. generator witnesses, we also
// instantiate the binder with placeholders eagerly.
fn instantiate_constituent_tys_for_auto_trait<'tcx>(
    infcx: &InferCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Result<Vec<Ty<'tcx>>, NoSolution> {
    let tcx = infcx.tcx;
    match *ty.kind() {
        ty::Uint(_)
        | ty::Int(_)
        | ty::Bool
        | ty::Float(_)
        | ty::FnDef(..)
        | ty::FnPtr(_)
        | ty::Str
        | ty::Error(_)
        | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
        | ty::Never
        | ty::Char => Ok(vec![]),

        ty::Placeholder(..)
        | ty::Dynamic(..)
        | ty::Param(..)
        | ty::Foreign(..)
        | ty::Alias(ty::Projection, ..)
        | ty::Bound(..)
        | ty::Infer(ty::TyVar(_)) => {
            // FIXME: Do we need to mark anything as ambiguous here? Yeah?
            Err(NoSolution)
        }

        ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => bug!(),

        ty::RawPtr(ty::TypeAndMut { ty: element_ty, .. }) | ty::Ref(_, element_ty, _) => {
            Ok(vec![element_ty])
        }

        ty::Array(element_ty, _) | ty::Slice(element_ty) => Ok(vec![element_ty]),

        ty::Tuple(ref tys) => {
            // (T1, ..., Tn) -- meets any bound that all of T1...Tn meet
            Ok(tys.iter().collect())
        }

        ty::Closure(_, ref substs) => Ok(vec![substs.as_closure().tupled_upvars_ty()]),

        ty::Generator(_, ref substs, _) => {
            let generator_substs = substs.as_generator();
            Ok(vec![generator_substs.tupled_upvars_ty(), generator_substs.witness()])
        }

        ty::GeneratorWitness(types) => {
            Ok(infcx.replace_bound_vars_with_placeholders(types).to_vec())
        }

        // For `PhantomData<T>`, we pass `T`.
        ty::Adt(def, substs) if def.is_phantom_data() => Ok(vec![substs.type_at(0)]),

        ty::Adt(def, substs) => Ok(def.all_fields().map(|f| f.ty(tcx, substs)).collect()),

        ty::Alias(ty::Opaque, ty::AliasTy { def_id, substs, .. }) => {
            // We can resolve the `impl Trait` to its concrete type,
            // which enforces a DAG between the functions requiring
            // the auto trait bounds in question.
            Ok(vec![tcx.bound_type_of(def_id).subst(tcx, substs)])
        }
    }
}
