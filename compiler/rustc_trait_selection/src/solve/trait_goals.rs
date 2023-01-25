//! Dealing with trait goals, i.e. `T: Trait<'a, U>`.

use std::iter;

use super::assembly::{self, Candidate, CandidateSource};
use super::infcx_ext::InferCtxtExt;
use super::{CanonicalResponse, Certainty, EvalCtxt, Goal, QueryResult};
use rustc_hir::def_id::DefId;
use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::util::supertraits;
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

    fn consider_builtin_unsize_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        let tcx = ecx.tcx();
        let a_ty = goal.predicate.self_ty();
        let b_ty = goal.predicate.trait_ref.substs.type_at(1);
        if b_ty.is_ty_var() {
            return ecx.make_canonical_response(Certainty::AMBIGUOUS);
        }
        ecx.infcx.probe(|_| {
            match (a_ty.kind(), b_ty.kind()) {
                // Trait upcasting, or `dyn Trait + Auto + 'a` -> `dyn Trait + 'b`
                (&ty::Dynamic(_, _, ty::Dyn), &ty::Dynamic(_, _, ty::Dyn)) => {
                    // Dyn upcasting is handled separately, since due to upcasting,
                    // when there are two supertraits that differ by substs, we
                    // may return more than one query response.
                    return Err(NoSolution);
                }
                // `T` -> `dyn Trait` unsizing
                (_, &ty::Dynamic(data, region, ty::Dyn)) => {
                    // Can only unsize to an object-safe type
                    if data
                        .principal_def_id()
                        .map_or(false, |def_id| !tcx.check_is_object_safe(def_id))
                    {
                        return Err(NoSolution);
                    }

                    let Some(sized_def_id) = tcx.lang_items().sized_trait() else {
                        return Err(NoSolution);
                    };
                    let nested_goals: Vec<_> = data
                        .iter()
                        // Check that the type implements all of the predicates of the def-id.
                        // (i.e. the principal, all of the associated types match, and any auto traits)
                        .map(|pred| goal.with(tcx, pred.with_self_ty(tcx, a_ty)))
                        .chain([
                            // The type must be Sized to be unsized.
                            goal.with(
                                tcx,
                                ty::Binder::dummy(tcx.mk_trait_ref(sized_def_id, [a_ty])),
                            ),
                            // The type must outlive the lifetime of the `dyn` we're unsizing into.
                            goal.with(tcx, ty::Binder::dummy(ty::OutlivesPredicate(a_ty, region))),
                        ])
                        .collect();

                    ecx.evaluate_all_and_make_canonical_response(nested_goals)
                }
                // `[T; n]` -> `[T]` unsizing
                (&ty::Array(a_elem_ty, ..), &ty::Slice(b_elem_ty)) => {
                    // We just require that the element type stays the same
                    let nested_goals = ecx.infcx.eq(goal.param_env, a_elem_ty, b_elem_ty)?;
                    ecx.evaluate_all_and_make_canonical_response(nested_goals)
                }
                // Struct unsizing `Struct<T>` -> `Struct<U>` where `T: Unsize<U>`
                (&ty::Adt(a_def, a_substs), &ty::Adt(b_def, b_substs))
                    if a_def.is_struct() && a_def.did() == b_def.did() =>
                {
                    let unsizing_params = tcx.unsizing_params_for_adt(a_def.did());
                    // We must be unsizing some type parameters. This also implies
                    // that the struct has a tail field.
                    if unsizing_params.is_empty() {
                        return Err(NoSolution);
                    }

                    let tail_field = a_def
                        .non_enum_variant()
                        .fields
                        .last()
                        .expect("expected unsized ADT to have a tail field");
                    let tail_field_ty = tcx.bound_type_of(tail_field.did);

                    let a_tail_ty = tail_field_ty.subst(tcx, a_substs);
                    let b_tail_ty = tail_field_ty.subst(tcx, b_substs);

                    // Substitute just the unsizing params from B into A. The type after
                    // this substitution must be equal to B. This is so we don't unsize
                    // unrelated type parameters.
                    let new_a_substs = tcx.mk_substs(a_substs.iter().enumerate().map(|(i, a)| {
                        if unsizing_params.contains(i as u32) { b_substs[i] } else { a }
                    }));
                    let unsized_a_ty = tcx.mk_adt(a_def, new_a_substs);

                    // Finally, we require that `TailA: Unsize<TailB>` for the tail field
                    // types.
                    let mut nested_goals = ecx.infcx.eq(goal.param_env, unsized_a_ty, b_ty)?;
                    nested_goals.push(goal.with(
                        tcx,
                        ty::Binder::dummy(
                            tcx.mk_trait_ref(goal.predicate.def_id(), [a_tail_ty, b_tail_ty]),
                        ),
                    ));

                    ecx.evaluate_all_and_make_canonical_response(nested_goals)
                }
                // Tuple unsizing `(.., T)` -> `(.., U)` where `T: Unsize<U>`
                (&ty::Tuple(a_tys), &ty::Tuple(b_tys))
                    if a_tys.len() == b_tys.len() && !a_tys.is_empty() =>
                {
                    let (a_last_ty, a_rest_tys) = a_tys.split_last().unwrap();
                    let b_last_ty = b_tys.last().unwrap();

                    // Substitute just the tail field of B., and require that they're equal.
                    let unsized_a_ty = tcx.mk_tup(a_rest_tys.iter().chain([b_last_ty]));
                    let mut nested_goals = ecx.infcx.eq(goal.param_env, unsized_a_ty, b_ty)?;

                    // Similar to ADTs, require that the rest of the fields are equal.
                    nested_goals.push(goal.with(
                        tcx,
                        ty::Binder::dummy(
                            tcx.mk_trait_ref(goal.predicate.def_id(), [*a_last_ty, *b_last_ty]),
                        ),
                    ));

                    ecx.evaluate_all_and_make_canonical_response(nested_goals)
                }
                _ => Err(NoSolution),
            }
        })
    }

    fn consider_builtin_dyn_upcast_candidates(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> Vec<CanonicalResponse<'tcx>> {
        let tcx = ecx.tcx();

        let a_ty = goal.predicate.self_ty();
        let b_ty = goal.predicate.trait_ref.substs.type_at(1);
        let ty::Dynamic(a_data, a_region, ty::Dyn) = *a_ty.kind() else {
            return vec![];
        };
        let ty::Dynamic(b_data, b_region, ty::Dyn) = *b_ty.kind() else {
            return vec![];
        };

        // All of a's auto traits need to be in b's auto traits.
        let auto_traits_compatible =
            b_data.auto_traits().all(|b| a_data.auto_traits().any(|a| a == b));
        if !auto_traits_compatible {
            return vec![];
        }

        let mut unsize_dyn_to_principal = |principal: Option<ty::PolyExistentialTraitRef<'tcx>>| {
            ecx.infcx.probe(|_| -> Result<_, NoSolution> {
                // Require that all of the trait predicates from A match B, except for
                // the auto traits. We do this by constructing a new A type with B's
                // auto traits, and equating these types.
                let new_a_data = principal
                    .into_iter()
                    .map(|trait_ref| trait_ref.map_bound(ty::ExistentialPredicate::Trait))
                    .chain(a_data.iter().filter(|a| {
                        matches!(a.skip_binder(), ty::ExistentialPredicate::Projection(_))
                    }))
                    .chain(
                        b_data
                            .auto_traits()
                            .map(ty::ExistentialPredicate::AutoTrait)
                            .map(ty::Binder::dummy),
                    );
                let new_a_data = tcx.mk_poly_existential_predicates(new_a_data);
                let new_a_ty = tcx.mk_dynamic(new_a_data, b_region, ty::Dyn);

                // We also require that A's lifetime outlives B's lifetime.
                let mut nested_obligations = ecx.infcx.eq(goal.param_env, new_a_ty, b_ty)?;
                nested_obligations.push(
                    goal.with(tcx, ty::Binder::dummy(ty::OutlivesPredicate(a_region, b_region))),
                );

                ecx.evaluate_all_and_make_canonical_response(nested_obligations)
            })
        };

        let mut responses = vec![];
        // If the principal def ids match (or are both none), then we're not doing
        // trait upcasting. We're just removing auto traits (or shortening the lifetime).
        if a_data.principal_def_id() == b_data.principal_def_id() {
            if let Ok(response) = unsize_dyn_to_principal(a_data.principal()) {
                responses.push(response);
            }
        } else if let Some(a_principal) = a_data.principal()
            && let Some(b_principal) = b_data.principal()
        {
            for super_trait_ref in supertraits(tcx, a_principal.with_self_ty(tcx, a_ty)) {
                if super_trait_ref.def_id() != b_principal.def_id() {
                    continue;
                }
                let erased_trait_ref = super_trait_ref
                    .map_bound(|trait_ref| ty::ExistentialTraitRef::erase_self_ty(tcx, trait_ref));
                if let Ok(response) = unsize_dyn_to_principal(Some(erased_trait_ref)) {
                    responses.push(response);
                }
            }
        }

        responses
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
