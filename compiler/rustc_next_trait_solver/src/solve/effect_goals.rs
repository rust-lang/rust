//! Dealing with host effect goals, i.e. enforcing the constness in
//! `T: const Trait` or `T: ~const Trait`.

use rustc_type_ir::fast_reject::DeepRejectCtxt;
use rustc_type_ir::inherent::*;
use rustc_type_ir::lang_items::TraitSolverLangItem;
use rustc_type_ir::solve::inspect::ProbeKind;
use rustc_type_ir::{self as ty, Interner, elaborate};
use tracing::instrument;

use super::assembly::{Candidate, structural_traits};
use crate::delegate::SolverDelegate;
use crate::solve::{
    BuiltinImplSource, CandidateSource, Certainty, EvalCtxt, Goal, GoalSource, NoSolution,
    QueryResult, assembly,
};

impl<D, I> assembly::GoalKind<D> for ty::HostEffectPredicate<I>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    fn self_ty(self) -> I::Ty {
        self.self_ty()
    }

    fn trait_ref(self, _: I) -> ty::TraitRef<I> {
        self.trait_ref
    }

    fn with_self_ty(self, cx: I, self_ty: I::Ty) -> Self {
        self.with_self_ty(cx, self_ty)
    }

    fn trait_def_id(self, _: I) -> I::DefId {
        self.def_id()
    }

    fn fast_reject_assumption(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
        assumption: I::Clause,
    ) -> Result<(), NoSolution> {
        if let Some(host_clause) = assumption.as_host_effect_clause() {
            if host_clause.def_id() == goal.predicate.def_id()
                && host_clause.constness().satisfies(goal.predicate.constness)
            {
                if DeepRejectCtxt::relate_rigid_rigid(ecx.cx()).args_may_unify(
                    goal.predicate.trait_ref.args,
                    host_clause.skip_binder().trait_ref.args,
                ) {
                    return Ok(());
                }
            }
        }

        Err(NoSolution)
    }

    fn match_assumption(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
        assumption: I::Clause,
        then: impl FnOnce(&mut EvalCtxt<'_, D>) -> QueryResult<I>,
    ) -> QueryResult<I> {
        let host_clause = assumption.as_host_effect_clause().unwrap();

        let assumption_trait_pred = ecx.instantiate_binder_with_infer(host_clause);
        ecx.eq(goal.param_env, goal.predicate.trait_ref, assumption_trait_pred.trait_ref)?;

        then(ecx)
    }

    /// Register additional assumptions for aliases corresponding to `~const` item bounds.
    ///
    /// Unlike item bounds, they are not simply implied by the well-formedness of the alias.
    /// Instead, they only hold if the const conditons on the alias also hold. This is why
    /// we also register the const conditions of the alias after matching the goal against
    /// the assumption.
    fn consider_additional_alias_assumptions(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
        alias_ty: ty::AliasTy<I>,
    ) -> Vec<Candidate<I>> {
        let cx = ecx.cx();
        let mut candidates = vec![];

        if !ecx.cx().alias_has_const_conditions(alias_ty.def_id) {
            return vec![];
        }

        for clause in elaborate::elaborate(
            cx,
            cx.explicit_implied_const_bounds(alias_ty.def_id)
                .iter_instantiated(cx, alias_ty.args)
                .map(|trait_ref| trait_ref.to_host_effect_clause(cx, goal.predicate.constness)),
        ) {
            candidates.extend(Self::probe_and_match_goal_against_assumption(
                ecx,
                CandidateSource::AliasBound,
                goal,
                clause,
                |ecx| {
                    // Const conditions must hold for the implied const bound to hold.
                    ecx.add_goals(
                        GoalSource::AliasBoundConstCondition,
                        cx.const_conditions(alias_ty.def_id)
                            .iter_instantiated(cx, alias_ty.args)
                            .map(|trait_ref| {
                                goal.with(
                                    cx,
                                    trait_ref.to_host_effect_clause(cx, goal.predicate.constness),
                                )
                            }),
                    );
                    ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                },
            ));
        }

        candidates
    }

    fn consider_impl_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
        impl_def_id: I::DefId,
    ) -> Result<Candidate<I>, NoSolution> {
        let cx = ecx.cx();

        let impl_trait_ref = cx.impl_trait_ref(impl_def_id);
        if !DeepRejectCtxt::relate_rigid_infer(ecx.cx())
            .args_may_unify(goal.predicate.trait_ref.args, impl_trait_ref.skip_binder().args)
        {
            return Err(NoSolution);
        }

        let impl_polarity = cx.impl_polarity(impl_def_id);
        match impl_polarity {
            ty::ImplPolarity::Negative => return Err(NoSolution),
            ty::ImplPolarity::Reservation => {
                unimplemented!("reservation impl for const trait: {:?}", goal)
            }
            ty::ImplPolarity::Positive => {}
        };

        if !cx.impl_is_const(impl_def_id) {
            return Err(NoSolution);
        }

        ecx.probe_trait_candidate(CandidateSource::Impl(impl_def_id)).enter(|ecx| {
            let impl_args = ecx.fresh_args_for_item(impl_def_id);
            ecx.record_impl_args(impl_args);
            let impl_trait_ref = impl_trait_ref.instantiate(cx, impl_args);

            ecx.eq(goal.param_env, goal.predicate.trait_ref, impl_trait_ref)?;
            let where_clause_bounds = cx
                .predicates_of(impl_def_id)
                .iter_instantiated(cx, impl_args)
                .map(|pred| goal.with(cx, pred));
            ecx.add_goals(GoalSource::ImplWhereBound, where_clause_bounds);

            // For this impl to be `const`, we need to check its `~const` bounds too.
            let const_conditions = cx
                .const_conditions(impl_def_id)
                .iter_instantiated(cx, impl_args)
                .map(|bound_trait_ref| {
                    goal.with(
                        cx,
                        bound_trait_ref.to_host_effect_clause(cx, goal.predicate.constness),
                    )
                });
            ecx.add_goals(GoalSource::ImplWhereBound, const_conditions);

            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_error_guaranteed_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        _guar: I::ErrorGuaranteed,
    ) -> Result<Candidate<I>, NoSolution> {
        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc)
            .enter(|ecx| ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes))
    }

    fn consider_auto_trait_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        unreachable!("auto traits are never const")
    }

    fn consider_trait_alias_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        unreachable!("trait aliases are never const")
    }

    fn consider_builtin_sized_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        unreachable!("Sized is never const")
    }

    fn consider_builtin_copy_clone_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        Err(NoSolution)
    }

    fn consider_builtin_fn_ptr_trait_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        todo!("Fn* are not yet const")
    }

    fn consider_builtin_fn_trait_candidates(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
        _kind: rustc_type_ir::ClosureKind,
    ) -> Result<Candidate<I>, NoSolution> {
        let cx = ecx.cx();

        let self_ty = goal.predicate.self_ty();
        let (inputs_and_output, def_id, args) =
            structural_traits::extract_fn_def_from_const_callable(cx, self_ty)?;

        // A built-in `Fn` impl only holds if the output is sized.
        // (FIXME: technically we only need to check this if the type is a fn ptr...)
        let output_is_sized_pred = inputs_and_output.map_bound(|(_, output)| {
            ty::TraitRef::new(cx, cx.require_lang_item(TraitSolverLangItem::Sized), [output])
        });
        let requirements = cx
            .const_conditions(def_id)
            .iter_instantiated(cx, args)
            .map(|trait_ref| {
                (
                    GoalSource::ImplWhereBound,
                    goal.with(cx, trait_ref.to_host_effect_clause(cx, goal.predicate.constness)),
                )
            })
            .chain([(GoalSource::ImplWhereBound, goal.with(cx, output_is_sized_pred))]);

        let pred = inputs_and_output
            .map_bound(|(inputs, _)| {
                ty::TraitRef::new(
                    cx,
                    goal.predicate.def_id(),
                    [goal.predicate.self_ty(), Ty::new_tup(cx, inputs.as_slice())],
                )
            })
            .to_host_effect_clause(cx, goal.predicate.constness);

        Self::probe_and_consider_implied_clause(
            ecx,
            CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
            goal,
            pred,
            requirements,
        )
    }

    fn consider_builtin_async_fn_trait_candidates(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
        _kind: rustc_type_ir::ClosureKind,
    ) -> Result<Candidate<I>, NoSolution> {
        todo!("AsyncFn* are not yet const")
    }

    fn consider_builtin_async_fn_kind_helper_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        unreachable!("AsyncFnKindHelper is not const")
    }

    fn consider_builtin_tuple_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        unreachable!("Tuple trait is not const")
    }

    fn consider_builtin_pointee_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        unreachable!("Pointee is not const")
    }

    fn consider_builtin_future_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        unreachable!("Future is not const")
    }

    fn consider_builtin_iterator_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        todo!("Iterator is not yet const")
    }

    fn consider_builtin_fused_iterator_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        unreachable!("FusedIterator is not const")
    }

    fn consider_builtin_async_iterator_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        unreachable!("AsyncIterator is not const")
    }

    fn consider_builtin_coroutine_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        unreachable!("Coroutine is not const")
    }

    fn consider_builtin_discriminant_kind_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        unreachable!("DiscriminantKind is not const")
    }

    fn consider_builtin_destruct_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        let cx = ecx.cx();

        let self_ty = goal.predicate.self_ty();
        let const_conditions = structural_traits::const_conditions_for_destruct(cx, self_ty)?;

        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
            ecx.add_goals(
                GoalSource::AliasBoundConstCondition,
                const_conditions.into_iter().map(|trait_ref| {
                    goal.with(
                        cx,
                        ty::Binder::dummy(trait_ref)
                            .to_host_effect_clause(cx, goal.predicate.constness),
                    )
                }),
            );
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_builtin_transmute_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        unreachable!("TransmuteFrom is not const")
    }

    fn consider_builtin_bikeshed_guaranteed_no_drop_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        unreachable!("BikeshedGuaranteedNoDrop is not const");
    }

    fn consider_structural_builtin_unsize_candidates(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Vec<Candidate<I>> {
        unreachable!("Unsize is not const")
    }
}

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    #[instrument(level = "trace", skip(self))]
    pub(super) fn compute_host_effect_goal(
        &mut self,
        goal: Goal<I, ty::HostEffectPredicate<I>>,
    ) -> QueryResult<I> {
        let (_, proven_via) = self.probe(|_| ProbeKind::ShadowedEnvProbing).enter(|ecx| {
            let trait_goal: Goal<I, ty::TraitPredicate<I>> =
                goal.with(ecx.cx(), goal.predicate.trait_ref);
            ecx.compute_trait_goal(trait_goal)
        })?;
        self.assemble_and_merge_candidates(proven_via, goal, |_ecx| Err(NoSolution))
    }
}
