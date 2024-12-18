//! Dealing with trait goals, i.e. `T: Trait<'a, U>`.

use rustc_ast_ir::Movability;
use rustc_type_ir::data_structures::IndexSet;
use rustc_type_ir::fast_reject::DeepRejectCtxt;
use rustc_type_ir::inherent::*;
use rustc_type_ir::lang_items::TraitSolverLangItem;
use rustc_type_ir::solve::CanonicalResponse;
use rustc_type_ir::visit::TypeVisitableExt as _;
use rustc_type_ir::{self as ty, Interner, TraitPredicate, TypingMode, Upcast as _, elaborate};
use tracing::{instrument, trace};

use crate::delegate::SolverDelegate;
use crate::solve::assembly::structural_traits::{self, AsyncCallableRelevantTypes};
use crate::solve::assembly::{self, Candidate};
use crate::solve::inspect::ProbeKind;
use crate::solve::{
    BuiltinImplSource, CandidateSource, Certainty, EvalCtxt, Goal, GoalSource, MaybeCause,
    NoSolution, QueryResult,
};

impl<D, I> assembly::GoalKind<D> for TraitPredicate<I>
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

    fn consider_additional_alias_assumptions(
        _ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
        _alias_ty: ty::AliasTy<I>,
    ) -> Vec<Candidate<I>> {
        vec![]
    }

    fn consider_impl_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, TraitPredicate<I>>,
        impl_def_id: I::DefId,
    ) -> Result<Candidate<I>, NoSolution> {
        let cx = ecx.cx();

        let impl_trait_ref = cx.impl_trait_ref(impl_def_id);
        if !DeepRejectCtxt::relate_rigid_infer(ecx.cx())
            .args_may_unify(goal.predicate.trait_ref.args, impl_trait_ref.skip_binder().args)
        {
            return Err(NoSolution);
        }

        // An upper bound of the certainty of this goal, used to lower the certainty
        // of reservation impl to ambiguous during coherence.
        let impl_polarity = cx.impl_polarity(impl_def_id);
        let maximal_certainty = match (impl_polarity, goal.predicate.polarity) {
            // In intercrate mode, this is ambiguous. But outside of intercrate,
            // it's not a real impl.
            (ty::ImplPolarity::Reservation, _) => match ecx.typing_mode() {
                TypingMode::Coherence => Certainty::AMBIGUOUS,
                TypingMode::Analysis { .. }
                | TypingMode::PostBorrowckAnalysis { .. }
                | TypingMode::PostAnalysis => return Err(NoSolution),
            },

            // Impl matches polarity
            (ty::ImplPolarity::Positive, ty::PredicatePolarity::Positive)
            | (ty::ImplPolarity::Negative, ty::PredicatePolarity::Negative) => Certainty::Yes,

            // Impl doesn't match polarity
            (ty::ImplPolarity::Positive, ty::PredicatePolarity::Negative)
            | (ty::ImplPolarity::Negative, ty::PredicatePolarity::Positive) => {
                return Err(NoSolution);
            }
        };

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

            // We currently elaborate all supertrait outlives obligations from impls.
            // This can be removed when we actually do coinduction correctly, and prove
            // all supertrait obligations unconditionally.
            let goal_clause: I::Clause = goal.predicate.upcast(cx);
            for clause in elaborate::elaborate(cx, [goal_clause]) {
                if matches!(
                    clause.kind().skip_binder(),
                    ty::ClauseKind::TypeOutlives(..) | ty::ClauseKind::RegionOutlives(..)
                ) {
                    ecx.add_goal(GoalSource::Misc, goal.with(cx, clause));
                }
            }

            ecx.evaluate_added_goals_and_make_canonical_response(maximal_certainty)
        })
    }

    fn consider_error_guaranteed_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        _guar: I::ErrorGuaranteed,
    ) -> Result<Candidate<I>, NoSolution> {
        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc)
            .enter(|ecx| ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes))
    }

    fn probe_and_match_goal_against_assumption(
        ecx: &mut EvalCtxt<'_, D>,
        source: CandidateSource<I>,
        goal: Goal<I, Self>,
        assumption: I::Clause,
        then: impl FnOnce(&mut EvalCtxt<'_, D>) -> QueryResult<I>,
    ) -> Result<Candidate<I>, NoSolution> {
        if let Some(trait_clause) = assumption.as_trait_clause() {
            if trait_clause.def_id() == goal.predicate.def_id()
                && trait_clause.polarity() == goal.predicate.polarity
            {
                if !DeepRejectCtxt::relate_rigid_rigid(ecx.cx()).args_may_unify(
                    goal.predicate.trait_ref.args,
                    trait_clause.skip_binder().trait_ref.args,
                ) {
                    return Err(NoSolution);
                }

                ecx.probe_trait_candidate(source).enter(|ecx| {
                    let assumption_trait_pred = ecx.instantiate_binder_with_infer(trait_clause);
                    ecx.eq(
                        goal.param_env,
                        goal.predicate.trait_ref,
                        assumption_trait_pred.trait_ref,
                    )?;
                    then(ecx)
                })
            } else {
                Err(NoSolution)
            }
        } else {
            Err(NoSolution)
        }
    }

    fn consider_auto_trait_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        if let Some(result) = ecx.disqualify_auto_trait_candidate_due_to_possible_impl(goal) {
            return result;
        }

        // Only consider auto impls of unsafe traits when there are no unsafe
        // fields.
        if ecx.cx().trait_is_unsafe(goal.predicate.def_id())
            && goal.predicate.self_ty().has_unsafe_fields()
        {
            return Err(NoSolution);
        }

        // We only look into opaque types during analysis for opaque types
        // outside of their defining scope. Doing so for opaques in the
        // defining scope may require calling `typeck` on the same item we're
        // currently type checking, which will result in a fatal cycle that
        // ideally we want to avoid, since we can make progress on this goal
        // via an alias bound or a locally-inferred hidden type instead.
        if let ty::Alias(ty::Opaque, opaque_ty) = goal.predicate.self_ty().kind() {
            debug_assert!(ecx.opaque_type_is_rigid(opaque_ty.def_id));
        }

        ecx.probe_and_evaluate_goal_for_constituent_tys(
            CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
            goal,
            structural_traits::instantiate_constituent_tys_for_auto_trait,
        )
    }

    fn consider_trait_alias_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        let cx = ecx.cx();

        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
            let nested_obligations = cx
                .predicates_of(goal.predicate.def_id())
                .iter_instantiated(cx, goal.predicate.trait_ref.args)
                .map(|p| goal.with(cx, p));
            // FIXME(-Znext-solver=coinductive): Should this be `GoalSource::ImplWhereBound`?
            ecx.add_goals(GoalSource::Misc, nested_obligations);
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_builtin_sized_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        ecx.probe_and_evaluate_goal_for_constituent_tys(
            CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
            goal,
            structural_traits::instantiate_constituent_tys_for_sized_trait,
        )
    }

    fn consider_builtin_copy_clone_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        ecx.probe_and_evaluate_goal_for_constituent_tys(
            CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
            goal,
            structural_traits::instantiate_constituent_tys_for_copy_clone_trait,
        )
    }

    fn consider_builtin_fn_ptr_trait_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        let self_ty = goal.predicate.self_ty();
        match goal.predicate.polarity {
            // impl FnPtr for FnPtr {}
            ty::PredicatePolarity::Positive => {
                if self_ty.is_fn_ptr() {
                    ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
                        ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                    })
                } else {
                    Err(NoSolution)
                }
            }
            //  impl !FnPtr for T where T != FnPtr && T is rigid {}
            ty::PredicatePolarity::Negative => {
                // If a type is rigid and not a fn ptr, then we know for certain
                // that it does *not* implement `FnPtr`.
                if !self_ty.is_fn_ptr() && self_ty.is_known_rigid() {
                    ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
                        ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                    })
                } else {
                    Err(NoSolution)
                }
            }
        }
    }

    fn consider_builtin_fn_trait_candidates(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
        goal_kind: ty::ClosureKind,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        let cx = ecx.cx();
        let tupled_inputs_and_output =
            match structural_traits::extract_tupled_inputs_and_output_from_callable(
                cx,
                goal.predicate.self_ty(),
                goal_kind,
            )? {
                Some(a) => a,
                None => {
                    return ecx.forced_ambiguity(MaybeCause::Ambiguity);
                }
            };

        // A built-in `Fn` impl only holds if the output is sized.
        // (FIXME: technically we only need to check this if the type is a fn ptr...)
        let output_is_sized_pred = tupled_inputs_and_output.map_bound(|(_, output)| {
            ty::TraitRef::new(cx, cx.require_lang_item(TraitSolverLangItem::Sized), [output])
        });

        let pred = tupled_inputs_and_output
            .map_bound(|(inputs, _)| {
                ty::TraitRef::new(cx, goal.predicate.def_id(), [goal.predicate.self_ty(), inputs])
            })
            .upcast(cx);
        Self::probe_and_consider_implied_clause(
            ecx,
            CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
            goal,
            pred,
            [(GoalSource::ImplWhereBound, goal.with(cx, output_is_sized_pred))],
        )
    }

    fn consider_builtin_async_fn_trait_candidates(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
        goal_kind: ty::ClosureKind,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        let cx = ecx.cx();
        let (tupled_inputs_and_output_and_coroutine, nested_preds) =
            structural_traits::extract_tupled_inputs_and_output_from_async_callable(
                cx,
                goal.predicate.self_ty(),
                goal_kind,
                // This region doesn't matter because we're throwing away the coroutine type
                Region::new_static(cx),
            )?;

        // A built-in `AsyncFn` impl only holds if the output is sized.
        // (FIXME: technically we only need to check this if the type is a fn ptr...)
        let output_is_sized_pred = tupled_inputs_and_output_and_coroutine.map_bound(
            |AsyncCallableRelevantTypes { output_coroutine_ty, .. }| {
                ty::TraitRef::new(cx, cx.require_lang_item(TraitSolverLangItem::Sized), [
                    output_coroutine_ty,
                ])
            },
        );

        let pred = tupled_inputs_and_output_and_coroutine
            .map_bound(|AsyncCallableRelevantTypes { tupled_inputs_ty, .. }| {
                ty::TraitRef::new(cx, goal.predicate.def_id(), [
                    goal.predicate.self_ty(),
                    tupled_inputs_ty,
                ])
            })
            .upcast(cx);
        Self::probe_and_consider_implied_clause(
            ecx,
            CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
            goal,
            pred,
            [goal.with(cx, output_is_sized_pred)]
                .into_iter()
                .chain(nested_preds.into_iter().map(|pred| goal.with(cx, pred)))
                .map(|goal| (GoalSource::ImplWhereBound, goal)),
        )
    }

    fn consider_builtin_async_fn_kind_helper_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        let [closure_fn_kind_ty, goal_kind_ty] = *goal.predicate.trait_ref.args.as_slice() else {
            panic!();
        };

        let Some(closure_kind) = closure_fn_kind_ty.expect_ty().to_opt_closure_kind() else {
            // We don't need to worry about the self type being an infer var.
            return Err(NoSolution);
        };
        let goal_kind = goal_kind_ty.expect_ty().to_opt_closure_kind().unwrap();
        if closure_kind.extends(goal_kind) {
            ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc)
                .enter(|ecx| ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes))
        } else {
            Err(NoSolution)
        }
    }

    /// ```rust, ignore (not valid rust syntax)
    /// impl Tuple for () {}
    /// impl Tuple for (T1,) {}
    /// impl Tuple for (T1, T2) {}
    /// impl Tuple for (T1, .., Tn) {}
    /// ```
    fn consider_builtin_tuple_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        if let ty::Tuple(..) = goal.predicate.self_ty().kind() {
            ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc)
                .enter(|ecx| ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes))
        } else {
            Err(NoSolution)
        }
    }

    fn consider_builtin_pointee_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc)
            .enter(|ecx| ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes))
    }

    fn consider_builtin_future_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        let ty::Coroutine(def_id, _) = goal.predicate.self_ty().kind() else {
            return Err(NoSolution);
        };

        // Coroutines are not futures unless they come from `async` desugaring
        let cx = ecx.cx();
        if !cx.coroutine_is_async(def_id) {
            return Err(NoSolution);
        }

        // Async coroutine unconditionally implement `Future`
        // Technically, we need to check that the future output type is Sized,
        // but that's already proven by the coroutine being WF.
        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc)
            .enter(|ecx| ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes))
    }

    fn consider_builtin_iterator_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        let ty::Coroutine(def_id, _) = goal.predicate.self_ty().kind() else {
            return Err(NoSolution);
        };

        // Coroutines are not iterators unless they come from `gen` desugaring
        let cx = ecx.cx();
        if !cx.coroutine_is_gen(def_id) {
            return Err(NoSolution);
        }

        // Gen coroutines unconditionally implement `Iterator`
        // Technically, we need to check that the iterator output type is Sized,
        // but that's already proven by the coroutines being WF.
        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc)
            .enter(|ecx| ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes))
    }

    fn consider_builtin_fused_iterator_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        let ty::Coroutine(def_id, _) = goal.predicate.self_ty().kind() else {
            return Err(NoSolution);
        };

        // Coroutines are not iterators unless they come from `gen` desugaring
        let cx = ecx.cx();
        if !cx.coroutine_is_gen(def_id) {
            return Err(NoSolution);
        }

        // Gen coroutines unconditionally implement `FusedIterator`.
        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc)
            .enter(|ecx| ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes))
    }

    fn consider_builtin_async_iterator_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        let ty::Coroutine(def_id, _) = goal.predicate.self_ty().kind() else {
            return Err(NoSolution);
        };

        // Coroutines are not iterators unless they come from `gen` desugaring
        let cx = ecx.cx();
        if !cx.coroutine_is_async_gen(def_id) {
            return Err(NoSolution);
        }

        // Gen coroutines unconditionally implement `Iterator`
        // Technically, we need to check that the iterator output type is Sized,
        // but that's already proven by the coroutines being WF.
        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc)
            .enter(|ecx| ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes))
    }

    fn consider_builtin_coroutine_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        let self_ty = goal.predicate.self_ty();
        let ty::Coroutine(def_id, args) = self_ty.kind() else {
            return Err(NoSolution);
        };

        // `async`-desugared coroutines do not implement the coroutine trait
        let cx = ecx.cx();
        if !cx.is_general_coroutine(def_id) {
            return Err(NoSolution);
        }

        let coroutine = args.as_coroutine();
        Self::probe_and_consider_implied_clause(
            ecx,
            CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
            goal,
            ty::TraitRef::new(cx, goal.predicate.def_id(), [self_ty, coroutine.resume_ty()])
                .upcast(cx),
            // Technically, we need to check that the coroutine types are Sized,
            // but that's already proven by the coroutine being WF.
            [],
        )
    }

    fn consider_builtin_discriminant_kind_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        // `DiscriminantKind` is automatically implemented for every type.
        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc)
            .enter(|ecx| ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes))
    }

    fn consider_builtin_async_destruct_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        // `AsyncDestruct` is automatically implemented for every type.
        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc)
            .enter(|ecx| ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes))
    }

    fn consider_builtin_destruct_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        // `Destruct` is automatically implemented for every type in
        // non-const environments.
        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc)
            .enter(|ecx| ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes))
    }

    fn consider_builtin_transmute_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return Err(NoSolution);
        }

        // `rustc_transmute` does not have support for type or const params
        if goal.has_non_region_placeholders() {
            return Err(NoSolution);
        }

        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
            let assume = ecx.structurally_normalize_const(
                goal.param_env,
                goal.predicate.trait_ref.args.const_at(2),
            )?;

            let certainty = ecx.is_transmutable(
                goal.param_env,
                goal.predicate.trait_ref.args.type_at(0),
                goal.predicate.trait_ref.args.type_at(1),
                assume,
            )?;
            ecx.evaluate_added_goals_and_make_canonical_response(certainty)
        })
    }

    /// ```ignore (builtin impl example)
    /// trait Trait {
    ///     fn foo(&self);
    /// }
    /// // results in the following builtin impl
    /// impl<'a, T: Trait + 'a> Unsize<dyn Trait + 'a> for T {}
    /// ```
    fn consider_structural_builtin_unsize_candidates(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Vec<Candidate<I>> {
        if goal.predicate.polarity != ty::PredicatePolarity::Positive {
            return vec![];
        }

        let result_to_single = |result| match result {
            Ok(resp) => vec![resp],
            Err(NoSolution) => vec![],
        };

        ecx.probe(|_| ProbeKind::UnsizeAssembly).enter(|ecx| {
            let a_ty = goal.predicate.self_ty();
            // We need to normalize the b_ty since it's matched structurally
            // in the other functions below.
            let Ok(b_ty) = ecx.structurally_normalize_ty(
                goal.param_env,
                goal.predicate.trait_ref.args.type_at(1),
            ) else {
                return vec![];
            };

            let goal = goal.with(ecx.cx(), (a_ty, b_ty));
            match (a_ty.kind(), b_ty.kind()) {
                (ty::Infer(ty::TyVar(..)), ..) => panic!("unexpected infer {a_ty:?} {b_ty:?}"),

                (_, ty::Infer(ty::TyVar(..))) => {
                    result_to_single(ecx.forced_ambiguity(MaybeCause::Ambiguity))
                }

                // Trait upcasting, or `dyn Trait + Auto + 'a` -> `dyn Trait + 'b`.
                (
                    ty::Dynamic(a_data, a_region, ty::Dyn),
                    ty::Dynamic(b_data, b_region, ty::Dyn),
                ) => ecx.consider_builtin_dyn_upcast_candidates(
                    goal, a_data, a_region, b_data, b_region,
                ),

                // `T` -> `dyn Trait` unsizing.
                (_, ty::Dynamic(b_region, b_data, ty::Dyn)) => result_to_single(
                    ecx.consider_builtin_unsize_to_dyn_candidate(goal, b_region, b_data),
                ),

                // `[T; N]` -> `[T]` unsizing
                (ty::Array(a_elem_ty, ..), ty::Slice(b_elem_ty)) => {
                    result_to_single(ecx.consider_builtin_array_unsize(goal, a_elem_ty, b_elem_ty))
                }

                // `Struct<T>` -> `Struct<U>` where `T: Unsize<U>`
                (ty::Adt(a_def, a_args), ty::Adt(b_def, b_args))
                    if a_def.is_struct() && a_def == b_def =>
                {
                    result_to_single(
                        ecx.consider_builtin_struct_unsize(goal, a_def, a_args, b_args),
                    )
                }

                //  `(A, B, T)` -> `(A, B, U)` where `T: Unsize<U>`
                (ty::Tuple(a_tys), ty::Tuple(b_tys))
                    if a_tys.len() == b_tys.len() && !a_tys.is_empty() =>
                {
                    result_to_single(ecx.consider_builtin_tuple_unsize(goal, a_tys, b_tys))
                }

                _ => vec![],
            }
        })
    }
}

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    /// Trait upcasting allows for coercions between trait objects:
    /// ```ignore (builtin impl example)
    /// trait Super {}
    /// trait Trait: Super {}
    /// // results in builtin impls upcasting to a super trait
    /// impl<'a, 'b: 'a> Unsize<dyn Super + 'a> for dyn Trait + 'b {}
    /// // and impls removing auto trait bounds.
    /// impl<'a, 'b: 'a> Unsize<dyn Trait + 'a> for dyn Trait + Send + 'b {}
    /// ```
    fn consider_builtin_dyn_upcast_candidates(
        &mut self,
        goal: Goal<I, (I::Ty, I::Ty)>,
        a_data: I::BoundExistentialPredicates,
        a_region: I::Region,
        b_data: I::BoundExistentialPredicates,
        b_region: I::Region,
    ) -> Vec<Candidate<I>> {
        let cx = self.cx();
        let Goal { predicate: (a_ty, _b_ty), .. } = goal;

        let mut responses = vec![];
        // If the principal def ids match (or are both none), then we're not doing
        // trait upcasting. We're just removing auto traits (or shortening the lifetime).
        let b_principal_def_id = b_data.principal_def_id();
        if a_data.principal_def_id() == b_principal_def_id || b_principal_def_id.is_none() {
            responses.extend(self.consider_builtin_upcast_to_principal(
                goal,
                CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
                a_data,
                a_region,
                b_data,
                b_region,
                a_data.principal(),
            ));
        } else if let Some(a_principal) = a_data.principal() {
            for new_a_principal in
                elaborate::supertraits(self.cx(), a_principal.with_self_ty(cx, a_ty)).skip(1)
            {
                responses.extend(self.consider_builtin_upcast_to_principal(
                    goal,
                    CandidateSource::BuiltinImpl(BuiltinImplSource::TraitUpcasting),
                    a_data,
                    a_region,
                    b_data,
                    b_region,
                    Some(new_a_principal.map_bound(|trait_ref| {
                        ty::ExistentialTraitRef::erase_self_ty(cx, trait_ref)
                    })),
                ));
            }
        }

        responses
    }

    fn consider_builtin_unsize_to_dyn_candidate(
        &mut self,
        goal: Goal<I, (I::Ty, I::Ty)>,
        b_data: I::BoundExistentialPredicates,
        b_region: I::Region,
    ) -> Result<Candidate<I>, NoSolution> {
        let cx = self.cx();
        let Goal { predicate: (a_ty, _), .. } = goal;

        // Can only unsize to an dyn-compatible trait.
        if b_data.principal_def_id().is_some_and(|def_id| !cx.trait_is_dyn_compatible(def_id)) {
            return Err(NoSolution);
        }

        self.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
            // Check that the type implements all of the predicates of the trait object.
            // (i.e. the principal, all of the associated types match, and any auto traits)
            ecx.add_goals(
                GoalSource::ImplWhereBound,
                b_data.iter().map(|pred| goal.with(cx, pred.with_self_ty(cx, a_ty))),
            );

            // The type must be `Sized` to be unsized.
            ecx.add_goal(
                GoalSource::ImplWhereBound,
                goal.with(
                    cx,
                    ty::TraitRef::new(cx, cx.require_lang_item(TraitSolverLangItem::Sized), [a_ty]),
                ),
            );

            // The type must outlive the lifetime of the `dyn` we're unsizing into.
            ecx.add_goal(GoalSource::Misc, goal.with(cx, ty::OutlivesPredicate(a_ty, b_region)));
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_builtin_upcast_to_principal(
        &mut self,
        goal: Goal<I, (I::Ty, I::Ty)>,
        source: CandidateSource<I>,
        a_data: I::BoundExistentialPredicates,
        a_region: I::Region,
        b_data: I::BoundExistentialPredicates,
        b_region: I::Region,
        upcast_principal: Option<ty::Binder<I, ty::ExistentialTraitRef<I>>>,
    ) -> Result<Candidate<I>, NoSolution> {
        let param_env = goal.param_env;

        // We may upcast to auto traits that are either explicitly listed in
        // the object type's bounds, or implied by the principal trait ref's
        // supertraits.
        let a_auto_traits: IndexSet<I::DefId> = a_data
            .auto_traits()
            .into_iter()
            .chain(a_data.principal_def_id().into_iter().flat_map(|principal_def_id| {
                elaborate::supertrait_def_ids(self.cx(), principal_def_id)
                    .filter(|def_id| self.cx().trait_is_auto(*def_id))
            }))
            .collect();

        // More than one projection in a_ty's bounds may match the projection
        // in b_ty's bound. Use this to first determine *which* apply without
        // having any inference side-effects. We process obligations because
        // unification may initially succeed due to deferred projection equality.
        let projection_may_match =
            |ecx: &mut EvalCtxt<'_, D>,
             source_projection: ty::Binder<I, ty::ExistentialProjection<I>>,
             target_projection: ty::Binder<I, ty::ExistentialProjection<I>>| {
                source_projection.item_def_id() == target_projection.item_def_id()
                    && ecx
                        .probe(|_| ProbeKind::UpcastProjectionCompatibility)
                        .enter(|ecx| -> Result<_, NoSolution> {
                            ecx.enter_forall(target_projection, |ecx, target_projection| {
                                let source_projection =
                                    ecx.instantiate_binder_with_infer(source_projection);
                                ecx.eq(param_env, source_projection, target_projection)?;
                                ecx.try_evaluate_added_goals()
                            })
                        })
                        .is_ok()
            };

        self.probe_trait_candidate(source).enter(|ecx| {
            for bound in b_data.iter() {
                match bound.skip_binder() {
                    // Check that a's supertrait (upcast_principal) is compatible
                    // with the target (b_ty).
                    ty::ExistentialPredicate::Trait(target_principal) => {
                        let source_principal = upcast_principal.unwrap();
                        let target_principal = bound.rebind(target_principal);
                        ecx.enter_forall(target_principal, |ecx, target_principal| {
                            let source_principal =
                                ecx.instantiate_binder_with_infer(source_principal);
                            ecx.eq(param_env, source_principal, target_principal)?;
                            ecx.try_evaluate_added_goals()
                        })?;
                    }
                    // Check that b_ty's projection is satisfied by exactly one of
                    // a_ty's projections. First, we look through the list to see if
                    // any match. If not, error. Then, if *more* than one matches, we
                    // return ambiguity. Otherwise, if exactly one matches, equate
                    // it with b_ty's projection.
                    ty::ExistentialPredicate::Projection(target_projection) => {
                        let target_projection = bound.rebind(target_projection);
                        let mut matching_projections =
                            a_data.projection_bounds().into_iter().filter(|source_projection| {
                                projection_may_match(ecx, *source_projection, target_projection)
                            });
                        let Some(source_projection) = matching_projections.next() else {
                            return Err(NoSolution);
                        };
                        if matching_projections.next().is_some() {
                            return ecx.evaluate_added_goals_and_make_canonical_response(
                                Certainty::AMBIGUOUS,
                            );
                        }
                        ecx.enter_forall(target_projection, |ecx, target_projection| {
                            let source_projection =
                                ecx.instantiate_binder_with_infer(source_projection);
                            ecx.eq(param_env, source_projection, target_projection)?;
                            ecx.try_evaluate_added_goals()
                        })?;
                    }
                    // Check that b_ty's auto traits are present in a_ty's bounds.
                    ty::ExistentialPredicate::AutoTrait(def_id) => {
                        if !a_auto_traits.contains(&def_id) {
                            return Err(NoSolution);
                        }
                    }
                }
            }

            // Also require that a_ty's lifetime outlives b_ty's lifetime.
            ecx.add_goal(
                GoalSource::ImplWhereBound,
                Goal::new(ecx.cx(), param_env, ty::OutlivesPredicate(a_region, b_region)),
            );

            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    /// We have the following builtin impls for arrays:
    /// ```ignore (builtin impl example)
    /// impl<T: ?Sized, const N: usize> Unsize<[T]> for [T; N] {}
    /// ```
    /// While the impl itself could theoretically not be builtin,
    /// the actual unsizing behavior is builtin. Its also easier to
    /// make all impls of `Unsize` builtin as we're able to use
    /// `#[rustc_deny_explicit_impl]` in this case.
    fn consider_builtin_array_unsize(
        &mut self,
        goal: Goal<I, (I::Ty, I::Ty)>,
        a_elem_ty: I::Ty,
        b_elem_ty: I::Ty,
    ) -> Result<Candidate<I>, NoSolution> {
        self.eq(goal.param_env, a_elem_ty, b_elem_ty)?;
        self.probe_builtin_trait_candidate(BuiltinImplSource::Misc)
            .enter(|ecx| ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes))
    }

    /// We generate a builtin `Unsize` impls for structs with generic parameters only
    /// mentioned by the last field.
    /// ```ignore (builtin impl example)
    /// struct Foo<T, U: ?Sized> {
    ///     sized_field: Vec<T>,
    ///     unsizable: Box<U>,
    /// }
    /// // results in the following builtin impl
    /// impl<T: ?Sized, U: ?Sized, V: ?Sized> Unsize<Foo<T, V>> for Foo<T, U>
    /// where
    ///     Box<U>: Unsize<Box<V>>,
    /// {}
    /// ```
    fn consider_builtin_struct_unsize(
        &mut self,
        goal: Goal<I, (I::Ty, I::Ty)>,
        def: I::AdtDef,
        a_args: I::GenericArgs,
        b_args: I::GenericArgs,
    ) -> Result<Candidate<I>, NoSolution> {
        let cx = self.cx();
        let Goal { predicate: (_a_ty, b_ty), .. } = goal;

        let unsizing_params = cx.unsizing_params_for_adt(def.def_id());
        // We must be unsizing some type parameters. This also implies
        // that the struct has a tail field.
        if unsizing_params.is_empty() {
            return Err(NoSolution);
        }

        let tail_field_ty = def.struct_tail_ty(cx).unwrap();

        let a_tail_ty = tail_field_ty.instantiate(cx, a_args);
        let b_tail_ty = tail_field_ty.instantiate(cx, b_args);

        // Instantiate just the unsizing params from B into A. The type after
        // this instantiation must be equal to B. This is so we don't unsize
        // unrelated type parameters.
        let new_a_args = cx.mk_args_from_iter(a_args.iter().enumerate().map(|(i, a)| {
            if unsizing_params.contains(i as u32) { b_args.get(i).unwrap() } else { a }
        }));
        let unsized_a_ty = Ty::new_adt(cx, def, new_a_args);

        // Finally, we require that `TailA: Unsize<TailB>` for the tail field
        // types.
        self.eq(goal.param_env, unsized_a_ty, b_ty)?;
        self.add_goal(
            GoalSource::ImplWhereBound,
            goal.with(
                cx,
                ty::TraitRef::new(cx, cx.require_lang_item(TraitSolverLangItem::Unsize), [
                    a_tail_ty, b_tail_ty,
                ]),
            ),
        );
        self.probe_builtin_trait_candidate(BuiltinImplSource::Misc)
            .enter(|ecx| ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes))
    }

    /// We generate the following builtin impl for tuples of all sizes.
    ///
    /// This impl is still unstable and we emit a feature error when it
    /// when it is used by a coercion.
    /// ```ignore (builtin impl example)
    /// impl<T: ?Sized, U: ?Sized, V: ?Sized> Unsize<(T, V)> for (T, U)
    /// where
    ///     U: Unsize<V>,
    /// {}
    /// ```
    fn consider_builtin_tuple_unsize(
        &mut self,
        goal: Goal<I, (I::Ty, I::Ty)>,
        a_tys: I::Tys,
        b_tys: I::Tys,
    ) -> Result<Candidate<I>, NoSolution> {
        let cx = self.cx();
        let Goal { predicate: (_a_ty, b_ty), .. } = goal;

        let (&a_last_ty, a_rest_tys) = a_tys.split_last().unwrap();
        let b_last_ty = b_tys.last().unwrap();

        // Instantiate just the tail field of B., and require that they're equal.
        let unsized_a_ty = Ty::new_tup_from_iter(cx, a_rest_tys.iter().copied().chain([b_last_ty]));
        self.eq(goal.param_env, unsized_a_ty, b_ty)?;

        // Similar to ADTs, require that we can unsize the tail.
        self.add_goal(
            GoalSource::ImplWhereBound,
            goal.with(
                cx,
                ty::TraitRef::new(cx, cx.require_lang_item(TraitSolverLangItem::Unsize), [
                    a_last_ty, b_last_ty,
                ]),
            ),
        );
        self.probe_builtin_trait_candidate(BuiltinImplSource::TupleUnsizing)
            .enter(|ecx| ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes))
    }

    // Return `Some` if there is an impl (built-in or user provided) that may
    // hold for the self type of the goal, which for coherence and soundness
    // purposes must disqualify the built-in auto impl assembled by considering
    // the type's constituent types.
    fn disqualify_auto_trait_candidate_due_to_possible_impl(
        &mut self,
        goal: Goal<I, TraitPredicate<I>>,
    ) -> Option<Result<Candidate<I>, NoSolution>> {
        let self_ty = goal.predicate.self_ty();
        match self_ty.kind() {
            // Stall int and float vars until they are resolved to a concrete
            // numerical type. That's because the check for impls below treats
            // int vars as matching any impl. Even if we filtered such impls,
            // we probably don't want to treat an `impl !AutoTrait for i32` as
            // disqualifying the built-in auto impl for `i64: AutoTrait` either.
            ty::Infer(ty::IntVar(_) | ty::FloatVar(_)) => {
                Some(self.forced_ambiguity(MaybeCause::Ambiguity))
            }

            // These types cannot be structurally decomposed into constituent
            // types, and therefore have no built-in auto impl.
            ty::Dynamic(..)
            | ty::Param(..)
            | ty::Foreign(..)
            | ty::Alias(ty::Projection | ty::Weak | ty::Inherent, ..)
            | ty::Placeholder(..) => Some(Err(NoSolution)),

            ty::Infer(_) | ty::Bound(_, _) => panic!("unexpected type `{self_ty:?}`"),

            // Coroutines have one special built-in candidate, `Unpin`, which
            // takes precedence over the structural auto trait candidate being
            // assembled.
            ty::Coroutine(def_id, _)
                if self.cx().is_lang_item(goal.predicate.def_id(), TraitSolverLangItem::Unpin) =>
            {
                match self.cx().coroutine_movability(def_id) {
                    Movability::Static => Some(Err(NoSolution)),
                    Movability::Movable => Some(
                        self.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
                            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                        }),
                    ),
                }
            }

            // If we still have an alias here, it must be rigid. For opaques, it's always
            // okay to consider auto traits because that'll reveal its hidden type. For
            // non-opaque aliases, we will not assemble any candidates since there's no way
            // to further look into its type.
            ty::Alias(..) => None,

            // For rigid types, any possible implementation that could apply to
            // the type (even if after unification and processing nested goals
            // it does not hold) will disqualify the built-in auto impl.
            //
            // This differs from the current stable behavior and fixes #84857.
            // Due to breakage found via crater, we currently instead lint
            // patterns which can be used to exploit this unsoundness on stable,
            // see #93367 for more details.
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Str
            | ty::Array(_, _)
            | ty::Pat(_, _)
            | ty::Slice(_)
            | ty::RawPtr(_, _)
            | ty::Ref(_, _, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(_, _)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::Tuple(_)
            | ty::Adt(_, _) => {
                let mut disqualifying_impl = None;
                self.cx().for_each_relevant_impl(
                    goal.predicate.def_id(),
                    goal.predicate.self_ty(),
                    |impl_def_id| {
                        disqualifying_impl = Some(impl_def_id);
                    },
                );
                if let Some(def_id) = disqualifying_impl {
                    trace!(?def_id, ?goal, "disqualified auto-trait implementation");
                    // No need to actually consider the candidate here,
                    // since we do that in `consider_impl_candidate`.
                    return Some(Err(NoSolution));
                } else {
                    None
                }
            }
            ty::Error(_) => None,
        }
    }

    /// Convenience function for traits that are structural, i.e. that only
    /// have nested subgoals that only change the self type. Unlike other
    /// evaluate-like helpers, this does a probe, so it doesn't need to be
    /// wrapped in one.
    fn probe_and_evaluate_goal_for_constituent_tys(
        &mut self,
        source: CandidateSource<I>,
        goal: Goal<I, TraitPredicate<I>>,
        constituent_tys: impl Fn(
            &EvalCtxt<'_, D>,
            I::Ty,
        ) -> Result<Vec<ty::Binder<I, I::Ty>>, NoSolution>,
    ) -> Result<Candidate<I>, NoSolution> {
        self.probe_trait_candidate(source).enter(|ecx| {
            let goals = constituent_tys(ecx, goal.predicate.self_ty())?
                .into_iter()
                .map(|ty| {
                    ecx.enter_forall(ty, |ecx, ty| {
                        goal.with(ecx.cx(), goal.predicate.with_self_ty(ecx.cx(), ty))
                    })
                })
                .collect::<Vec<_>>();
            ecx.add_goals(GoalSource::ImplWhereBound, goals);
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }
}

/// How we've proven this trait goal.
///
/// This is used by `NormalizesTo` goals to only normalize
/// by using the same 'kind of candidate' we've used to prove
/// its corresponding trait goal. Most notably, we do not
/// normalize by using an impl if the trait goal has been
/// proven via a `ParamEnv` candidate.
///
/// This is necessary to avoid unnecessary region constraints,
/// see trait-system-refactor-initiative#125 for more details.
#[derive(Debug, Clone, Copy)]
pub(super) enum TraitGoalProvenVia {
    /// We've proven the trait goal by something which is
    /// is not a non-global where-bound or an alias-bound.
    ///
    /// This means we don't disable any candidates during
    /// normalization.
    Misc,
    ParamEnv,
    AliasBound,
}

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    pub(super) fn merge_trait_candidates(
        &mut self,
        goal: Goal<I, TraitPredicate<I>>,
        candidates: Vec<Candidate<I>>,
    ) -> Result<(CanonicalResponse<I>, Option<TraitGoalProvenVia>), NoSolution> {
        if let TypingMode::Coherence = self.typing_mode() {
            let all_candidates: Vec<_> = candidates.into_iter().map(|c| c.result).collect();
            return if let Some(response) = self.try_merge_responses(&all_candidates) {
                Ok((response, Some(TraitGoalProvenVia::Misc)))
            } else {
                self.flounder(&all_candidates).map(|r| (r, None))
            };
        }

        // FIXME: prefer trivial builtin impls

        // If there are non-global where-bounds, prefer where-bounds
        // (including global ones) over everything else.
        let has_non_global_where_bounds = candidates.iter().any(|c| match c.source {
            CandidateSource::ParamEnv(idx) => {
                let where_bound = goal.param_env.caller_bounds().get(idx);
                where_bound.has_bound_vars() || !where_bound.is_global()
            }
            _ => false,
        });
        if has_non_global_where_bounds {
            let where_bounds: Vec<_> = candidates
                .iter()
                .filter(|c| matches!(c.source, CandidateSource::ParamEnv(_)))
                .map(|c| c.result)
                .collect();

            return if let Some(response) = self.try_merge_responses(&where_bounds) {
                Ok((response, Some(TraitGoalProvenVia::ParamEnv)))
            } else {
                Ok((self.bail_with_ambiguity(&where_bounds), None))
            };
        }

        if candidates.iter().any(|c| matches!(c.source, CandidateSource::AliasBound)) {
            let alias_bounds: Vec<_> = candidates
                .iter()
                .filter(|c| matches!(c.source, CandidateSource::AliasBound))
                .map(|c| c.result)
                .collect();
            return if let Some(response) = self.try_merge_responses(&alias_bounds) {
                Ok((response, Some(TraitGoalProvenVia::AliasBound)))
            } else {
                Ok((self.bail_with_ambiguity(&alias_bounds), None))
            };
        }

        let all_candidates: Vec<_> = candidates.into_iter().map(|c| c.result).collect();
        if let Some(response) = self.try_merge_responses(&all_candidates) {
            Ok((response, Some(TraitGoalProvenVia::Misc)))
        } else {
            self.flounder(&all_candidates).map(|r| (r, None))
        }
    }

    #[instrument(level = "trace", skip(self))]
    pub(super) fn compute_trait_goal(
        &mut self,
        goal: Goal<I, TraitPredicate<I>>,
    ) -> Result<(CanonicalResponse<I>, Option<TraitGoalProvenVia>), NoSolution> {
        let candidates = self.assemble_and_evaluate_candidates(goal);
        self.merge_trait_candidates(goal, candidates)
    }
}
