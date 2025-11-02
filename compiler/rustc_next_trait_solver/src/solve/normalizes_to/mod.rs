mod anon_const;
mod free_alias;
mod inherent;
mod opaque_types;

use rustc_type_ir::fast_reject::DeepRejectCtxt;
use rustc_type_ir::inherent::*;
use rustc_type_ir::lang_items::{SolverAdtLangItem, SolverLangItem, SolverTraitLangItem};
use rustc_type_ir::solve::SizedTraitKind;
use rustc_type_ir::{self as ty, Interner, NormalizesTo, PredicateKind, Upcast as _};
use tracing::instrument;

use crate::delegate::SolverDelegate;
use crate::solve::assembly::structural_traits::{self, AsyncCallableRelevantTypes};
use crate::solve::assembly::{self, Candidate};
use crate::solve::inspect::ProbeKind;
use crate::solve::{
    BuiltinImplSource, CandidateSource, Certainty, EvalCtxt, Goal, GoalSource, MaybeCause,
    NoSolution, QueryResult,
};

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    #[instrument(level = "trace", skip(self), ret)]
    pub(super) fn compute_normalizes_to_goal(
        &mut self,
        goal: Goal<I, NormalizesTo<I>>,
    ) -> QueryResult<I> {
        debug_assert!(self.term_is_fully_unconstrained(goal));
        let cx = self.cx();
        match goal.predicate.alias.kind(cx) {
            ty::AliasTermKind::ProjectionTy | ty::AliasTermKind::ProjectionConst => {
                let trait_ref = goal.predicate.alias.trait_ref(cx);
                let (_, proven_via) =
                    self.probe(|_| ProbeKind::ShadowedEnvProbing).enter(|ecx| {
                        let trait_goal: Goal<I, ty::TraitPredicate<I>> = goal.with(cx, trait_ref);
                        ecx.compute_trait_goal(trait_goal)
                    })?;
                self.assemble_and_merge_candidates(proven_via, goal, |ecx| {
                    ecx.probe(|&result| ProbeKind::RigidAlias { result }).enter(|this| {
                        this.structurally_instantiate_normalizes_to_term(
                            goal,
                            goal.predicate.alias,
                        );
                        this.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                    })
                })
            }
            ty::AliasTermKind::InherentTy | ty::AliasTermKind::InherentConst => {
                self.normalize_inherent_associated_term(goal)
            }
            ty::AliasTermKind::OpaqueTy => self.normalize_opaque_type(goal),
            ty::AliasTermKind::FreeTy | ty::AliasTermKind::FreeConst => {
                self.normalize_free_alias(goal)
            }
            ty::AliasTermKind::UnevaluatedConst => self.normalize_anon_const(goal),
        }
    }

    /// When normalizing an associated item, constrain the expected term to `term`.
    ///
    /// We know `term` to always be a fully unconstrained inference variable, so
    /// `eq` should never fail here. However, in case `term` contains aliases, we
    /// emit nested `AliasRelate` goals to structurally normalize the alias.
    pub fn instantiate_normalizes_to_term(
        &mut self,
        goal: Goal<I, NormalizesTo<I>>,
        term: I::Term,
    ) {
        self.eq(goal.param_env, goal.predicate.term, term)
            .expect("expected goal term to be fully unconstrained");
    }

    /// Unlike `instantiate_normalizes_to_term` this instantiates the expected term
    /// with a rigid alias. Using this is pretty much always wrong.
    pub fn structurally_instantiate_normalizes_to_term(
        &mut self,
        goal: Goal<I, NormalizesTo<I>>,
        term: ty::AliasTerm<I>,
    ) {
        self.relate_rigid_alias_non_alias(goal.param_env, term, ty::Invariant, goal.predicate.term)
            .expect("expected goal term to be fully unconstrained");
    }
}

impl<D, I> assembly::GoalKind<D> for NormalizesTo<I>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    fn self_ty(self) -> I::Ty {
        self.self_ty()
    }

    fn trait_ref(self, cx: I) -> ty::TraitRef<I> {
        self.alias.trait_ref(cx)
    }

    fn with_replaced_self_ty(self, cx: I, self_ty: I::Ty) -> Self {
        self.with_replaced_self_ty(cx, self_ty)
    }

    fn trait_def_id(self, cx: I) -> I::TraitId {
        self.trait_def_id(cx)
    }

    fn fast_reject_assumption(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
        assumption: I::Clause,
    ) -> Result<(), NoSolution> {
        if let Some(projection_pred) = assumption.as_projection_clause()
            && projection_pred.item_def_id() == goal.predicate.def_id()
            && DeepRejectCtxt::relate_rigid_rigid(ecx.cx()).args_may_unify(
                goal.predicate.alias.args,
                projection_pred.skip_binder().projection_term.args,
            )
        {
            Ok(())
        } else {
            Err(NoSolution)
        }
    }

    fn match_assumption(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
        assumption: I::Clause,
        then: impl FnOnce(&mut EvalCtxt<'_, D>) -> QueryResult<I>,
    ) -> QueryResult<I> {
        let cx = ecx.cx();
        // FIXME(generic_associated_types): Addresses aggressive inference in #92917.
        //
        // If this type is a GAT with currently unconstrained arguments, we do not
        // want to normalize it via a candidate which only applies for a specific
        // instantiation. We could otherwise keep the GAT as rigid and succeed this way.
        // See tests/ui/generic-associated-types/no-incomplete-gat-arg-inference.rs.
        //
        // This only avoids normalization if the GAT arguments are fully unconstrained.
        // This is quite arbitrary but fixing it causes some ambiguity, see #125196.
        match goal.predicate.alias.kind(cx) {
            ty::AliasTermKind::ProjectionTy | ty::AliasTermKind::ProjectionConst => {
                for arg in goal.predicate.alias.own_args(cx).iter() {
                    let Some(term) = arg.as_term() else {
                        continue;
                    };
                    let term = ecx.structurally_normalize_term(goal.param_env, term)?;
                    if term.is_infer() {
                        return ecx.evaluate_added_goals_and_make_canonical_response(
                            Certainty::AMBIGUOUS,
                        );
                    }
                }
            }
            ty::AliasTermKind::OpaqueTy
            | ty::AliasTermKind::InherentTy
            | ty::AliasTermKind::InherentConst
            | ty::AliasTermKind::FreeTy
            | ty::AliasTermKind::FreeConst
            | ty::AliasTermKind::UnevaluatedConst => {}
        }

        let projection_pred = assumption.as_projection_clause().unwrap();

        let assumption_projection_pred = ecx.instantiate_binder_with_infer(projection_pred);
        ecx.eq(goal.param_env, goal.predicate.alias, assumption_projection_pred.projection_term)?;

        ecx.instantiate_normalizes_to_term(goal, assumption_projection_pred.term);

        // Add GAT where clauses from the trait's definition
        // FIXME: We don't need these, since these are the type's own WF obligations.
        ecx.add_goals(
            GoalSource::AliasWellFormed,
            cx.own_predicates_of(goal.predicate.def_id())
                .iter_instantiated(cx, goal.predicate.alias.args)
                .map(|pred| goal.with(cx, pred)),
        );

        then(ecx)
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
        goal: Goal<I, NormalizesTo<I>>,
        impl_def_id: I::ImplId,
        then: impl FnOnce(&mut EvalCtxt<'_, D>, Certainty) -> QueryResult<I>,
    ) -> Result<Candidate<I>, NoSolution> {
        let cx = ecx.cx();

        let goal_trait_ref = goal.predicate.alias.trait_ref(cx);
        let impl_trait_ref = cx.impl_trait_ref(impl_def_id);
        if !DeepRejectCtxt::relate_rigid_infer(ecx.cx()).args_may_unify(
            goal.predicate.alias.trait_ref(cx).args,
            impl_trait_ref.skip_binder().args,
        ) {
            return Err(NoSolution);
        }

        // We have to ignore negative impls when projecting.
        let impl_polarity = cx.impl_polarity(impl_def_id);
        match impl_polarity {
            ty::ImplPolarity::Negative => return Err(NoSolution),
            ty::ImplPolarity::Reservation => {
                unimplemented!("reservation impl for trait with assoc item: {:?}", goal)
            }
            ty::ImplPolarity::Positive => {}
        };

        ecx.probe_trait_candidate(CandidateSource::Impl(impl_def_id)).enter(|ecx| {
            let impl_args = ecx.fresh_args_for_item(impl_def_id.into());
            let impl_trait_ref = impl_trait_ref.instantiate(cx, impl_args);

            ecx.eq(goal.param_env, goal_trait_ref, impl_trait_ref)?;

            let where_clause_bounds = cx
                .predicates_of(impl_def_id.into())
                .iter_instantiated(cx, impl_args)
                .map(|pred| goal.with(cx, pred));
            ecx.add_goals(GoalSource::ImplWhereBound, where_clause_bounds);

            // Bail if the nested goals don't hold here. This is to avoid unnecessarily
            // computing the `type_of` query for associated types that never apply, as
            // this may result in query cycles in the case of RPITITs.
            // See <https://github.com/rust-lang/trait-system-refactor-initiative/issues/185>.
            ecx.try_evaluate_added_goals()?;

            // Add GAT where clauses from the trait's definition.
            // FIXME: We don't need these, since these are the type's own WF obligations.
            ecx.add_goals(
                GoalSource::AliasWellFormed,
                cx.own_predicates_of(goal.predicate.def_id())
                    .iter_instantiated(cx, goal.predicate.alias.args)
                    .map(|pred| goal.with(cx, pred)),
            );

            let error_response = |ecx: &mut EvalCtxt<'_, D>, guar| {
                let error_term = match goal.predicate.alias.kind(cx) {
                    ty::AliasTermKind::ProjectionTy => Ty::new_error(cx, guar).into(),
                    ty::AliasTermKind::ProjectionConst => Const::new_error(cx, guar).into(),
                    kind => panic!("expected projection, found {kind:?}"),
                };
                ecx.instantiate_normalizes_to_term(goal, error_term);
                ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            };

            let target_item_def_id = match ecx.fetch_eligible_assoc_item(
                goal_trait_ref,
                goal.predicate.def_id(),
                impl_def_id,
            ) {
                Ok(Some(target_item_def_id)) => target_item_def_id,
                Ok(None) => {
                    match ecx.typing_mode() {
                        // In case the associated item is hidden due to specialization,
                        // normalizing this associated item is always ambiguous. Treating
                        // the associated item as rigid would be incomplete and allow for
                        // overlapping impls, see #105782.
                        //
                        // As this ambiguity is unavoidable we emit a nested ambiguous
                        // goal instead of using `Certainty::AMBIGUOUS`. This allows us to
                        // return the nested goals to the parent `AliasRelate` goal. This
                        // would be relevant if any of the nested goals refer to the `term`.
                        // This is not the case here and we only prefer adding an ambiguous
                        // nested goal for consistency.
                        ty::TypingMode::Coherence => {
                            ecx.add_goal(GoalSource::Misc, goal.with(cx, PredicateKind::Ambiguous));
                            return ecx
                                .evaluate_added_goals_and_make_canonical_response(Certainty::Yes);
                        }
                        // Outside of coherence, we treat the associated item as rigid instead.
                        ty::TypingMode::Analysis { .. }
                        | ty::TypingMode::Borrowck { .. }
                        | ty::TypingMode::PostBorrowckAnalysis { .. }
                        | ty::TypingMode::PostAnalysis => {
                            ecx.structurally_instantiate_normalizes_to_term(
                                goal,
                                goal.predicate.alias,
                            );
                            return ecx
                                .evaluate_added_goals_and_make_canonical_response(Certainty::Yes);
                        }
                    };
                }
                Err(guar) => return error_response(ecx, guar),
            };

            if !cx.has_item_definition(target_item_def_id) {
                // If the impl is missing an item, it's either because the user forgot to
                // provide it, or the user is not *obligated* to provide it (because it
                // has a trivially false `Sized` predicate). If it's the latter, we cannot
                // delay a bug because we can have trivially false where clauses, so we
                // treat it as rigid.
                if cx.impl_self_is_guaranteed_unsized(impl_def_id) {
                    match ecx.typing_mode() {
                        // Trying to normalize such associated items is always ambiguous
                        // during coherence to avoid cyclic reasoning. See the example in
                        // tests/ui/traits/trivial-unsized-projection-in-coherence.rs.
                        //
                        // As this ambiguity is unavoidable we emit a nested ambiguous
                        // goal instead of using `Certainty::AMBIGUOUS`. This allows us to
                        // return the nested goals to the parent `AliasRelate` goal. This
                        // would be relevant if any of the nested goals refer to the `term`.
                        // This is not the case here and we only prefer adding an ambiguous
                        // nested goal for consistency.
                        ty::TypingMode::Coherence => {
                            ecx.add_goal(GoalSource::Misc, goal.with(cx, PredicateKind::Ambiguous));
                            return then(ecx, Certainty::Yes);
                        }
                        ty::TypingMode::Analysis { .. }
                        | ty::TypingMode::Borrowck { .. }
                        | ty::TypingMode::PostBorrowckAnalysis { .. }
                        | ty::TypingMode::PostAnalysis => {
                            ecx.structurally_instantiate_normalizes_to_term(
                                goal,
                                goal.predicate.alias,
                            );
                            return then(ecx, Certainty::Yes);
                        }
                    }
                } else {
                    return error_response(ecx, cx.delay_bug("missing item"));
                }
            }

            let target_container_def_id = cx.parent(target_item_def_id);

            // Getting the right args here is complex, e.g. given:
            // - a goal `<Vec<u32> as Trait<i32>>::Assoc<u64>`
            // - the applicable impl `impl<T> Trait<i32> for Vec<T>`
            // - and the impl which defines `Assoc` being `impl<T, U> Trait<U> for Vec<T>`
            //
            // We first rebase the goal args onto the impl, going from `[Vec<u32>, i32, u64]`
            // to `[u32, u64]`.
            //
            // And then map these args to the args of the defining impl of `Assoc`, going
            // from `[u32, u64]` to `[u32, i32, u64]`.
            let target_args = ecx.translate_args(
                goal,
                impl_def_id,
                impl_args,
                impl_trait_ref,
                target_container_def_id,
            )?;

            if !cx.check_args_compatible(target_item_def_id, target_args) {
                return error_response(
                    ecx,
                    cx.delay_bug("associated item has mismatched arguments"),
                );
            }

            // Finally we construct the actual value of the associated type.
            let term = match goal.predicate.alias.kind(cx) {
                ty::AliasTermKind::ProjectionTy => {
                    cx.type_of(target_item_def_id).map_bound(|ty| ty.into())
                }
                ty::AliasTermKind::ProjectionConst => {
                    // FIXME(mgca): once const items are actual aliases defined as equal to type system consts
                    // this should instead return that.
                    if cx.features().associated_const_equality() {
                        panic!("associated const projection is not supported yet")
                    } else {
                        ty::EarlyBinder::bind(
                            Const::new_error_with_message(
                                cx,
                                "associated const projection is not supported yet",
                            )
                            .into(),
                        )
                    }
                }
                kind => panic!("expected projection, found {kind:?}"),
            };

            ecx.instantiate_normalizes_to_term(goal, term.instantiate(cx, target_args));
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    /// Fail to normalize if the predicate contains an error, alternatively, we could normalize to `ty::Error`
    /// and succeed. Can experiment with this to figure out what results in better error messages.
    fn consider_error_guaranteed_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        _guar: I::ErrorGuaranteed,
    ) -> Result<Candidate<I>, NoSolution> {
        Err(NoSolution)
    }

    fn consider_auto_trait_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        _goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        ecx.cx().delay_bug("associated types not allowed on auto traits");
        Err(NoSolution)
    }

    fn consider_trait_alias_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        panic!("trait aliases do not have associated types: {:?}", goal);
    }

    fn consider_builtin_sizedness_candidates(
        _ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
        _sizedness: SizedTraitKind,
    ) -> Result<Candidate<I>, NoSolution> {
        panic!("`Sized`/`MetaSized` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_copy_clone_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        panic!("`Copy`/`Clone` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_fn_ptr_trait_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        panic!("`FnPtr` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_fn_trait_candidates(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
        goal_kind: ty::ClosureKind,
    ) -> Result<Candidate<I>, NoSolution> {
        let cx = ecx.cx();
        let tupled_inputs_and_output =
            match structural_traits::extract_tupled_inputs_and_output_from_callable(
                cx,
                goal.predicate.self_ty(),
                goal_kind,
            )? {
                Some(tupled_inputs_and_output) => tupled_inputs_and_output,
                None => {
                    return ecx.forced_ambiguity(MaybeCause::Ambiguity);
                }
            };
        let (inputs, output) = ecx.instantiate_binder_with_infer(tupled_inputs_and_output);

        // A built-in `Fn` impl only holds if the output is sized.
        // (FIXME: technically we only need to check this if the type is a fn ptr...)
        let output_is_sized_pred =
            ty::TraitRef::new(cx, cx.require_trait_lang_item(SolverTraitLangItem::Sized), [output]);

        let pred = ty::ProjectionPredicate {
            projection_term: ty::AliasTerm::new(
                cx,
                goal.predicate.def_id(),
                [goal.predicate.self_ty(), inputs],
            ),
            term: output.into(),
        }
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
        let cx = ecx.cx();

        let env_region = match goal_kind {
            ty::ClosureKind::Fn | ty::ClosureKind::FnMut => goal.predicate.alias.args.region_at(2),
            // Doesn't matter what this region is
            ty::ClosureKind::FnOnce => Region::new_static(cx),
        };
        let (tupled_inputs_and_output_and_coroutine, nested_preds) =
            structural_traits::extract_tupled_inputs_and_output_from_async_callable(
                cx,
                goal.predicate.self_ty(),
                goal_kind,
                env_region,
            )?;
        let AsyncCallableRelevantTypes {
            tupled_inputs_ty,
            output_coroutine_ty,
            coroutine_return_ty,
        } = ecx.instantiate_binder_with_infer(tupled_inputs_and_output_and_coroutine);

        // A built-in `AsyncFn` impl only holds if the output is sized.
        // (FIXME: technically we only need to check this if the type is a fn ptr...)
        let output_is_sized_pred = ty::TraitRef::new(
            cx,
            cx.require_trait_lang_item(SolverTraitLangItem::Sized),
            [output_coroutine_ty],
        );

        let (projection_term, term) =
            if cx.is_lang_item(goal.predicate.def_id(), SolverLangItem::CallOnceFuture) {
                (
                    ty::AliasTerm::new(
                        cx,
                        goal.predicate.def_id(),
                        [goal.predicate.self_ty(), tupled_inputs_ty],
                    ),
                    output_coroutine_ty.into(),
                )
            } else if cx.is_lang_item(goal.predicate.def_id(), SolverLangItem::CallRefFuture) {
                (
                    ty::AliasTerm::new(
                        cx,
                        goal.predicate.def_id(),
                        [
                            I::GenericArg::from(goal.predicate.self_ty()),
                            tupled_inputs_ty.into(),
                            env_region.into(),
                        ],
                    ),
                    output_coroutine_ty.into(),
                )
            } else if cx.is_lang_item(goal.predicate.def_id(), SolverLangItem::AsyncFnOnceOutput) {
                (
                    ty::AliasTerm::new(
                        cx,
                        goal.predicate.def_id(),
                        [goal.predicate.self_ty(), tupled_inputs_ty],
                    ),
                    coroutine_return_ty.into(),
                )
            } else {
                panic!("no such associated type in `AsyncFn*`: {:?}", goal.predicate.def_id())
            };
        let pred = ty::ProjectionPredicate { projection_term, term }.upcast(cx);

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
        let [
            closure_fn_kind_ty,
            goal_kind_ty,
            borrow_region,
            tupled_inputs_ty,
            tupled_upvars_ty,
            coroutine_captures_by_ref_ty,
        ] = *goal.predicate.alias.args.as_slice()
        else {
            panic!();
        };

        // Bail if the upvars haven't been constrained.
        if tupled_upvars_ty.expect_ty().is_ty_var() {
            return ecx.forced_ambiguity(MaybeCause::Ambiguity);
        }

        let Some(closure_kind) = closure_fn_kind_ty.expect_ty().to_opt_closure_kind() else {
            // We don't need to worry about the self type being an infer var.
            return Err(NoSolution);
        };
        let Some(goal_kind) = goal_kind_ty.expect_ty().to_opt_closure_kind() else {
            return Err(NoSolution);
        };
        if !closure_kind.extends(goal_kind) {
            return Err(NoSolution);
        }

        let upvars_ty = ty::CoroutineClosureSignature::tupled_upvars_by_closure_kind(
            ecx.cx(),
            goal_kind,
            tupled_inputs_ty.expect_ty(),
            tupled_upvars_ty.expect_ty(),
            coroutine_captures_by_ref_ty.expect_ty(),
            borrow_region.expect_region(),
        );

        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
            ecx.instantiate_normalizes_to_term(goal, upvars_ty.into());
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_builtin_tuple_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        panic!("`Tuple` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_pointee_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        let cx = ecx.cx();
        let metadata_def_id = cx.require_lang_item(SolverLangItem::Metadata);
        assert_eq!(metadata_def_id, goal.predicate.def_id());
        let metadata_ty = match goal.predicate.self_ty().kind() {
            ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..)
            | ty::Array(..)
            | ty::Pat(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Infer(ty::IntVar(..) | ty::FloatVar(..))
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::Foreign(..) => Ty::new_unit(cx),

            ty::Error(e) => Ty::new_error(cx, e),

            ty::Str | ty::Slice(_) => Ty::new_usize(cx),

            ty::Dynamic(_, _) => {
                let dyn_metadata = cx.require_lang_item(SolverLangItem::DynMetadata);
                cx.type_of(dyn_metadata)
                    .instantiate(cx, &[I::GenericArg::from(goal.predicate.self_ty())])
            }

            ty::Alias(_, _) | ty::Param(_) | ty::Placeholder(..) => {
                // This is the "fallback impl" for type parameters, unnormalizable projections
                // and opaque types: If the `self_ty` is `Sized`, then the metadata is `()`.
                // FIXME(ptr_metadata): This impl overlaps with the other impls and shouldn't
                // exist. Instead, `Pointee<Metadata = ()>` should be a supertrait of `Sized`.
                let alias_bound_result =
                    ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
                        let sized_predicate = ty::TraitRef::new(
                            cx,
                            cx.require_trait_lang_item(SolverTraitLangItem::Sized),
                            [I::GenericArg::from(goal.predicate.self_ty())],
                        );
                        ecx.add_goal(GoalSource::Misc, goal.with(cx, sized_predicate));
                        ecx.instantiate_normalizes_to_term(goal, Ty::new_unit(cx).into());
                        ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                    });
                // In case the dummy alias-bound candidate does not apply, we instead treat this projection
                // as rigid.
                return alias_bound_result.or_else(|NoSolution| {
                    ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|this| {
                        this.structurally_instantiate_normalizes_to_term(
                            goal,
                            goal.predicate.alias,
                        );
                        this.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                    })
                });
            }

            ty::Adt(def, args) if def.is_struct() => match def.struct_tail_ty(cx) {
                None => Ty::new_unit(cx),
                Some(tail_ty) => {
                    Ty::new_projection(cx, metadata_def_id, [tail_ty.instantiate(cx, args)])
                }
            },
            ty::Adt(_, _) => Ty::new_unit(cx),

            ty::Tuple(elements) => match elements.last() {
                None => Ty::new_unit(cx),
                Some(tail_ty) => Ty::new_projection(cx, metadata_def_id, [tail_ty]),
            },

            ty::UnsafeBinder(_) => {
                // FIXME(unsafe_binder): Figure out how to handle pointee for unsafe binders.
                todo!()
            }

            ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_))
            | ty::Bound(..) => panic!(
                "unexpected self ty `{:?}` when normalizing `<T as Pointee>::Metadata`",
                goal.predicate.self_ty()
            ),
        };

        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
            ecx.instantiate_normalizes_to_term(goal, metadata_ty.into());
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_builtin_future_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        let self_ty = goal.predicate.self_ty();
        let ty::Coroutine(def_id, args) = self_ty.kind() else {
            return Err(NoSolution);
        };

        // Coroutines are not futures unless they come from `async` desugaring
        let cx = ecx.cx();
        if !cx.coroutine_is_async(def_id) {
            return Err(NoSolution);
        }

        let term = args.as_coroutine().return_ty().into();

        Self::probe_and_consider_implied_clause(
            ecx,
            CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
            goal,
            ty::ProjectionPredicate {
                projection_term: ty::AliasTerm::new(ecx.cx(), goal.predicate.def_id(), [self_ty]),
                term,
            }
            .upcast(cx),
            // Technically, we need to check that the future type is Sized,
            // but that's already proven by the coroutine being WF.
            [],
        )
    }

    fn consider_builtin_iterator_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        let self_ty = goal.predicate.self_ty();
        let ty::Coroutine(def_id, args) = self_ty.kind() else {
            return Err(NoSolution);
        };

        // Coroutines are not Iterators unless they come from `gen` desugaring
        let cx = ecx.cx();
        if !cx.coroutine_is_gen(def_id) {
            return Err(NoSolution);
        }

        let term = args.as_coroutine().yield_ty().into();

        Self::probe_and_consider_implied_clause(
            ecx,
            CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
            goal,
            ty::ProjectionPredicate {
                projection_term: ty::AliasTerm::new(ecx.cx(), goal.predicate.def_id(), [self_ty]),
                term,
            }
            .upcast(cx),
            // Technically, we need to check that the iterator type is Sized,
            // but that's already proven by the generator being WF.
            [],
        )
    }

    fn consider_builtin_fused_iterator_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        panic!("`FusedIterator` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_async_iterator_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        let self_ty = goal.predicate.self_ty();
        let ty::Coroutine(def_id, args) = self_ty.kind() else {
            return Err(NoSolution);
        };

        // Coroutines are not AsyncIterators unless they come from `gen` desugaring
        let cx = ecx.cx();
        if !cx.coroutine_is_async_gen(def_id) {
            return Err(NoSolution);
        }

        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
            let expected_ty = ecx.next_ty_infer();
            // Take `AsyncIterator<Item = I>` and turn it into the corresponding
            // coroutine yield ty `Poll<Option<I>>`.
            let wrapped_expected_ty = Ty::new_adt(
                cx,
                cx.adt_def(cx.require_adt_lang_item(SolverAdtLangItem::Poll)),
                cx.mk_args(&[Ty::new_adt(
                    cx,
                    cx.adt_def(cx.require_adt_lang_item(SolverAdtLangItem::Option)),
                    cx.mk_args(&[expected_ty.into()]),
                )
                .into()]),
            );
            let yield_ty = args.as_coroutine().yield_ty();
            ecx.eq(goal.param_env, wrapped_expected_ty, yield_ty)?;
            ecx.instantiate_normalizes_to_term(goal, expected_ty.into());
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_builtin_coroutine_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
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

        let term = if cx.is_lang_item(goal.predicate.def_id(), SolverLangItem::CoroutineReturn) {
            coroutine.return_ty().into()
        } else if cx.is_lang_item(goal.predicate.def_id(), SolverLangItem::CoroutineYield) {
            coroutine.yield_ty().into()
        } else {
            panic!("unexpected associated item `{:?}` for `{self_ty:?}`", goal.predicate.def_id())
        };

        Self::probe_and_consider_implied_clause(
            ecx,
            CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
            goal,
            ty::ProjectionPredicate {
                projection_term: ty::AliasTerm::new(
                    ecx.cx(),
                    goal.predicate.def_id(),
                    [self_ty, coroutine.resume_ty()],
                ),
                term,
            }
            .upcast(cx),
            // Technically, we need to check that the coroutine type is Sized,
            // but that's already proven by the coroutine being WF.
            [],
        )
    }

    fn consider_structural_builtin_unsize_candidates(
        _ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Vec<Candidate<I>> {
        panic!("`Unsize` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_discriminant_kind_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        let self_ty = goal.predicate.self_ty();
        let discriminant_ty = match self_ty.kind() {
            ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..)
            | ty::Array(..)
            | ty::Pat(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Infer(ty::IntVar(..) | ty::FloatVar(..))
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::Foreign(..)
            | ty::Adt(_, _)
            | ty::Str
            | ty::Slice(_)
            | ty::Dynamic(_, _)
            | ty::Tuple(_)
            | ty::Error(_) => self_ty.discriminant_ty(ecx.cx()),

            ty::UnsafeBinder(_) => {
                // FIXME(unsafe_binders): instantiate this with placeholders?? i guess??
                todo!("discr subgoal...")
            }

            // Given an alias, parameter, or placeholder we add an impl candidate normalizing to a rigid
            // alias. In case there's a where-bound further constraining this alias it is preferred over
            // this impl candidate anyways. It's still a bit scuffed.
            ty::Alias(_, _) | ty::Param(_) | ty::Placeholder(..) => {
                return ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
                    ecx.structurally_instantiate_normalizes_to_term(goal, goal.predicate.alias);
                    ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                });
            }

            ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_))
            | ty::Bound(..) => panic!(
                "unexpected self ty `{:?}` when normalizing `<T as DiscriminantKind>::Discriminant`",
                goal.predicate.self_ty()
            ),
        };

        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
            ecx.instantiate_normalizes_to_term(goal, discriminant_ty.into());
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_builtin_destruct_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        panic!("`Destruct` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_transmute_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        panic!("`TransmuteFrom` does not have an associated type: {:?}", goal)
    }

    fn consider_builtin_bikeshed_guaranteed_no_drop_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        unreachable!("`BikeshedGuaranteedNoDrop` does not have an associated type: {:?}", goal)
    }
}

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    fn translate_args(
        &mut self,
        goal: Goal<I, ty::NormalizesTo<I>>,
        impl_def_id: I::ImplId,
        impl_args: I::GenericArgs,
        impl_trait_ref: rustc_type_ir::TraitRef<I>,
        target_container_def_id: I::DefId,
    ) -> Result<I::GenericArgs, NoSolution> {
        let cx = self.cx();
        Ok(if target_container_def_id == impl_trait_ref.def_id.into() {
            // Default value from the trait definition. No need to rebase.
            goal.predicate.alias.args
        } else if target_container_def_id == impl_def_id.into() {
            // Same impl, no need to fully translate, just a rebase from
            // the trait is sufficient.
            goal.predicate.alias.args.rebase_onto(cx, impl_trait_ref.def_id.into(), impl_args)
        } else {
            let target_args = self.fresh_args_for_item(target_container_def_id);
            let target_trait_ref = cx
                .impl_trait_ref(target_container_def_id.try_into().unwrap())
                .instantiate(cx, target_args);
            // Relate source impl to target impl by equating trait refs.
            self.eq(goal.param_env, impl_trait_ref, target_trait_ref)?;
            // Also add predicates since they may be needed to constrain the
            // target impl's params.
            self.add_goals(
                GoalSource::Misc,
                cx.predicates_of(target_container_def_id)
                    .iter_instantiated(cx, target_args)
                    .map(|pred| goal.with(cx, pred)),
            );
            goal.predicate.alias.args.rebase_onto(cx, impl_trait_ref.def_id.into(), target_args)
        })
    }
}
