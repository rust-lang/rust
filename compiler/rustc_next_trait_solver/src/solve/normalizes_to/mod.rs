mod anon_const;
mod inherent;
mod opaque_types;
mod weak_types;

use rustc_type_ir::fast_reject::DeepRejectCtxt;
use rustc_type_ir::inherent::*;
use rustc_type_ir::lang_items::TraitSolverLangItem;
use rustc_type_ir::{self as ty, Interner, NormalizesTo, Upcast as _};
use tracing::instrument;

use crate::delegate::SolverDelegate;
use crate::solve::assembly::structural_traits::{self, AsyncCallableRelevantTypes};
use crate::solve::assembly::{self, Candidate};
use crate::solve::inspect::ProbeKind;
use crate::solve::{
    BuiltinImplSource, CandidateSource, Certainty, EvalCtxt, Goal, GoalSource, MaybeCause,
    NoSolution, QueryResult, Reveal,
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
        self.set_is_normalizes_to_goal();
        debug_assert!(self.term_is_fully_unconstrained(goal));
        let normalize_result = self
            .probe(|&result| ProbeKind::TryNormalizeNonRigid { result })
            .enter(|this| this.normalize_at_least_one_step(goal));

        match normalize_result {
            Ok(res) => Ok(res),
            Err(NoSolution) => {
                self.probe(|&result| ProbeKind::RigidAlias { result }).enter(|this| {
                    let Goal { param_env, predicate: NormalizesTo { alias, term } } = goal;
                    this.add_rigid_constraints(param_env, alias)?;
                    this.relate_rigid_alias_non_alias(param_env, alias, ty::Invariant, term)?;
                    this.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                })
            }
        }
    }

    /// Register any obligations that are used to validate that an alias should be
    /// treated as rigid.
    ///
    /// An alias may be considered rigid if it fails normalization, but we also don't
    /// want to consider aliases that are not well-formed to be rigid simply because
    /// they fail normalization.
    ///
    /// For example, some `<T as Trait>::Assoc` where `T: Trait` does not hold, or an
    /// opaque type whose hidden type doesn't actually satisfy the opaque item bounds.
    fn add_rigid_constraints(
        &mut self,
        param_env: I::ParamEnv,
        rigid_alias: ty::AliasTerm<I>,
    ) -> Result<(), NoSolution> {
        let cx = self.cx();
        match rigid_alias.kind(cx) {
            // Projections are rigid only if their trait ref holds,
            // and the GAT where-clauses hold.
            ty::AliasTermKind::ProjectionTy | ty::AliasTermKind::ProjectionConst => {
                let trait_ref = rigid_alias.trait_ref(cx);
                self.add_goal(GoalSource::AliasWellFormed, Goal::new(cx, param_env, trait_ref));
                Ok(())
            }
            ty::AliasTermKind::OpaqueTy => {
                match param_env.reveal() {
                    // In user-facing mode, paques are only rigid if we may not define it.
                    Reveal::UserFacing => {
                        if rigid_alias
                            .def_id
                            .as_local()
                            .is_some_and(|def_id| self.can_define_opaque_ty(def_id))
                        {
                            Err(NoSolution)
                        } else {
                            Ok(())
                        }
                    }
                    // Opaques are never rigid in reveal-all mode.
                    Reveal::All => Err(NoSolution),
                }
            }
            // FIXME(generic_const_exprs): we would need to support generic consts here
            ty::AliasTermKind::UnevaluatedConst => Err(NoSolution),
            // Inherent and weak types are never rigid. This type must not be well-formed.
            ty::AliasTermKind::WeakTy | ty::AliasTermKind::InherentTy => Err(NoSolution),
        }
    }

    /// Normalize the given alias by at least one step. If the alias is rigid, this
    /// returns `NoSolution`.
    #[instrument(level = "trace", skip(self), ret)]
    fn normalize_at_least_one_step(&mut self, goal: Goal<I, NormalizesTo<I>>) -> QueryResult<I> {
        match goal.predicate.alias.kind(self.cx()) {
            ty::AliasTermKind::ProjectionTy | ty::AliasTermKind::ProjectionConst => {
                let candidates = self.assemble_and_evaluate_candidates(goal);
                self.merge_candidates(candidates)
            }
            ty::AliasTermKind::InherentTy => self.normalize_inherent_associated_type(goal),
            ty::AliasTermKind::OpaqueTy => self.normalize_opaque_type(goal),
            ty::AliasTermKind::WeakTy => self.normalize_weak_type(goal),
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

    fn with_self_ty(self, cx: I, self_ty: I::Ty) -> Self {
        self.with_self_ty(cx, self_ty)
    }

    fn trait_def_id(self, cx: I) -> I::DefId {
        self.trait_def_id(cx)
    }

    fn probe_and_match_goal_against_assumption(
        ecx: &mut EvalCtxt<'_, D>,
        source: CandidateSource<I>,
        goal: Goal<I, Self>,
        assumption: I::Clause,
        then: impl FnOnce(&mut EvalCtxt<'_, D>) -> QueryResult<I>,
    ) -> Result<Candidate<I>, NoSolution> {
        if let Some(projection_pred) = assumption.as_projection_clause() {
            if projection_pred.projection_def_id() == goal.predicate.def_id() {
                let cx = ecx.cx();
                if !DeepRejectCtxt::relate_rigid_rigid(ecx.cx()).args_may_unify(
                    goal.predicate.alias.args,
                    projection_pred.skip_binder().projection_term.args,
                ) {
                    return Err(NoSolution);
                }
                ecx.probe_trait_candidate(source).enter(|ecx| {
                    let assumption_projection_pred =
                        ecx.instantiate_binder_with_infer(projection_pred);
                    ecx.eq(
                        goal.param_env,
                        goal.predicate.alias,
                        assumption_projection_pred.projection_term,
                    )?;

                    ecx.instantiate_normalizes_to_term(goal, assumption_projection_pred.term);

                    // Add GAT where clauses from the trait's definition
                    // FIXME: We don't need these, since these are the type's own WF obligations.
                    ecx.add_goals(
                        GoalSource::Misc,
                        cx.own_predicates_of(goal.predicate.def_id())
                            .iter_instantiated(cx, goal.predicate.alias.args)
                            .map(|pred| goal.with(cx, pred)),
                    );

                    then(ecx)
                })
            } else {
                Err(NoSolution)
            }
        } else {
            Err(NoSolution)
        }
    }

    fn consider_impl_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, NormalizesTo<I>>,
        impl_def_id: I::DefId,
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
            let impl_args = ecx.fresh_args_for_item(impl_def_id);
            let impl_trait_ref = impl_trait_ref.instantiate(cx, impl_args);

            ecx.eq(goal.param_env, goal_trait_ref, impl_trait_ref)?;

            let where_clause_bounds = cx
                .predicates_of(impl_def_id)
                .iter_instantiated(cx, impl_args)
                .map(|pred| goal.with(cx, pred));
            ecx.add_goals(GoalSource::ImplWhereBound, where_clause_bounds);

            // Add GAT where clauses from the trait's definition.
            // FIXME: We don't need these, since these are the type's own WF obligations.
            ecx.add_goals(
                GoalSource::Misc,
                cx.own_predicates_of(goal.predicate.def_id())
                    .iter_instantiated(cx, goal.predicate.alias.args)
                    .map(|pred| goal.with(cx, pred)),
            );

            // In case the associated item is hidden due to specialization, we have to
            // return ambiguity this would otherwise be incomplete, resulting in
            // unsoundness during coherence (#105782).
            let Some(target_item_def_id) = ecx.fetch_eligible_assoc_item(
                goal.param_env,
                goal_trait_ref,
                goal.predicate.def_id(),
                impl_def_id,
            )?
            else {
                return ecx.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS);
            };

            let error_response = |ecx: &mut EvalCtxt<'_, D>, msg: &str| {
                let guar = cx.delay_bug(msg);
                let error_term = match goal.predicate.alias.kind(cx) {
                    ty::AliasTermKind::ProjectionTy => Ty::new_error(cx, guar).into(),
                    ty::AliasTermKind::ProjectionConst => Const::new_error(cx, guar).into(),
                    kind => panic!("expected projection, found {kind:?}"),
                };
                ecx.instantiate_normalizes_to_term(goal, error_term);
                ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            };

            if !cx.has_item_definition(target_item_def_id) {
                return error_response(ecx, "missing item");
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
                return error_response(ecx, "associated item has mismatched arguments");
            }

            // Finally we construct the actual value of the associated type.
            let term = match goal.predicate.alias.kind(cx) {
                ty::AliasTermKind::ProjectionTy => {
                    cx.type_of(target_item_def_id).map_bound(|ty| ty.into())
                }
                ty::AliasTermKind::ProjectionConst => {
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

    fn consider_builtin_sized_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        panic!("`Sized` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_copy_clone_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        panic!("`Copy`/`Clone` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_pointer_like_candidate(
        _ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        panic!("`PointerLike` does not have an associated type: {:?}", goal);
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
        let output_is_sized_pred = tupled_inputs_and_output.map_bound(|(_, output)| {
            ty::TraitRef::new(cx, cx.require_lang_item(TraitSolverLangItem::Sized), [output])
        });

        let pred = tupled_inputs_and_output
            .map_bound(|(inputs, output)| ty::ProjectionPredicate {
                projection_term: ty::AliasTerm::new(cx, goal.predicate.def_id(), [
                    goal.predicate.self_ty(),
                    inputs,
                ]),
                term: output.into(),
            })
            .upcast(cx);

        // A built-in `Fn` impl only holds if the output is sized.
        // (FIXME: technically we only need to check this if the type is a fn ptr...)
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
        let output_is_sized_pred = tupled_inputs_and_output_and_coroutine.map_bound(
            |AsyncCallableRelevantTypes { output_coroutine_ty: output_ty, .. }| {
                ty::TraitRef::new(cx, cx.require_lang_item(TraitSolverLangItem::Sized), [output_ty])
            },
        );

        let pred = tupled_inputs_and_output_and_coroutine
            .map_bound(
                |AsyncCallableRelevantTypes {
                     tupled_inputs_ty,
                     output_coroutine_ty,
                     coroutine_return_ty,
                 }| {
                    let (projection_term, term) = if cx
                        .is_lang_item(goal.predicate.def_id(), TraitSolverLangItem::CallOnceFuture)
                    {
                        (
                            ty::AliasTerm::new(cx, goal.predicate.def_id(), [
                                goal.predicate.self_ty(),
                                tupled_inputs_ty,
                            ]),
                            output_coroutine_ty.into(),
                        )
                    } else if cx
                        .is_lang_item(goal.predicate.def_id(), TraitSolverLangItem::CallRefFuture)
                    {
                        (
                            ty::AliasTerm::new(cx, goal.predicate.def_id(), [
                                I::GenericArg::from(goal.predicate.self_ty()),
                                tupled_inputs_ty.into(),
                                env_region.into(),
                            ]),
                            output_coroutine_ty.into(),
                        )
                    } else if cx.is_lang_item(
                        goal.predicate.def_id(),
                        TraitSolverLangItem::AsyncFnOnceOutput,
                    ) {
                        (
                            ty::AliasTerm::new(cx, goal.predicate.def_id(), [
                                I::GenericArg::from(goal.predicate.self_ty()),
                                tupled_inputs_ty.into(),
                            ]),
                            coroutine_return_ty.into(),
                        )
                    } else {
                        panic!(
                            "no such associated type in `AsyncFn*`: {:?}",
                            goal.predicate.def_id()
                        )
                    };
                    ty::ProjectionPredicate { projection_term, term }
                },
            )
            .upcast(cx);

        // A built-in `AsyncFn` impl only holds if the output is sized.
        // (FIXME: technically we only need to check this if the type is a fn ptr...)
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
        let metadata_def_id = cx.require_lang_item(TraitSolverLangItem::Metadata);
        assert_eq!(metadata_def_id, goal.predicate.def_id());
        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
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
                | ty::Foreign(..)
                | ty::Dynamic(_, _, ty::DynStar) => Ty::new_unit(cx),

                ty::Error(e) => Ty::new_error(cx, e),

                ty::Str | ty::Slice(_) => Ty::new_usize(cx),

                ty::Dynamic(_, _, ty::Dyn) => {
                    let dyn_metadata = cx.require_lang_item(TraitSolverLangItem::DynMetadata);
                    cx.type_of(dyn_metadata)
                        .instantiate(cx, &[I::GenericArg::from(goal.predicate.self_ty())])
                }

                ty::Alias(_, _) | ty::Param(_) | ty::Placeholder(..) => {
                    // This is the "fallback impl" for type parameters, unnormalizable projections
                    // and opaque types: If the `self_ty` is `Sized`, then the metadata is `()`.
                    // FIXME(ptr_metadata): This impl overlaps with the other impls and shouldn't
                    // exist. Instead, `Pointee<Metadata = ()>` should be a supertrait of `Sized`.
                    let sized_predicate =
                        ty::TraitRef::new(cx, cx.require_lang_item(TraitSolverLangItem::Sized), [
                            I::GenericArg::from(goal.predicate.self_ty()),
                        ]);
                    // FIXME(-Znext-solver=coinductive): Should this be `GoalSource::ImplWhereBound`?
                    ecx.add_goal(GoalSource::Misc, goal.with(cx, sized_predicate));
                    Ty::new_unit(cx)
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

                ty::Infer(
                    ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_),
                )
                | ty::Bound(..) => panic!(
                    "unexpected self ty `{:?}` when normalizing `<T as Pointee>::Metadata`",
                    goal.predicate.self_ty()
                ),
            };

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
                cx.adt_def(cx.require_lang_item(TraitSolverLangItem::Poll)),
                cx.mk_args(&[Ty::new_adt(
                    cx,
                    cx.adt_def(cx.require_lang_item(TraitSolverLangItem::Option)),
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

        let term = if cx.is_lang_item(goal.predicate.def_id(), TraitSolverLangItem::CoroutineReturn)
        {
            coroutine.return_ty().into()
        } else if cx.is_lang_item(goal.predicate.def_id(), TraitSolverLangItem::CoroutineYield) {
            coroutine.yield_ty().into()
        } else {
            panic!("unexpected associated item `{:?}` for `{self_ty:?}`", goal.predicate.def_id())
        };

        Self::probe_and_consider_implied_clause(
            ecx,
            CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
            goal,
            ty::ProjectionPredicate {
                projection_term: ty::AliasTerm::new(ecx.cx(), goal.predicate.def_id(), [
                    self_ty,
                    coroutine.resume_ty(),
                ]),
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
            | ty::Dynamic(_, _, _)
            | ty::Tuple(_)
            | ty::Error(_) => self_ty.discriminant_ty(ecx.cx()),

            // We do not call `Ty::discriminant_ty` on alias, param, or placeholder
            // types, which return `<self_ty as DiscriminantKind>::Discriminant`
            // (or ICE in the case of placeholders). Projecting a type to itself
            // is never really productive.
            ty::Alias(_, _) | ty::Param(_) | ty::Placeholder(..) => {
                return Err(NoSolution);
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

    fn consider_builtin_async_destruct_candidate(
        ecx: &mut EvalCtxt<'_, D>,
        goal: Goal<I, Self>,
    ) -> Result<Candidate<I>, NoSolution> {
        let self_ty = goal.predicate.self_ty();
        let async_destructor_ty = match self_ty.kind() {
            ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..)
            | ty::Array(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Infer(ty::IntVar(..) | ty::FloatVar(..))
            | ty::Never
            | ty::Adt(_, _)
            | ty::Str
            | ty::Slice(_)
            | ty::Tuple(_)
            | ty::Error(_) => self_ty.async_destructor_ty(ecx.cx()),

            // We do not call `Ty::async_destructor_ty` on alias, param, or placeholder
            // types, which return `<self_ty as AsyncDestruct>::AsyncDestructor`
            // (or ICE in the case of placeholders). Projecting a type to itself
            // is never really productive.
            ty::Alias(_, _) | ty::Param(_) | ty::Placeholder(..) => {
                return Err(NoSolution);
            }

            ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_))
            | ty::Foreign(..)
            | ty::Bound(..) => panic!(
                "unexpected self ty `{:?}` when normalizing `<T as AsyncDestruct>::AsyncDestructor`",
                goal.predicate.self_ty()
            ),

            ty::Pat(..) | ty::Dynamic(..) | ty::Coroutine(..) | ty::CoroutineWitness(..) => panic!(
                "`consider_builtin_async_destruct_candidate` is not yet implemented for type: {self_ty:?}"
            ),
        };

        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
            ecx.eq(goal.param_env, goal.predicate.term, async_destructor_ty.into())
                .expect("expected goal term to be fully unconstrained");
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
}

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    fn translate_args(
        &mut self,
        goal: Goal<I, ty::NormalizesTo<I>>,
        impl_def_id: I::DefId,
        impl_args: I::GenericArgs,
        impl_trait_ref: rustc_type_ir::TraitRef<I>,
        target_container_def_id: I::DefId,
    ) -> Result<I::GenericArgs, NoSolution> {
        let cx = self.cx();
        Ok(if target_container_def_id == impl_trait_ref.def_id {
            // Default value from the trait definition. No need to rebase.
            goal.predicate.alias.args
        } else if target_container_def_id == impl_def_id {
            // Same impl, no need to fully translate, just a rebase from
            // the trait is sufficient.
            goal.predicate.alias.args.rebase_onto(cx, impl_trait_ref.def_id, impl_args)
        } else {
            let target_args = self.fresh_args_for_item(target_container_def_id);
            let target_trait_ref =
                cx.impl_trait_ref(target_container_def_id).instantiate(cx, target_args);
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
            goal.predicate.alias.args.rebase_onto(cx, impl_trait_ref.def_id, target_args)
        })
    }
}
