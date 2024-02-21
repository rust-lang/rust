use crate::traits::{check_args_compatible, specialization_graph};

use super::assembly::{self, structural_traits, Candidate};
use super::{EvalCtxt, GoalSource};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::LangItem;
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::specialization_graph::LeafDef;
use rustc_infer::traits::Reveal;
use rustc_middle::traits::solve::{
    CandidateSource, CanonicalResponse, Certainty, Goal, QueryResult,
};
use rustc_middle::traits::BuiltinImplSource;
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams};
use rustc_middle::ty::NormalizesTo;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{ToPredicate, TypeVisitableExt};
use rustc_span::{sym, ErrorGuaranteed, DUMMY_SP};

mod anon_const;
mod inherent;
mod opaque_types;
mod weak_types;

impl<'tcx> EvalCtxt<'_, 'tcx> {
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn compute_normalizes_to_goal(
        &mut self,
        goal: Goal<'tcx, NormalizesTo<'tcx>>,
    ) -> QueryResult<'tcx> {
        let def_id = goal.predicate.def_id();
        match self.tcx().def_kind(def_id) {
            DefKind::AssocTy | DefKind::AssocConst => {
                match self.tcx().associated_item(def_id).container {
                    ty::AssocItemContainer::TraitContainer => {
                        // To only compute normalization once for each projection we only
                        // assemble normalization candidates if the expected term is an
                        // unconstrained inference variable.
                        //
                        // Why: For better cache hits, since if we have an unconstrained RHS then
                        // there are only as many cache keys as there are (canonicalized) alias
                        // types in each normalizes-to goal. This also weakens inference in a
                        // forwards-compatible way so we don't use the value of the RHS term to
                        // affect candidate assembly for projections.
                        //
                        // E.g. for `<T as Trait>::Assoc == u32` we recursively compute the goal
                        // `exists<U> <T as Trait>::Assoc == U` and then take the resulting type for
                        // `U` and equate it with `u32`. This means that we don't need a separate
                        // projection cache in the solver, since we're piggybacking off of regular
                        // goal caching.
                        if self.term_is_fully_unconstrained(goal) {
                            let candidates = self.assemble_and_evaluate_candidates(goal);
                            self.merge_candidates(candidates)
                        } else {
                            self.set_normalizes_to_hack_goal(goal);
                            self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                        }
                    }
                    ty::AssocItemContainer::ImplContainer => {
                        self.normalize_inherent_associated_type(goal)
                    }
                }
            }
            DefKind::AnonConst => self.normalize_anon_const(goal),
            DefKind::OpaqueTy => self.normalize_opaque_type(goal),
            DefKind::TyAlias => self.normalize_weak_type(goal),
            kind => bug!("unknown DefKind {} in projection goal: {goal:#?}", kind.descr(def_id)),
        }
    }
}

impl<'tcx> assembly::GoalKind<'tcx> for NormalizesTo<'tcx> {
    fn self_ty(self) -> Ty<'tcx> {
        self.self_ty()
    }

    fn trait_ref(self, tcx: TyCtxt<'tcx>) -> ty::TraitRef<'tcx> {
        self.alias.trait_ref(tcx)
    }

    fn with_self_ty(self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> Self {
        self.with_self_ty(tcx, self_ty)
    }

    fn trait_def_id(self, tcx: TyCtxt<'tcx>) -> DefId {
        self.trait_def_id(tcx)
    }

    fn probe_and_match_goal_against_assumption(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        assumption: ty::Clause<'tcx>,
        then: impl FnOnce(&mut EvalCtxt<'_, 'tcx>) -> QueryResult<'tcx>,
    ) -> QueryResult<'tcx> {
        if let Some(projection_pred) = assumption.as_projection_clause() {
            if projection_pred.projection_def_id() == goal.predicate.def_id() {
                let tcx = ecx.tcx();
                ecx.probe_misc_candidate("assumption").enter(|ecx| {
                    let assumption_projection_pred =
                        ecx.instantiate_binder_with_infer(projection_pred);
                    ecx.eq(
                        goal.param_env,
                        goal.predicate.alias,
                        assumption_projection_pred.projection_ty,
                    )?;
                    ecx.eq(goal.param_env, goal.predicate.term, assumption_projection_pred.term)
                        .expect("expected goal term to be fully unconstrained");

                    // Add GAT where clauses from the trait's definition
                    ecx.add_goals(
                        GoalSource::Misc,
                        tcx.predicates_of(goal.predicate.def_id())
                            .instantiate_own(tcx, goal.predicate.alias.args)
                            .map(|(pred, _)| goal.with(tcx, pred)),
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
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, NormalizesTo<'tcx>>,
        impl_def_id: DefId,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        let tcx = ecx.tcx();

        let goal_trait_ref = goal.predicate.alias.trait_ref(tcx);
        let impl_trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
        let drcx = DeepRejectCtxt { treat_obligation_params: TreatParams::ForLookup };
        if !drcx.args_may_unify(goal_trait_ref.args, impl_trait_ref.skip_binder().args) {
            return Err(NoSolution);
        }

        ecx.probe_trait_candidate(CandidateSource::Impl(impl_def_id)).enter(|ecx| {
            let impl_args = ecx.fresh_args_for_item(impl_def_id);
            let impl_trait_ref = impl_trait_ref.instantiate(tcx, impl_args);

            ecx.eq(goal.param_env, goal_trait_ref, impl_trait_ref)?;

            let where_clause_bounds = tcx
                .predicates_of(impl_def_id)
                .instantiate(tcx, impl_args)
                .predicates
                .into_iter()
                .map(|pred| goal.with(tcx, pred));
            ecx.add_goals(GoalSource::ImplWhereBound, where_clause_bounds);

            // Add GAT where clauses from the trait's definition
            ecx.add_goals(
                GoalSource::Misc,
                tcx.predicates_of(goal.predicate.def_id())
                    .instantiate_own(tcx, goal.predicate.alias.args)
                    .map(|(pred, _)| goal.with(tcx, pred)),
            );

            // In case the associated item is hidden due to specialization, we have to
            // return ambiguity this would otherwise be incomplete, resulting in
            // unsoundness during coherence (#105782).
            let Some(assoc_def) = fetch_eligible_assoc_item_def(
                ecx,
                goal.param_env,
                goal_trait_ref,
                goal.predicate.def_id(),
                impl_def_id,
            )?
            else {
                return ecx.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS);
            };

            let error_response = |ecx: &mut EvalCtxt<'_, 'tcx>, reason| {
                let guar = tcx.dcx().span_delayed_bug(tcx.def_span(assoc_def.item.def_id), reason);
                let error_term = match assoc_def.item.kind {
                    ty::AssocKind::Const => ty::Const::new_error(
                        tcx,
                        guar,
                        tcx.type_of(goal.predicate.def_id())
                            .instantiate(tcx, goal.predicate.alias.args),
                    )
                    .into(),
                    ty::AssocKind::Type => Ty::new_error(tcx, guar).into(),
                    // This makes no sense...
                    ty::AssocKind::Fn => span_bug!(
                        tcx.def_span(assoc_def.item.def_id),
                        "cannot project to an associated function"
                    ),
                };
                ecx.eq(goal.param_env, goal.predicate.term, error_term)
                    .expect("expected goal term to be fully unconstrained");
                ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            };

            if !assoc_def.item.defaultness(tcx).has_value() {
                return error_response(ecx, "missing value for assoc item in impl");
            }

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
            let impl_args_with_gat =
                goal.predicate.alias.args.rebase_onto(tcx, goal_trait_ref.def_id, impl_args);
            let args = ecx.translate_args(
                goal.param_env,
                impl_def_id,
                impl_args_with_gat,
                assoc_def.defining_node,
            );

            if !check_args_compatible(tcx, assoc_def.item, args) {
                return error_response(
                    ecx,
                    "associated item has mismatched generic item arguments",
                );
            }

            // Finally we construct the actual value of the associated type.
            let term = match assoc_def.item.kind {
                ty::AssocKind::Type => tcx.type_of(assoc_def.item.def_id).map_bound(|ty| ty.into()),
                ty::AssocKind::Const => {
                    if tcx.features().associated_const_equality {
                        bug!("associated const projection is not supported yet")
                    } else {
                        ty::EarlyBinder::bind(
                            ty::Const::new_error_with_message(
                                tcx,
                                tcx.type_of(assoc_def.item.def_id).instantiate_identity(),
                                DUMMY_SP,
                                "associated const projection is not supported yet",
                            )
                            .into(),
                        )
                    }
                }
                ty::AssocKind::Fn => unreachable!("we should never project to a fn"),
            };

            ecx.eq(goal.param_env, goal.predicate.term, term.instantiate(tcx, args))
                .expect("expected goal term to be fully unconstrained");
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    /// Fail to normalize if the predicate contains an error, alternatively, we could normalize to `ty::Error`
    /// and succeed. Can experiment with this to figure out what results in better error messages.
    fn consider_error_guaranteed_candidate(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        _guar: ErrorGuaranteed,
    ) -> QueryResult<'tcx> {
        Err(NoSolution)
    }

    fn consider_auto_trait_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        ecx.tcx().dcx().span_delayed_bug(
            ecx.tcx().def_span(goal.predicate.def_id()),
            "associated types not allowed on auto traits",
        );
        Err(NoSolution)
    }

    fn consider_trait_alias_candidate(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        bug!("trait aliases do not have associated types: {:?}", goal);
    }

    fn consider_builtin_sized_candidate(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        bug!("`Sized` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_copy_clone_candidate(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        bug!("`Copy`/`Clone` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_pointer_like_candidate(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        bug!("`PointerLike` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_fn_ptr_trait_candidate(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        bug!("`FnPtr` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_fn_trait_candidates(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        goal_kind: ty::ClosureKind,
    ) -> QueryResult<'tcx> {
        let tcx = ecx.tcx();
        let tupled_inputs_and_output =
            match structural_traits::extract_tupled_inputs_and_output_from_callable(
                tcx,
                goal.predicate.self_ty(),
                goal_kind,
            )? {
                Some(tupled_inputs_and_output) => tupled_inputs_and_output,
                None => {
                    return ecx
                        .evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS);
                }
            };
        let output_is_sized_pred = tupled_inputs_and_output.map_bound(|(_, output)| {
            ty::TraitRef::from_lang_item(tcx, LangItem::Sized, DUMMY_SP, [output])
        });

        let pred = tupled_inputs_and_output
            .map_bound(|(inputs, output)| ty::ProjectionPredicate {
                projection_ty: ty::AliasTy::new(
                    tcx,
                    goal.predicate.def_id(),
                    [goal.predicate.self_ty(), inputs],
                ),
                term: output.into(),
            })
            .to_predicate(tcx);

        // A built-in `Fn` impl only holds if the output is sized.
        // (FIXME: technically we only need to check this if the type is a fn ptr...)
        Self::consider_implied_clause(ecx, goal, pred, [goal.with(tcx, output_is_sized_pred)])
    }

    fn consider_builtin_async_fn_trait_candidates(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        goal_kind: ty::ClosureKind,
    ) -> QueryResult<'tcx> {
        let tcx = ecx.tcx();

        let env_region = match goal_kind {
            ty::ClosureKind::Fn | ty::ClosureKind::FnMut => goal.predicate.alias.args.region_at(2),
            // Doesn't matter what this region is
            ty::ClosureKind::FnOnce => tcx.lifetimes.re_static,
        };
        let (tupled_inputs_and_output_and_coroutine, nested_preds) =
            structural_traits::extract_tupled_inputs_and_output_from_async_callable(
                tcx,
                goal.predicate.self_ty(),
                goal_kind,
                env_region,
            )?;
        let output_is_sized_pred =
            tupled_inputs_and_output_and_coroutine.map_bound(|(_, output, _)| {
                ty::TraitRef::from_lang_item(tcx, LangItem::Sized, DUMMY_SP, [output])
            });

        let pred = tupled_inputs_and_output_and_coroutine
            .map_bound(|(inputs, output, coroutine)| {
                let (projection_ty, term) = match tcx.item_name(goal.predicate.def_id()) {
                    sym::CallOnceFuture => (
                        ty::AliasTy::new(
                            tcx,
                            goal.predicate.def_id(),
                            [goal.predicate.self_ty(), inputs],
                        ),
                        coroutine.into(),
                    ),
                    sym::CallMutFuture | sym::CallFuture => (
                        ty::AliasTy::new(
                            tcx,
                            goal.predicate.def_id(),
                            [
                                ty::GenericArg::from(goal.predicate.self_ty()),
                                inputs.into(),
                                env_region.into(),
                            ],
                        ),
                        coroutine.into(),
                    ),
                    sym::Output => (
                        ty::AliasTy::new(
                            tcx,
                            goal.predicate.def_id(),
                            [ty::GenericArg::from(goal.predicate.self_ty()), inputs.into()],
                        ),
                        output.into(),
                    ),
                    name => bug!("no such associated type: {name}"),
                };
                ty::ProjectionPredicate { projection_ty, term }
            })
            .to_predicate(tcx);

        // A built-in `AsyncFn` impl only holds if the output is sized.
        // (FIXME: technically we only need to check this if the type is a fn ptr...)
        Self::consider_implied_clause(
            ecx,
            goal,
            pred,
            [goal.with(tcx, output_is_sized_pred)]
                .into_iter()
                .chain(nested_preds.into_iter().map(|pred| goal.with(tcx, pred))),
        )
    }

    fn consider_builtin_async_fn_kind_helper_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        let [
            closure_fn_kind_ty,
            goal_kind_ty,
            borrow_region,
            tupled_inputs_ty,
            tupled_upvars_ty,
            coroutine_captures_by_ref_ty,
        ] = **goal.predicate.alias.args
        else {
            bug!();
        };

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
            ecx.tcx(),
            goal_kind,
            tupled_inputs_ty.expect_ty(),
            tupled_upvars_ty.expect_ty(),
            coroutine_captures_by_ref_ty.expect_ty(),
            borrow_region.expect_region(),
        );

        ecx.eq(goal.param_env, goal.predicate.term.ty().unwrap(), upvars_ty)?;
        ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    fn consider_builtin_tuple_candidate(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        bug!("`Tuple` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_pointee_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        let tcx = ecx.tcx();
        let metadata_def_id = tcx.require_lang_item(LangItem::Metadata, None);
        assert_eq!(metadata_def_id, goal.predicate.def_id());
        ecx.probe_misc_candidate("builtin pointee").enter(|ecx| {
            let metadata_ty = match goal.predicate.self_ty().kind() {
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
                | ty::Coroutine(..)
                | ty::CoroutineWitness(..)
                | ty::Never
                | ty::Foreign(..) => tcx.types.unit,

                ty::Error(e) => Ty::new_error(tcx, *e),

                ty::Str | ty::Slice(_) => tcx.types.usize,

                ty::Dynamic(_, _, _) => {
                    let dyn_metadata = tcx.require_lang_item(LangItem::DynMetadata, None);
                    tcx.type_of(dyn_metadata)
                        .instantiate(tcx, &[ty::GenericArg::from(goal.predicate.self_ty())])
                }

                ty::Alias(_, _) | ty::Param(_) | ty::Placeholder(..) => {
                    // This is the "fallback impl" for type parameters, unnormalizable projections
                    // and opaque types: If the `self_ty` is `Sized`, then the metadata is `()`.
                    // FIXME(ptr_metadata): This impl overlaps with the other impls and shouldn't
                    // exist. Instead, `Pointee<Metadata = ()>` should be a supertrait of `Sized`.
                    let sized_predicate = ty::TraitRef::from_lang_item(
                        tcx,
                        LangItem::Sized,
                        DUMMY_SP,
                        [ty::GenericArg::from(goal.predicate.self_ty())],
                    );
                    // FIXME(-Znext-solver=coinductive): Should this be `GoalSource::ImplWhereBound`?
                    ecx.add_goal(GoalSource::Misc, goal.with(tcx, sized_predicate));
                    tcx.types.unit
                }

                ty::Adt(def, args) if def.is_struct() => match def.non_enum_variant().tail_opt() {
                    None => tcx.types.unit,
                    Some(tail_def) => {
                        let tail_ty = tail_def.ty(tcx, args);
                        Ty::new_projection(tcx, metadata_def_id, [tail_ty])
                    }
                },
                ty::Adt(_, _) => tcx.types.unit,

                ty::Tuple(elements) => match elements.last() {
                    None => tcx.types.unit,
                    Some(&tail_ty) => Ty::new_projection(tcx, metadata_def_id, [tail_ty]),
                },

                ty::Infer(
                    ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_),
                )
                | ty::Bound(..) => bug!(
                    "unexpected self ty `{:?}` when normalizing `<T as Pointee>::Metadata`",
                    goal.predicate.self_ty()
                ),
            };

            ecx.eq(goal.param_env, goal.predicate.term, metadata_ty.into())
                .expect("expected goal term to be fully unconstrained");
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_builtin_future_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        let self_ty = goal.predicate.self_ty();
        let ty::Coroutine(def_id, args) = *self_ty.kind() else {
            return Err(NoSolution);
        };

        // Coroutines are not futures unless they come from `async` desugaring
        let tcx = ecx.tcx();
        if !tcx.coroutine_is_async(def_id) {
            return Err(NoSolution);
        }

        let term = args.as_coroutine().return_ty().into();

        Self::consider_implied_clause(
            ecx,
            goal,
            ty::ProjectionPredicate {
                projection_ty: ty::AliasTy::new(ecx.tcx(), goal.predicate.def_id(), [self_ty]),
                term,
            }
            .to_predicate(tcx),
            // Technically, we need to check that the future type is Sized,
            // but that's already proven by the coroutine being WF.
            [],
        )
    }

    fn consider_builtin_iterator_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        let self_ty = goal.predicate.self_ty();
        let ty::Coroutine(def_id, args) = *self_ty.kind() else {
            return Err(NoSolution);
        };

        // Coroutines are not Iterators unless they come from `gen` desugaring
        let tcx = ecx.tcx();
        if !tcx.coroutine_is_gen(def_id) {
            return Err(NoSolution);
        }

        let term = args.as_coroutine().yield_ty().into();

        Self::consider_implied_clause(
            ecx,
            goal,
            ty::ProjectionPredicate {
                projection_ty: ty::AliasTy::new(ecx.tcx(), goal.predicate.def_id(), [self_ty]),
                term,
            }
            .to_predicate(tcx),
            // Technically, we need to check that the iterator type is Sized,
            // but that's already proven by the generator being WF.
            [],
        )
    }

    fn consider_builtin_async_iterator_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        let self_ty = goal.predicate.self_ty();
        let ty::Coroutine(def_id, args) = *self_ty.kind() else {
            return Err(NoSolution);
        };

        // Coroutines are not AsyncIterators unless they come from `gen` desugaring
        let tcx = ecx.tcx();
        if !tcx.coroutine_is_async_gen(def_id) {
            return Err(NoSolution);
        }

        ecx.probe_misc_candidate("builtin AsyncIterator kind").enter(|ecx| {
            // Take `AsyncIterator<Item = I>` and turn it into the corresponding
            // coroutine yield ty `Poll<Option<I>>`.
            let expected_ty = Ty::new_adt(
                tcx,
                tcx.adt_def(tcx.require_lang_item(LangItem::Poll, None)),
                tcx.mk_args(&[Ty::new_adt(
                    tcx,
                    tcx.adt_def(tcx.require_lang_item(LangItem::Option, None)),
                    tcx.mk_args(&[goal.predicate.term.into()]),
                )
                .into()]),
            );
            let yield_ty = args.as_coroutine().yield_ty();
            ecx.eq(goal.param_env, expected_ty, yield_ty)?;
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_builtin_coroutine_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        let self_ty = goal.predicate.self_ty();
        let ty::Coroutine(def_id, args) = *self_ty.kind() else {
            return Err(NoSolution);
        };

        // `async`-desugared coroutines do not implement the coroutine trait
        let tcx = ecx.tcx();
        if !tcx.is_general_coroutine(def_id) {
            return Err(NoSolution);
        }

        let coroutine = args.as_coroutine();

        let name = tcx.associated_item(goal.predicate.def_id()).name;
        let term = if name == sym::Return {
            coroutine.return_ty().into()
        } else if name == sym::Yield {
            coroutine.yield_ty().into()
        } else {
            bug!("unexpected associated item `<{self_ty} as Coroutine>::{name}`")
        };

        Self::consider_implied_clause(
            ecx,
            goal,
            ty::ProjectionPredicate {
                projection_ty: ty::AliasTy::new(
                    ecx.tcx(),
                    goal.predicate.def_id(),
                    [self_ty, coroutine.resume_ty()],
                ),
                term,
            }
            .to_predicate(tcx),
            // Technically, we need to check that the coroutine type is Sized,
            // but that's already proven by the coroutine being WF.
            [],
        )
    }

    fn consider_structural_builtin_unsize_candidates(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> Vec<(CanonicalResponse<'tcx>, BuiltinImplSource)> {
        bug!("`Unsize` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_discriminant_kind_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        let self_ty = goal.predicate.self_ty();
        let discriminant_ty = match *self_ty.kind() {
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
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::Foreign(..)
            | ty::Adt(_, _)
            | ty::Str
            | ty::Slice(_)
            | ty::Dynamic(_, _, _)
            | ty::Tuple(_)
            | ty::Error(_) => self_ty.discriminant_ty(ecx.tcx()),

            // We do not call `Ty::discriminant_ty` on alias, param, or placeholder
            // types, which return `<self_ty as DiscriminantKind>::Discriminant`
            // (or ICE in the case of placeholders). Projecting a type to itself
            // is never really productive.
            ty::Alias(_, _) | ty::Param(_) | ty::Placeholder(..) => {
                return Err(NoSolution);
            }

            ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_))
            | ty::Bound(..) => bug!(
                "unexpected self ty `{:?}` when normalizing `<T as DiscriminantKind>::Discriminant`",
                goal.predicate.self_ty()
            ),
        };

        ecx.probe_misc_candidate("builtin discriminant kind").enter(|ecx| {
            ecx.eq(goal.param_env, goal.predicate.term, discriminant_ty.into())
                .expect("expected goal term to be fully unconstrained");
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_builtin_destruct_candidate(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        bug!("`Destruct` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_transmute_candidate(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        bug!("`BikeshedIntrinsicFrom` does not have an associated type: {:?}", goal)
    }
}

/// This behavior is also implemented in `rustc_ty_utils` and in the old `project` code.
///
/// FIXME: We should merge these 3 implementations as it's likely that they otherwise
/// diverge.
#[instrument(level = "debug", skip(ecx, param_env), ret)]
fn fetch_eligible_assoc_item_def<'tcx>(
    ecx: &EvalCtxt<'_, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    goal_trait_ref: ty::TraitRef<'tcx>,
    trait_assoc_def_id: DefId,
    impl_def_id: DefId,
) -> Result<Option<LeafDef>, NoSolution> {
    let node_item = specialization_graph::assoc_def(ecx.tcx(), impl_def_id, trait_assoc_def_id)
        .map_err(|ErrorGuaranteed { .. }| NoSolution)?;

    let eligible = if node_item.is_final() {
        // Non-specializable items are always projectable.
        true
    } else {
        // Only reveal a specializable default if we're past type-checking
        // and the obligation is monomorphic, otherwise passes such as
        // transmute checking and polymorphic MIR optimizations could
        // get a result which isn't correct for all monomorphizations.
        if param_env.reveal() == Reveal::All {
            let poly_trait_ref = ecx.resolve_vars_if_possible(goal_trait_ref);
            !poly_trait_ref.still_further_specializable()
        } else {
            debug!(?node_item.item.def_id, "not eligible due to default");
            false
        }
    };

    if eligible { Ok(Some(node_item)) } else { Ok(None) }
}
