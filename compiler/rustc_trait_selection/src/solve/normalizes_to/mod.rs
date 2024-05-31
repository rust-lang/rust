use crate::traits::specialization_graph;

use super::assembly::structural_traits::AsyncCallableRelevantTypes;
use super::assembly::{self, structural_traits, Candidate};
use super::{EvalCtxt, GoalSource};
use rustc_hir::def_id::DefId;
use rustc_hir::LangItem;
use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::solve::inspect::ProbeKind;
use rustc_infer::traits::solve::MaybeCause;
use rustc_infer::traits::specialization_graph::LeafDef;
use rustc_infer::traits::Reveal;
use rustc_middle::traits::solve::{CandidateSource, Certainty, Goal, QueryResult};
use rustc_middle::traits::BuiltinImplSource;
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams};
use rustc_middle::ty::NormalizesTo;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{TypeVisitableExt, Upcast};
use rustc_middle::{bug, span_bug};
use rustc_span::{sym, ErrorGuaranteed, DUMMY_SP};

mod anon_const;
mod inherent;
mod opaque_types;
mod weak_types;

impl<'tcx> EvalCtxt<'_, InferCtxt<'tcx>> {
    #[instrument(level = "trace", skip(self), ret)]
    pub(super) fn compute_normalizes_to_goal(
        &mut self,
        goal: Goal<'tcx, NormalizesTo<'tcx>>,
    ) -> QueryResult<'tcx> {
        self.set_is_normalizes_to_goal();
        debug_assert!(self.term_is_fully_unconstrained(goal));
        let normalize_result = self
            .probe(|&result| ProbeKind::TryNormalizeNonRigid { result })
            .enter(|this| this.normalize_at_least_one_step(goal));

        match normalize_result {
            Ok(res) => Ok(res),
            Err(NoSolution) => {
                let Goal { param_env, predicate: NormalizesTo { alias, term } } = goal;
                self.relate_rigid_alias_non_alias(param_env, alias, ty::Variance::Invariant, term)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
        }
    }

    /// Normalize the given alias by at least one step. If the alias is rigid, this
    /// returns `NoSolution`.
    #[instrument(level = "trace", skip(self), ret)]
    fn normalize_at_least_one_step(
        &mut self,
        goal: Goal<'tcx, NormalizesTo<'tcx>>,
    ) -> QueryResult<'tcx> {
        match goal.predicate.alias.kind(self.interner()) {
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
        goal: Goal<'tcx, NormalizesTo<'tcx>>,
        term: ty::Term<'tcx>,
    ) {
        self.eq(goal.param_env, goal.predicate.term, term)
            .expect("expected goal term to be fully unconstrained");
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
        ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        source: CandidateSource<'tcx>,
        goal: Goal<'tcx, Self>,
        assumption: ty::Clause<'tcx>,
        then: impl FnOnce(&mut EvalCtxt<'_, InferCtxt<'tcx>>) -> QueryResult<'tcx>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        if let Some(projection_pred) = assumption.as_projection_clause() {
            if projection_pred.projection_def_id() == goal.predicate.def_id() {
                let tcx = ecx.interner();
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
        ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, NormalizesTo<'tcx>>,
        impl_def_id: DefId,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        let tcx = ecx.interner();

        let goal_trait_ref = goal.predicate.alias.trait_ref(tcx);
        let impl_trait_header = tcx.impl_trait_header(impl_def_id).unwrap();
        let drcx = DeepRejectCtxt { treat_obligation_params: TreatParams::ForLookup };
        if !drcx.args_may_unify(
            goal.predicate.trait_ref(tcx).args,
            impl_trait_header.trait_ref.skip_binder().args,
        ) {
            return Err(NoSolution);
        }

        // We have to ignore negative impls when projecting.
        let impl_polarity = impl_trait_header.polarity;
        match impl_polarity {
            ty::ImplPolarity::Negative => return Err(NoSolution),
            ty::ImplPolarity::Reservation => {
                unimplemented!("reservation impl for trait with assoc item: {:?}", goal)
            }
            ty::ImplPolarity::Positive => {}
        };

        ecx.probe_trait_candidate(CandidateSource::Impl(impl_def_id)).enter(|ecx| {
            let impl_args = ecx.fresh_args_for_item(impl_def_id);
            let impl_trait_ref = impl_trait_header.trait_ref.instantiate(tcx, impl_args);

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

            let error_response = |ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>, reason| {
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
                ecx.instantiate_normalizes_to_term(goal, error_term);
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

            if !tcx.check_args_compatible(assoc_def.item.def_id, args) {
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

            ecx.instantiate_normalizes_to_term(goal, term.instantiate(tcx, args));
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    /// Fail to normalize if the predicate contains an error, alternatively, we could normalize to `ty::Error`
    /// and succeed. Can experiment with this to figure out what results in better error messages.
    fn consider_error_guaranteed_candidate(
        _ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        _guar: ErrorGuaranteed,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        Err(NoSolution)
    }

    fn consider_auto_trait_candidate(
        ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        ecx.interner().dcx().span_delayed_bug(
            ecx.interner().def_span(goal.predicate.def_id()),
            "associated types not allowed on auto traits",
        );
        Err(NoSolution)
    }

    fn consider_trait_alias_candidate(
        _ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        bug!("trait aliases do not have associated types: {:?}", goal);
    }

    fn consider_builtin_sized_candidate(
        _ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        bug!("`Sized` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_copy_clone_candidate(
        _ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        bug!("`Copy`/`Clone` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_pointer_like_candidate(
        _ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        bug!("`PointerLike` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_fn_ptr_trait_candidate(
        _ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        bug!("`FnPtr` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_fn_trait_candidates(
        ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
        goal_kind: ty::ClosureKind,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        let tcx = ecx.interner();
        let tupled_inputs_and_output =
            match structural_traits::extract_tupled_inputs_and_output_from_callable(
                tcx,
                goal.predicate.self_ty(),
                goal_kind,
            )? {
                Some(tupled_inputs_and_output) => tupled_inputs_and_output,
                None => {
                    return ecx.forced_ambiguity(MaybeCause::Ambiguity);
                }
            };
        let output_is_sized_pred = tupled_inputs_and_output.map_bound(|(_, output)| {
            ty::TraitRef::new(tcx, tcx.require_lang_item(LangItem::Sized, None), [output])
        });

        let pred = tupled_inputs_and_output
            .map_bound(|(inputs, output)| ty::ProjectionPredicate {
                projection_term: ty::AliasTerm::new(
                    tcx,
                    goal.predicate.def_id(),
                    [goal.predicate.self_ty(), inputs],
                ),
                term: output.into(),
            })
            .upcast(tcx);

        // A built-in `Fn` impl only holds if the output is sized.
        // (FIXME: technically we only need to check this if the type is a fn ptr...)
        Self::probe_and_consider_implied_clause(
            ecx,
            CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
            goal,
            pred,
            [(GoalSource::ImplWhereBound, goal.with(tcx, output_is_sized_pred))],
        )
    }

    fn consider_builtin_async_fn_trait_candidates(
        ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
        goal_kind: ty::ClosureKind,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        let tcx = ecx.interner();

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
        let output_is_sized_pred = tupled_inputs_and_output_and_coroutine.map_bound(
            |AsyncCallableRelevantTypes { output_coroutine_ty: output_ty, .. }| {
                ty::TraitRef::new(tcx, tcx.require_lang_item(LangItem::Sized, None), [output_ty])
            },
        );

        let pred = tupled_inputs_and_output_and_coroutine
            .map_bound(
                |AsyncCallableRelevantTypes {
                     tupled_inputs_ty,
                     output_coroutine_ty,
                     coroutine_return_ty,
                 }| {
                    let lang_items = tcx.lang_items();
                    let (projection_term, term) = if Some(goal.predicate.def_id())
                        == lang_items.call_once_future()
                    {
                        (
                            ty::AliasTerm::new(
                                tcx,
                                goal.predicate.def_id(),
                                [goal.predicate.self_ty(), tupled_inputs_ty],
                            ),
                            output_coroutine_ty.into(),
                        )
                    } else if Some(goal.predicate.def_id()) == lang_items.call_ref_future() {
                        (
                            ty::AliasTerm::new(
                                tcx,
                                goal.predicate.def_id(),
                                [
                                    ty::GenericArg::from(goal.predicate.self_ty()),
                                    tupled_inputs_ty.into(),
                                    env_region.into(),
                                ],
                            ),
                            output_coroutine_ty.into(),
                        )
                    } else if Some(goal.predicate.def_id()) == lang_items.async_fn_once_output() {
                        (
                            ty::AliasTerm::new(
                                tcx,
                                goal.predicate.def_id(),
                                [
                                    ty::GenericArg::from(goal.predicate.self_ty()),
                                    tupled_inputs_ty.into(),
                                ],
                            ),
                            coroutine_return_ty.into(),
                        )
                    } else {
                        bug!("no such associated type in `AsyncFn*`: {:?}", goal.predicate.def_id())
                    };
                    ty::ProjectionPredicate { projection_term, term }
                },
            )
            .upcast(tcx);

        // A built-in `AsyncFn` impl only holds if the output is sized.
        // (FIXME: technically we only need to check this if the type is a fn ptr...)
        Self::probe_and_consider_implied_clause(
            ecx,
            CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
            goal,
            pred,
            [goal.with(tcx, output_is_sized_pred)]
                .into_iter()
                .chain(nested_preds.into_iter().map(|pred| goal.with(tcx, pred)))
                .map(|goal| (GoalSource::ImplWhereBound, goal)),
        )
    }

    fn consider_builtin_async_fn_kind_helper_candidate(
        ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
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
            ecx.interner(),
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
        _ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        bug!("`Tuple` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_pointee_candidate(
        ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        let tcx = ecx.interner();
        let metadata_def_id = tcx.require_lang_item(LangItem::Metadata, None);
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
                | ty::Dynamic(_, _, ty::DynStar) => tcx.types.unit,

                ty::Error(e) => Ty::new_error(tcx, *e),

                ty::Str | ty::Slice(_) => tcx.types.usize,

                ty::Dynamic(_, _, ty::Dyn) => {
                    let dyn_metadata = tcx.require_lang_item(LangItem::DynMetadata, None);
                    tcx.type_of(dyn_metadata)
                        .instantiate(tcx, &[ty::GenericArg::from(goal.predicate.self_ty())])
                }

                ty::Alias(_, _) | ty::Param(_) | ty::Placeholder(..) => {
                    // This is the "fallback impl" for type parameters, unnormalizable projections
                    // and opaque types: If the `self_ty` is `Sized`, then the metadata is `()`.
                    // FIXME(ptr_metadata): This impl overlaps with the other impls and shouldn't
                    // exist. Instead, `Pointee<Metadata = ()>` should be a supertrait of `Sized`.
                    let sized_predicate = ty::TraitRef::new(
                        tcx,
                        tcx.require_lang_item(LangItem::Sized, None),
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

            ecx.instantiate_normalizes_to_term(goal, metadata_ty.into());
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_builtin_future_candidate(
        ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        let self_ty = goal.predicate.self_ty();
        let ty::Coroutine(def_id, args) = *self_ty.kind() else {
            return Err(NoSolution);
        };

        // Coroutines are not futures unless they come from `async` desugaring
        let tcx = ecx.interner();
        if !tcx.coroutine_is_async(def_id) {
            return Err(NoSolution);
        }

        let term = args.as_coroutine().return_ty().into();

        Self::probe_and_consider_implied_clause(
            ecx,
            CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
            goal,
            ty::ProjectionPredicate {
                projection_term: ty::AliasTerm::new(
                    ecx.interner(),
                    goal.predicate.def_id(),
                    [self_ty],
                ),
                term,
            }
            .upcast(tcx),
            // Technically, we need to check that the future type is Sized,
            // but that's already proven by the coroutine being WF.
            [],
        )
    }

    fn consider_builtin_iterator_candidate(
        ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        let self_ty = goal.predicate.self_ty();
        let ty::Coroutine(def_id, args) = *self_ty.kind() else {
            return Err(NoSolution);
        };

        // Coroutines are not Iterators unless they come from `gen` desugaring
        let tcx = ecx.interner();
        if !tcx.coroutine_is_gen(def_id) {
            return Err(NoSolution);
        }

        let term = args.as_coroutine().yield_ty().into();

        Self::probe_and_consider_implied_clause(
            ecx,
            CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
            goal,
            ty::ProjectionPredicate {
                projection_term: ty::AliasTerm::new(
                    ecx.interner(),
                    goal.predicate.def_id(),
                    [self_ty],
                ),
                term,
            }
            .upcast(tcx),
            // Technically, we need to check that the iterator type is Sized,
            // but that's already proven by the generator being WF.
            [],
        )
    }

    fn consider_builtin_fused_iterator_candidate(
        _ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        bug!("`FusedIterator` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_async_iterator_candidate(
        ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        let self_ty = goal.predicate.self_ty();
        let ty::Coroutine(def_id, args) = *self_ty.kind() else {
            return Err(NoSolution);
        };

        // Coroutines are not AsyncIterators unless they come from `gen` desugaring
        let tcx = ecx.interner();
        if !tcx.coroutine_is_async_gen(def_id) {
            return Err(NoSolution);
        }

        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
            let expected_ty = ecx.next_ty_infer();
            // Take `AsyncIterator<Item = I>` and turn it into the corresponding
            // coroutine yield ty `Poll<Option<I>>`.
            let wrapped_expected_ty = Ty::new_adt(
                tcx,
                tcx.adt_def(tcx.require_lang_item(LangItem::Poll, None)),
                tcx.mk_args(&[Ty::new_adt(
                    tcx,
                    tcx.adt_def(tcx.require_lang_item(LangItem::Option, None)),
                    tcx.mk_args(&[expected_ty.into()]),
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
        ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        let self_ty = goal.predicate.self_ty();
        let ty::Coroutine(def_id, args) = *self_ty.kind() else {
            return Err(NoSolution);
        };

        // `async`-desugared coroutines do not implement the coroutine trait
        let tcx = ecx.interner();
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

        Self::probe_and_consider_implied_clause(
            ecx,
            CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
            goal,
            ty::ProjectionPredicate {
                projection_term: ty::AliasTerm::new(
                    ecx.interner(),
                    goal.predicate.def_id(),
                    [self_ty, coroutine.resume_ty()],
                ),
                term,
            }
            .upcast(tcx),
            // Technically, we need to check that the coroutine type is Sized,
            // but that's already proven by the coroutine being WF.
            [],
        )
    }

    fn consider_structural_builtin_unsize_candidates(
        _ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Vec<Candidate<'tcx>> {
        bug!("`Unsize` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_discriminant_kind_candidate(
        ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        let self_ty = goal.predicate.self_ty();
        let discriminant_ty = match *self_ty.kind() {
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
            | ty::Error(_) => self_ty.discriminant_ty(ecx.interner()),

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

        ecx.probe_builtin_trait_candidate(BuiltinImplSource::Misc).enter(|ecx| {
            ecx.instantiate_normalizes_to_term(goal, discriminant_ty.into());
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_builtin_async_destruct_candidate(
        ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        let self_ty = goal.predicate.self_ty();
        let async_destructor_ty = match *self_ty.kind() {
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
            | ty::Error(_) => self_ty.async_destructor_ty(ecx.interner()),

            // We do not call `Ty::async_destructor_ty` on alias, param, or placeholder
            // types, which return `<self_ty as AsyncDestruct>::AsyncDestructor`
            // (or ICE in the case of placeholders). Projecting a type to itself
            // is never really productive.
            ty::Alias(_, _) | ty::Param(_) | ty::Placeholder(..) => {
                return Err(NoSolution);
            }

            ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_))
            | ty::Foreign(..)
            | ty::Bound(..) => bug!(
                "unexpected self ty `{:?}` when normalizing `<T as AsyncDestruct>::AsyncDestructor`",
                goal.predicate.self_ty()
            ),

            ty::Pat(..) | ty::Dynamic(..) | ty::Coroutine(..) | ty::CoroutineWitness(..) => bug!(
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
        _ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        bug!("`Destruct` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_transmute_candidate(
        _ecx: &mut EvalCtxt<'_, InferCtxt<'tcx>>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Candidate<'tcx>, NoSolution> {
        bug!("`BikeshedIntrinsicFrom` does not have an associated type: {:?}", goal)
    }
}

/// This behavior is also implemented in `rustc_ty_utils` and in the old `project` code.
///
/// FIXME: We should merge these 3 implementations as it's likely that they otherwise
/// diverge.
#[instrument(level = "trace", skip(ecx, param_env), ret)]
fn fetch_eligible_assoc_item_def<'tcx>(
    ecx: &EvalCtxt<'_, InferCtxt<'tcx>>,
    param_env: ty::ParamEnv<'tcx>,
    goal_trait_ref: ty::TraitRef<'tcx>,
    trait_assoc_def_id: DefId,
    impl_def_id: DefId,
) -> Result<Option<LeafDef>, NoSolution> {
    let node_item =
        specialization_graph::assoc_def(ecx.interner(), impl_def_id, trait_assoc_def_id)
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
            trace!(?node_item.item.def_id, "not eligible due to default");
            false
        }
    };

    if eligible { Ok(Some(node_item)) } else { Ok(None) }
}
