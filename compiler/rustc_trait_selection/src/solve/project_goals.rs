use crate::traits::specialization_graph;

use super::assembly::{self, structural_traits};
use super::EvalCtxt;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::LangItem;
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::specialization_graph::LeafDef;
use rustc_infer::traits::Reveal;
use rustc_middle::traits::solve::inspect::CandidateKind;
use rustc_middle::traits::solve::{CanonicalResponse, Certainty, Goal, QueryResult};
use rustc_middle::traits::BuiltinImplSource;
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams};
use rustc_middle::ty::ProjectionPredicate;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{ToPredicate, TypeVisitableExt};
use rustc_span::{sym, ErrorGuaranteed, DUMMY_SP};

impl<'tcx> EvalCtxt<'_, 'tcx> {
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn compute_projection_goal(
        &mut self,
        goal: Goal<'tcx, ProjectionPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        let def_id = goal.predicate.def_id();
        match self.tcx().def_kind(def_id) {
            DefKind::AssocTy | DefKind::AssocConst => {
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
                    match self.tcx().associated_item(def_id).container {
                        ty::AssocItemContainer::TraitContainer => {
                            let candidates = self.assemble_and_evaluate_candidates(goal);
                            self.merge_candidates(candidates)
                        }
                        ty::AssocItemContainer::ImplContainer => {
                            self.normalize_inherent_associated_type(goal)
                        }
                    }
                } else {
                    self.set_normalizes_to_hack_goal(goal);
                    self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                }
            }
            DefKind::AnonConst => self.normalize_anon_const(goal),
            DefKind::OpaqueTy => self.normalize_opaque_type(goal),
            DefKind::TyAlias => self.normalize_weak_type(goal),
            kind => bug!("unknown DefKind {} in projection goal: {goal:#?}", kind.descr(def_id)),
        }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn normalize_anon_const(
        &mut self,
        goal: Goal<'tcx, ty::ProjectionPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        if let Some(normalized_const) = self.try_const_eval_resolve(
            goal.param_env,
            ty::UnevaluatedConst::new(
                goal.predicate.projection_ty.def_id,
                goal.predicate.projection_ty.args,
            ),
            self.tcx()
                .type_of(goal.predicate.projection_ty.def_id)
                .no_bound_vars()
                .expect("const ty should not rely on other generics"),
        ) {
            self.eq(goal.param_env, normalized_const, goal.predicate.term.ct().unwrap())?;
            self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        } else {
            self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
        }
    }
}

impl<'tcx> assembly::GoalKind<'tcx> for ProjectionPredicate<'tcx> {
    fn self_ty(self) -> Ty<'tcx> {
        self.self_ty()
    }

    fn trait_ref(self, tcx: TyCtxt<'tcx>) -> ty::TraitRef<'tcx> {
        self.projection_ty.trait_ref(tcx)
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
                ecx.probe_candidate("assumption").enter(|ecx| {
                    let assumption_projection_pred =
                        ecx.instantiate_binder_with_infer(projection_pred);
                    ecx.eq(
                        goal.param_env,
                        goal.predicate.projection_ty,
                        assumption_projection_pred.projection_ty,
                    )?;
                    ecx.eq(goal.param_env, goal.predicate.term, assumption_projection_pred.term)
                        .expect("expected goal term to be fully unconstrained");

                    // Add GAT where clauses from the trait's definition
                    ecx.add_goals(
                        tcx.predicates_of(goal.predicate.def_id())
                            .instantiate_own(tcx, goal.predicate.projection_ty.args)
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
        goal: Goal<'tcx, ProjectionPredicate<'tcx>>,
        impl_def_id: DefId,
    ) -> QueryResult<'tcx> {
        let tcx = ecx.tcx();

        let goal_trait_ref = goal.predicate.projection_ty.trait_ref(tcx);
        let impl_trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
        let drcx = DeepRejectCtxt { treat_obligation_params: TreatParams::ForLookup };
        if !drcx.args_refs_may_unify(goal_trait_ref.args, impl_trait_ref.skip_binder().args) {
            return Err(NoSolution);
        }

        ecx.probe(|r| CandidateKind::Candidate { name: "impl".into(), result: *r }).enter(|ecx| {
            let impl_args = ecx.fresh_args_for_item(impl_def_id);
            let impl_trait_ref = impl_trait_ref.instantiate(tcx, impl_args);

            ecx.eq(goal.param_env, goal_trait_ref, impl_trait_ref)?;

            let where_clause_bounds = tcx
                .predicates_of(impl_def_id)
                .instantiate(tcx, impl_args)
                .predicates
                .into_iter()
                .map(|pred| goal.with(tcx, pred));
            ecx.add_goals(where_clause_bounds);

            // Add GAT where clauses from the trait's definition
            ecx.add_goals(
                tcx.predicates_of(goal.predicate.def_id())
                    .instantiate_own(tcx, goal.predicate.projection_ty.args)
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

            if !assoc_def.item.defaultness(tcx).has_value() {
                let guar = tcx.sess.delay_span_bug(
                    tcx.def_span(assoc_def.item.def_id),
                    "missing value for assoc item in impl",
                );
                let error_term = match assoc_def.item.kind {
                    ty::AssocKind::Const => ty::Const::new_error(
                        tcx,
                        guar,
                        tcx.type_of(goal.predicate.def_id())
                            .instantiate(tcx, goal.predicate.projection_ty.args),
                    )
                    .into(),
                    ty::AssocKind::Type => Ty::new_error(tcx, guar).into(),
                    ty::AssocKind::Fn => unreachable!(),
                };
                ecx.eq(goal.param_env, goal.predicate.term, error_term)
                    .expect("expected goal term to be fully unconstrained");
                return ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes);
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
            let impl_args_with_gat = goal.predicate.projection_ty.args.rebase_onto(
                tcx,
                goal_trait_ref.def_id,
                impl_args,
            );
            let args = ecx.translate_args(
                goal.param_env,
                impl_def_id,
                impl_args_with_gat,
                assoc_def.defining_node,
            );

            // Finally we construct the actual value of the associated type.
            let term = match assoc_def.item.kind {
                ty::AssocKind::Type => tcx.type_of(assoc_def.item.def_id).map_bound(|ty| ty.into()),
                ty::AssocKind::Const => bug!("associated const projection is not supported yet"),
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
        ecx.tcx().sess.delay_span_bug(
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
                projection_ty: tcx
                    .mk_alias_ty(goal.predicate.def_id(), [goal.predicate.self_ty(), inputs]),
                term: output.into(),
            })
            .to_predicate(tcx);
        // A built-in `Fn` impl only holds if the output is sized.
        // (FIXME: technically we only need to check this if the type is a fn ptr...)
        Self::consider_implied_clause(ecx, goal, pred, [goal.with(tcx, output_is_sized_pred)])
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
        ecx.probe_candidate("builtin pointee").enter(|ecx| {
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
                | ty::Infer(ty::IntVar(..) | ty::FloatVar(..))
                | ty::Generator(..)
                | ty::GeneratorWitness(..)
                | ty::GeneratorWitnessMIR(..)
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
                    // FIXME(ptr_metadata): It would also be possible to return a `Ok(Ambig)` with no constraints.
                    let sized_predicate = ty::TraitRef::from_lang_item(
                        tcx,
                        LangItem::Sized,
                        DUMMY_SP,
                        [ty::GenericArg::from(goal.predicate.self_ty())],
                    );
                    ecx.add_goal(goal.with(tcx, sized_predicate));
                    tcx.types.unit
                }

                ty::Adt(def, args) if def.is_struct() => match def.non_enum_variant().tail_opt() {
                    None => tcx.types.unit,
                    Some(field_def) => {
                        let self_ty = field_def.ty(tcx, args);
                        ecx.add_goal(goal.with(
                            tcx,
                            ty::Binder::dummy(goal.predicate.with_self_ty(tcx, self_ty)),
                        ));
                        return ecx
                            .evaluate_added_goals_and_make_canonical_response(Certainty::Yes);
                    }
                },
                ty::Adt(_, _) => tcx.types.unit,

                ty::Tuple(elements) => match elements.last() {
                    None => tcx.types.unit,
                    Some(&self_ty) => {
                        ecx.add_goal(goal.with(
                            tcx,
                            ty::Binder::dummy(goal.predicate.with_self_ty(tcx, self_ty)),
                        ));
                        return ecx
                            .evaluate_added_goals_and_make_canonical_response(Certainty::Yes);
                    }
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
        let ty::Generator(def_id, args, _) = *self_ty.kind() else {
            return Err(NoSolution);
        };

        // Generators are not futures unless they come from `async` desugaring
        let tcx = ecx.tcx();
        if !tcx.generator_is_async(def_id) {
            return Err(NoSolution);
        }

        let term = args.as_generator().return_ty().into();

        Self::consider_implied_clause(
            ecx,
            goal,
            ty::Binder::dummy(ty::ProjectionPredicate {
                projection_ty: ecx.tcx().mk_alias_ty(goal.predicate.def_id(), [self_ty]),
                term,
            })
            .to_predicate(tcx),
            // Technically, we need to check that the future type is Sized,
            // but that's already proven by the generator being WF.
            [],
        )
    }

    fn consider_builtin_generator_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        let self_ty = goal.predicate.self_ty();
        let ty::Generator(def_id, args, _) = *self_ty.kind() else {
            return Err(NoSolution);
        };

        // `async`-desugared generators do not implement the generator trait
        let tcx = ecx.tcx();
        if tcx.generator_is_async(def_id) {
            return Err(NoSolution);
        }

        let generator = args.as_generator();

        let name = tcx.associated_item(goal.predicate.def_id()).name;
        let term = if name == sym::Return {
            generator.return_ty().into()
        } else if name == sym::Yield {
            generator.yield_ty().into()
        } else {
            bug!("unexpected associated item `<{self_ty} as Generator>::{name}`")
        };

        Self::consider_implied_clause(
            ecx,
            goal,
            ty::Binder::dummy(ty::ProjectionPredicate {
                projection_ty: ecx
                    .tcx()
                    .mk_alias_ty(goal.predicate.def_id(), [self_ty, generator.resume_ty()]),
                term,
            })
            .to_predicate(tcx),
            // Technically, we need to check that the future type is Sized,
            // but that's already proven by the generator being WF.
            [],
        )
    }

    fn consider_builtin_unsize_and_upcast_candidates(
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
            | ty::Infer(ty::IntVar(..) | ty::FloatVar(..))
            | ty::Generator(..)
            | ty::GeneratorWitness(..)
            | ty::GeneratorWitnessMIR(..)
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

        ecx.probe_candidate("builtin discriminant kind").enter(|ecx| {
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
