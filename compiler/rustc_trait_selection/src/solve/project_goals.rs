use crate::traits::{specialization_graph, translate_substs};

use super::assembly;
use super::trait_goals::structural_traits;
use super::EvalCtxt;
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::LangItem;
use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::specialization_graph::LeafDef;
use rustc_infer::traits::Reveal;
use rustc_middle::traits::solve::{CanonicalResponse, Certainty, Goal, QueryResult};
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams};
use rustc_middle::ty::ProjectionPredicate;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{ToPredicate, TypeVisitableExt};
use rustc_span::{sym, DUMMY_SP};
use std::iter;

impl<'tcx> EvalCtxt<'_, 'tcx> {
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn compute_projection_goal(
        &mut self,
        goal: Goal<'tcx, ProjectionPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        // To only compute normalization once for each projection we only
        // normalize if the expected term is an unconstrained inference variable.
        //
        // E.g. for `<T as Trait>::Assoc == u32` we recursively compute the goal
        // `exists<U> <T as Trait>::Assoc == U` and then take the resulting type for
        // `U` and equate it with `u32`. This means that we don't need a separate
        // projection cache in the solver.
        if self.term_is_fully_unconstrained(goal) {
            let candidates = self.assemble_and_evaluate_candidates(goal);
            self.merge_candidates(candidates)
        } else {
            let predicate = goal.predicate;
            let unconstrained_rhs = self.next_term_infer_of_kind(predicate.term);
            let unconstrained_predicate = ProjectionPredicate {
                projection_ty: goal.predicate.projection_ty,
                term: unconstrained_rhs,
            };

            self.set_normalizes_to_hack_goal(goal.with(self.tcx(), unconstrained_predicate));
            self.try_evaluate_added_goals()?;
            self.eq(goal.param_env, unconstrained_rhs, predicate.term)?;
            self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
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

    fn consider_implied_clause(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        assumption: ty::Predicate<'tcx>,
        requirements: impl IntoIterator<Item = Goal<'tcx, ty::Predicate<'tcx>>>,
    ) -> QueryResult<'tcx> {
        if let Some(poly_projection_pred) = assumption.to_opt_poly_projection_pred()
            && poly_projection_pred.projection_def_id() == goal.predicate.def_id()
        {
            ecx.probe(|ecx| {
                let assumption_projection_pred =
                    ecx.instantiate_binder_with_infer(poly_projection_pred);
                ecx.eq(
                    goal.param_env,
                    goal.predicate.projection_ty,
                    assumption_projection_pred.projection_ty,
                )?;
                ecx.eq(goal.param_env, goal.predicate.term, assumption_projection_pred.term)?;
                ecx.add_goals(requirements);
                ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            })
        } else {
            Err(NoSolution)
        }
    }

    fn consider_object_bound_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        assumption: ty::Predicate<'tcx>,
    ) -> QueryResult<'tcx> {
        if let Some(poly_projection_pred) = assumption.to_opt_poly_projection_pred()
            && poly_projection_pred.projection_def_id() == goal.predicate.def_id()
        {
            ecx.probe(|ecx| {
                let tcx = ecx.tcx();

                let assumption_projection_pred =
                    ecx.instantiate_binder_with_infer(poly_projection_pred);
                ecx.eq(
                    goal.param_env,
                    goal.predicate.projection_ty,
                    assumption_projection_pred.projection_ty,
                )?;

                let ty::Dynamic(bounds, _, _) = *goal.predicate.self_ty().kind() else {
                    bug!("expected object type in `consider_object_bound_candidate`");
                };
                ecx.add_goals(
                    structural_traits::predicates_for_object_candidate(
                        &ecx,
                        goal.param_env,
                        goal.predicate.projection_ty.trait_ref(tcx),
                        bounds,
                    )
                    .into_iter()
                    .map(|pred| goal.with(tcx, pred)),
                );
                ecx.eq(goal.param_env, goal.predicate.term, assumption_projection_pred.term)?;
                ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            })
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
        if iter::zip(goal_trait_ref.substs, impl_trait_ref.skip_binder().substs)
            .any(|(goal, imp)| !drcx.generic_args_may_unify(goal, imp))
        {
            return Err(NoSolution);
        }

        ecx.probe(|ecx| {
            let impl_substs = ecx.fresh_substs_for_item(impl_def_id);
            let impl_trait_ref = impl_trait_ref.subst(tcx, impl_substs);

            ecx.eq(goal.param_env, goal_trait_ref, impl_trait_ref)?;

            let where_clause_bounds = tcx
                .predicates_of(impl_def_id)
                .instantiate(tcx, impl_substs)
                .predicates
                .into_iter()
                .map(|pred| goal.with(tcx, pred));
            ecx.add_goals(where_clause_bounds);

            // In case the associated item is hidden due to specialization, we have to
            // return ambiguity this would otherwise be incomplete, resulting in
            // unsoundness during coherence (#105782).
            let Some(assoc_def) = fetch_eligible_assoc_item_def(
                ecx.infcx,
                goal.param_env,
                goal_trait_ref,
                goal.predicate.def_id(),
                impl_def_id
            )? else {
                return ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes);
            };

            if !assoc_def.item.defaultness(tcx).has_value() {
                tcx.sess.delay_span_bug(
                    tcx.def_span(assoc_def.item.def_id),
                    "missing value for assoc item in impl",
                );
            }

            // Getting the right substitutions here is complex, e.g. given:
            // - a goal `<Vec<u32> as Trait<i32>>::Assoc<u64>`
            // - the applicable impl `impl<T> Trait<i32> for Vec<T>`
            // - and the impl which defines `Assoc` being `impl<T, U> Trait<U> for Vec<T>`
            //
            // We first rebase the goal substs onto the impl, going from `[Vec<u32>, i32, u64]`
            // to `[u32, u64]`.
            //
            // And then map these substs to the substs of the defining impl of `Assoc`, going
            // from `[u32, u64]` to `[u32, i32, u64]`.
            let impl_substs_with_gat = goal.predicate.projection_ty.substs.rebase_onto(
                tcx,
                goal_trait_ref.def_id,
                impl_substs,
            );
            let substs = translate_substs(
                ecx.infcx,
                goal.param_env,
                impl_def_id,
                impl_substs_with_gat,
                assoc_def.defining_node,
            );

            // Finally we construct the actual value of the associated type.
            let is_const = matches!(tcx.def_kind(assoc_def.item.def_id), DefKind::AssocConst);
            let ty = tcx.type_of(assoc_def.item.def_id);
            let term: ty::EarlyBinder<ty::Term<'tcx>> = if is_const {
                let identity_substs =
                    ty::InternalSubsts::identity_for_item(tcx, assoc_def.item.def_id);
                let did = ty::WithOptConstParam::unknown(assoc_def.item.def_id);
                let kind =
                    ty::ConstKind::Unevaluated(ty::UnevaluatedConst::new(did, identity_substs));
                ty.map_bound(|ty| tcx.mk_const(kind, ty).into())
            } else {
                ty.map_bound(|ty| ty.into())
            };

            ecx.eq(goal.param_env, goal.predicate.term, term.subst(tcx, substs))?;
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_auto_trait_candidate(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        bug!("auto traits do not have associated types: {:?}", goal);
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
        let output_is_sized_pred = tupled_inputs_and_output
            .map_bound(|(_, output)| tcx.at(DUMMY_SP).mk_trait_ref(LangItem::Sized, [output]));

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
        ecx.probe(|ecx| {
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

                ty::Error(e) => tcx.ty_error(*e),

                ty::Str | ty::Slice(_) => tcx.types.usize,

                ty::Dynamic(_, _, _) => {
                    let dyn_metadata = tcx.require_lang_item(LangItem::DynMetadata, None);
                    tcx.type_of(dyn_metadata)
                        .subst(tcx, &[ty::GenericArg::from(goal.predicate.self_ty())])
                }

                ty::Alias(_, _) | ty::Param(_) | ty::Placeholder(..) => {
                    // FIXME(ptr_metadata): It would also be possible to return a `Ok(Ambig)` with no constraints.
                    let sized_predicate = ty::Binder::dummy(tcx.at(DUMMY_SP).mk_trait_ref(
                        LangItem::Sized,
                        [ty::GenericArg::from(goal.predicate.self_ty())],
                    ));

                    ecx.add_goal(goal.with(tcx, sized_predicate));
                    ecx.eq(goal.param_env, goal.predicate.term, tcx.types.unit.into())?;
                    return ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes);
                }

                ty::Adt(def, substs) if def.is_struct() => {
                    match def.non_enum_variant().fields.last() {
                        None => tcx.types.unit,
                        Some(field_def) => {
                            let self_ty = field_def.ty(tcx, substs);
                            ecx.add_goal(goal.with(
                                tcx,
                                ty::Binder::dummy(goal.predicate.with_self_ty(tcx, self_ty)),
                            ));
                            return ecx
                                .evaluate_added_goals_and_make_canonical_response(Certainty::Yes);
                        }
                    }
                }
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

            ecx.eq(goal.param_env, goal.predicate.term, metadata_ty.into())?;
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_builtin_future_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        let self_ty = goal.predicate.self_ty();
        let ty::Generator(def_id, substs, _) = *self_ty.kind() else {
            return Err(NoSolution);
        };

        // Generators are not futures unless they come from `async` desugaring
        let tcx = ecx.tcx();
        if !tcx.generator_is_async(def_id) {
            return Err(NoSolution);
        }

        let term = substs.as_generator().return_ty().into();

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
        let ty::Generator(def_id, substs, _) = *self_ty.kind() else {
            return Err(NoSolution);
        };

        // `async`-desugared generators do not implement the generator trait
        let tcx = ecx.tcx();
        if tcx.generator_is_async(def_id) {
            return Err(NoSolution);
        }

        let generator = substs.as_generator();

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

    fn consider_builtin_unsize_candidate(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        bug!("`Unsize` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_dyn_upcast_candidates(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> Vec<CanonicalResponse<'tcx>> {
        bug!("`Unsize` does not have an associated type: {:?}", goal);
    }

    fn consider_builtin_discriminant_kind_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        let discriminant = goal.predicate.self_ty().discriminant_ty(ecx.tcx());
        ecx.probe(|ecx| {
            ecx.eq(goal.param_env, goal.predicate.term, discriminant.into())?;
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_builtin_destruct_candidate(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        bug!("`Destruct` does not have an associated type: {:?}", goal);
    }
}

/// This behavior is also implemented in `rustc_ty_utils` and in the old `project` code.
///
/// FIXME: We should merge these 3 implementations as it's likely that they otherwise
/// diverge.
#[instrument(level = "debug", skip(infcx, param_env), ret)]
fn fetch_eligible_assoc_item_def<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    goal_trait_ref: ty::TraitRef<'tcx>,
    trait_assoc_def_id: DefId,
    impl_def_id: DefId,
) -> Result<Option<LeafDef>, NoSolution> {
    let node_item = specialization_graph::assoc_def(infcx.tcx, impl_def_id, trait_assoc_def_id)
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
            let poly_trait_ref = infcx.resolve_vars_if_possible(goal_trait_ref);
            !poly_trait_ref.still_further_specializable()
        } else {
            debug!(?node_item.item.def_id, "not eligible due to default");
            false
        }
    };

    if eligible { Ok(Some(node_item)) } else { Ok(None) }
}
