//! Dealing with trait goals, i.e. `T: Trait<'a, U>`.

use crate::traits::supertrait_def_ids;

use super::assembly::structural_traits::AsyncCallableRelevantTypes;
use super::assembly::{self, structural_traits, Candidate};
use super::{EvalCtxt, GoalSource, SolverMode};
use rustc_data_structures::fx::FxIndexSet;
use rustc_hir::def_id::DefId;
use rustc_hir::{LangItem, Movability};
use rustc_infer::traits::query::NoSolution;
use rustc_middle::traits::solve::inspect::ProbeKind;
use rustc_middle::traits::solve::{
    CandidateSource, CanonicalResponse, Certainty, Goal, QueryResult,
};
use rustc_middle::traits::{BuiltinImplSource, Reveal};
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams, TreatProjections};
use rustc_middle::ty::{self, ToPredicate, Ty, TyCtxt};
use rustc_middle::ty::{TraitPredicate, TypeVisitableExt};
use rustc_span::{ErrorGuaranteed, DUMMY_SP};

impl<'tcx> assembly::GoalKind<'tcx> for TraitPredicate<'tcx> {
    fn self_ty(self) -> Ty<'tcx> {
        self.self_ty()
    }

    fn trait_ref(self, _: TyCtxt<'tcx>) -> ty::TraitRef<'tcx> {
        self.trait_ref
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
    ) -> Result<Candidate<'tcx>, NoSolution> {
        let tcx = ecx.tcx();

        let impl_trait_header = tcx.impl_trait_header(impl_def_id).unwrap();
        let drcx = DeepRejectCtxt { treat_obligation_params: TreatParams::ForLookup };
        if !drcx.args_may_unify(
            goal.predicate.trait_ref.args,
            impl_trait_header.skip_binder().trait_ref.args,
        ) {
            return Err(NoSolution);
        }

        // An upper bound of the certainty of this goal, used to lower the certainty
        // of reservation impl to ambiguous during coherence.
        let impl_polarity = impl_trait_header.skip_binder().polarity;
        let maximal_certainty = match impl_polarity {
            ty::ImplPolarity::Positive | ty::ImplPolarity::Negative => {
                match impl_polarity == goal.predicate.polarity {
                    true => Certainty::Yes,
                    false => return Err(NoSolution),
                }
            }
            ty::ImplPolarity::Reservation => match ecx.solver_mode() {
                SolverMode::Normal => return Err(NoSolution),
                SolverMode::Coherence => Certainty::AMBIGUOUS,
            },
        };

        ecx.probe_trait_candidate(CandidateSource::Impl(impl_def_id)).enter(|ecx| {
            let impl_args = ecx.fresh_args_for_item(impl_def_id);
            let impl_trait_ref = impl_trait_header.instantiate(tcx, impl_args).trait_ref;

            ecx.eq(goal.param_env, goal.predicate.trait_ref, impl_trait_ref)?;
            let where_clause_bounds = tcx
                .predicates_of(impl_def_id)
                .instantiate(tcx, impl_args)
                .predicates
                .into_iter()
                .map(|pred| goal.with(tcx, pred));
            ecx.add_goals(GoalSource::ImplWhereBound, where_clause_bounds);

            ecx.evaluate_added_goals_and_make_canonical_response(maximal_certainty)
        })
    }

    fn consider_error_guaranteed_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        _guar: ErrorGuaranteed,
    ) -> QueryResult<'tcx> {
        ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    fn probe_and_match_goal_against_assumption(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        assumption: ty::Clause<'tcx>,
        then: impl FnOnce(&mut EvalCtxt<'_, 'tcx>) -> QueryResult<'tcx>,
    ) -> QueryResult<'tcx> {
        if let Some(trait_clause) = assumption.as_trait_clause() {
            if trait_clause.def_id() == goal.predicate.def_id()
                && trait_clause.polarity() == goal.predicate.polarity
            {
                // FIXME: Constness
                ecx.probe_misc_candidate("assumption").enter(|ecx| {
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
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return Err(NoSolution);
        }

        if let Some(result) = ecx.disqualify_auto_trait_candidate_due_to_possible_impl(goal) {
            return result;
        }

        // Don't call `type_of` on a local TAIT that's in the defining scope,
        // since that may require calling `typeck` on the same item we're
        // currently type checking, which will result in a fatal cycle that
        // ideally we want to avoid, since we can make progress on this goal
        // via an alias bound or a locally-inferred hidden type instead.
        //
        // Also, don't call `type_of` on a TAIT in `Reveal::All` mode, since
        // we already normalize the self type in
        // `assemble_candidates_after_normalizing_self_ty`, and we'd
        // just be registering an identical candidate here.
        //
        // We always return `Err(NoSolution)` here in `SolverMode::Coherence`
        // since we'll always register an ambiguous candidate in
        // `assemble_candidates_after_normalizing_self_ty` due to normalizing
        // the TAIT.
        if let ty::Alias(ty::Opaque, opaque_ty) = goal.predicate.self_ty().kind() {
            if matches!(goal.param_env.reveal(), Reveal::All)
                || matches!(ecx.solver_mode(), SolverMode::Coherence)
                || opaque_ty
                    .def_id
                    .as_local()
                    .is_some_and(|def_id| ecx.can_define_opaque_ty(def_id))
            {
                return Err(NoSolution);
            }
        }

        ecx.probe_and_evaluate_goal_for_constituent_tys(
            goal,
            structural_traits::instantiate_constituent_tys_for_auto_trait,
        )
    }

    fn consider_trait_alias_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return Err(NoSolution);
        }

        let tcx = ecx.tcx();

        ecx.probe_misc_candidate("trait alias").enter(|ecx| {
            let nested_obligations = tcx
                .predicates_of(goal.predicate.def_id())
                .instantiate(tcx, goal.predicate.trait_ref.args);
            // FIXME(-Znext-solver=coinductive): Should this be `GoalSource::ImplWhereBound`?
            ecx.add_goals(
                GoalSource::Misc,
                nested_obligations.predicates.into_iter().map(|p| goal.with(tcx, p)),
            );
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_builtin_sized_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return Err(NoSolution);
        }

        ecx.probe_and_evaluate_goal_for_constituent_tys(
            goal,
            structural_traits::instantiate_constituent_tys_for_sized_trait,
        )
    }

    fn consider_builtin_copy_clone_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return Err(NoSolution);
        }

        ecx.probe_and_evaluate_goal_for_constituent_tys(
            goal,
            structural_traits::instantiate_constituent_tys_for_copy_clone_trait,
        )
    }

    fn consider_builtin_pointer_like_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return Err(NoSolution);
        }

        // The regions of a type don't affect the size of the type
        let tcx = ecx.tcx();
        // We should erase regions from both the param-env and type, since both
        // may have infer regions. Specifically, after canonicalizing and instantiating,
        // early bound regions turn into region vars in both the new and old solver.
        let key = tcx.erase_regions(goal.param_env.and(goal.predicate.self_ty()));
        // But if there are inference variables, we have to wait until it's resolved.
        if key.has_non_region_infer() {
            return ecx.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS);
        }

        if let Ok(layout) = tcx.layout_of(key)
            && layout.layout.is_pointer_like(&tcx.data_layout)
        {
            // FIXME: We could make this faster by making a no-constraints response
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        } else {
            Err(NoSolution)
        }
    }

    fn consider_builtin_fn_ptr_trait_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        let self_ty = goal.predicate.self_ty();
        match goal.predicate.polarity {
            ty::ImplPolarity::Positive => {
                if self_ty.is_fn_ptr() {
                    ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                } else {
                    Err(NoSolution)
                }
            }
            ty::ImplPolarity::Negative => {
                // If a type is rigid and not a fn ptr, then we know for certain
                // that it does *not* implement `FnPtr`.
                if !self_ty.is_fn_ptr() && self_ty.is_known_rigid() {
                    ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                } else {
                    Err(NoSolution)
                }
            }
            // FIXME: Goal polarity should be split from impl polarity
            ty::ImplPolarity::Reservation => {
                bug!("we never expect a `Reservation` polarity in a trait goal")
            }
        }
    }

    fn consider_builtin_fn_trait_candidates(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        goal_kind: ty::ClosureKind,
    ) -> QueryResult<'tcx> {
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return Err(NoSolution);
        }

        let tcx = ecx.tcx();
        let tupled_inputs_and_output =
            match structural_traits::extract_tupled_inputs_and_output_from_callable(
                tcx,
                goal.predicate.self_ty(),
                goal_kind,
            )? {
                Some(a) => a,
                None => {
                    return ecx
                        .evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS);
                }
            };
        let output_is_sized_pred = tupled_inputs_and_output.map_bound(|(_, output)| {
            ty::TraitRef::from_lang_item(tcx, LangItem::Sized, DUMMY_SP, [output])
        });

        let pred = tupled_inputs_and_output
            .map_bound(|(inputs, _)| {
                ty::TraitRef::new(tcx, goal.predicate.def_id(), [goal.predicate.self_ty(), inputs])
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
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return Err(NoSolution);
        }

        let tcx = ecx.tcx();
        let (tupled_inputs_and_output_and_coroutine, nested_preds) =
            structural_traits::extract_tupled_inputs_and_output_from_async_callable(
                tcx,
                goal.predicate.self_ty(),
                goal_kind,
                // This region doesn't matter because we're throwing away the coroutine type
                tcx.lifetimes.re_static,
            )?;
        let output_is_sized_pred = tupled_inputs_and_output_and_coroutine.map_bound(
            |AsyncCallableRelevantTypes { output_coroutine_ty, .. }| {
                ty::TraitRef::from_lang_item(tcx, LangItem::Sized, DUMMY_SP, [output_coroutine_ty])
            },
        );

        let pred = tupled_inputs_and_output_and_coroutine
            .map_bound(|AsyncCallableRelevantTypes { tupled_inputs_ty, .. }| {
                ty::TraitRef::new(
                    tcx,
                    goal.predicate.def_id(),
                    [goal.predicate.self_ty(), tupled_inputs_ty],
                )
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
        let [closure_fn_kind_ty, goal_kind_ty] = **goal.predicate.trait_ref.args else {
            bug!();
        };

        let Some(closure_kind) = closure_fn_kind_ty.expect_ty().to_opt_closure_kind() else {
            // We don't need to worry about the self type being an infer var.
            return Err(NoSolution);
        };
        let goal_kind = goal_kind_ty.expect_ty().to_opt_closure_kind().unwrap();
        if closure_kind.extends(goal_kind) {
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        } else {
            Err(NoSolution)
        }
    }

    fn consider_builtin_tuple_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return Err(NoSolution);
        }

        if let ty::Tuple(..) = goal.predicate.self_ty().kind() {
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        } else {
            Err(NoSolution)
        }
    }

    fn consider_builtin_pointee_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return Err(NoSolution);
        }

        ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    fn consider_builtin_future_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return Err(NoSolution);
        }

        let ty::Coroutine(def_id, _) = *goal.predicate.self_ty().kind() else {
            return Err(NoSolution);
        };

        // Coroutines are not futures unless they come from `async` desugaring
        let tcx = ecx.tcx();
        if !tcx.coroutine_is_async(def_id) {
            return Err(NoSolution);
        }

        // Async coroutine unconditionally implement `Future`
        // Technically, we need to check that the future output type is Sized,
        // but that's already proven by the coroutine being WF.
        ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    fn consider_builtin_iterator_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return Err(NoSolution);
        }

        let ty::Coroutine(def_id, _) = *goal.predicate.self_ty().kind() else {
            return Err(NoSolution);
        };

        // Coroutines are not iterators unless they come from `gen` desugaring
        let tcx = ecx.tcx();
        if !tcx.coroutine_is_gen(def_id) {
            return Err(NoSolution);
        }

        // Gen coroutines unconditionally implement `Iterator`
        // Technically, we need to check that the iterator output type is Sized,
        // but that's already proven by the coroutines being WF.
        ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    fn consider_builtin_async_iterator_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return Err(NoSolution);
        }

        let ty::Coroutine(def_id, _) = *goal.predicate.self_ty().kind() else {
            return Err(NoSolution);
        };

        // Coroutines are not iterators unless they come from `gen` desugaring
        let tcx = ecx.tcx();
        if !tcx.coroutine_is_async_gen(def_id) {
            return Err(NoSolution);
        }

        // Gen coroutines unconditionally implement `Iterator`
        // Technically, we need to check that the iterator output type is Sized,
        // but that's already proven by the coroutines being WF.
        ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    fn consider_builtin_coroutine_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return Err(NoSolution);
        }

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
        Self::consider_implied_clause(
            ecx,
            goal,
            ty::TraitRef::new(tcx, goal.predicate.def_id(), [self_ty, coroutine.resume_ty()])
                .to_predicate(tcx),
            // Technically, we need to check that the coroutine types are Sized,
            // but that's already proven by the coroutine being WF.
            [],
        )
    }

    fn consider_builtin_discriminant_kind_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return Err(NoSolution);
        }

        // `DiscriminantKind` is automatically implemented for every type.
        ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    fn consider_builtin_destruct_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return Err(NoSolution);
        }

        // FIXME(-Znext-solver): Implement this when we get const working in the new solver

        // `Destruct` is automatically implemented for every type in
        // non-const environments.
        ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    fn consider_builtin_transmute_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return Err(NoSolution);
        }

        // `rustc_transmute` does not have support for type or const params
        if goal.has_non_region_placeholders() {
            return Err(NoSolution);
        }

        // Erase regions because we compute layouts in `rustc_transmute`,
        // which will ICE for region vars.
        let args = ecx.tcx().erase_regions(goal.predicate.trait_ref.args);

        let Some(assume) =
            rustc_transmute::Assume::from_const(ecx.tcx(), goal.param_env, args.const_at(2))
        else {
            return Err(NoSolution);
        };

        let certainty = ecx.is_transmutable(
            rustc_transmute::Types { dst: args.type_at(0), src: args.type_at(1) },
            assume,
        )?;
        ecx.evaluate_added_goals_and_make_canonical_response(certainty)
    }

    /// ```ignore (builtin impl example)
    /// trait Trait {
    ///     fn foo(&self);
    /// }
    /// // results in the following builtin impl
    /// impl<'a, T: Trait + 'a> Unsize<dyn Trait + 'a> for T {}
    /// ```
    fn consider_structural_builtin_unsize_candidates(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> Vec<(CanonicalResponse<'tcx>, BuiltinImplSource)> {
        if goal.predicate.polarity != ty::ImplPolarity::Positive {
            return vec![];
        }

        let misc_candidate = |ecx: &mut EvalCtxt<'_, 'tcx>, certainty| {
            (
                ecx.evaluate_added_goals_and_make_canonical_response(certainty).unwrap(),
                BuiltinImplSource::Misc,
            )
        };

        let result_to_single = |result, source| match result {
            Ok(resp) => vec![(resp, source)],
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

            let goal = goal.with(ecx.tcx(), (a_ty, b_ty));
            match (a_ty.kind(), b_ty.kind()) {
                (ty::Infer(ty::TyVar(..)), ..) => bug!("unexpected infer {a_ty:?} {b_ty:?}"),
                (_, ty::Infer(ty::TyVar(..))) => vec![misc_candidate(ecx, Certainty::AMBIGUOUS)],

                // Trait upcasting, or `dyn Trait + Auto + 'a` -> `dyn Trait + 'b`.
                (
                    &ty::Dynamic(a_data, a_region, ty::Dyn),
                    &ty::Dynamic(b_data, b_region, ty::Dyn),
                ) => ecx.consider_builtin_dyn_upcast_candidates(
                    goal, a_data, a_region, b_data, b_region,
                ),

                // `T` -> `dyn Trait` unsizing.
                (_, &ty::Dynamic(b_region, b_data, ty::Dyn)) => result_to_single(
                    ecx.consider_builtin_unsize_to_dyn_candidate(goal, b_region, b_data),
                    BuiltinImplSource::Misc,
                ),

                // `[T; N]` -> `[T]` unsizing
                (&ty::Array(a_elem_ty, ..), &ty::Slice(b_elem_ty)) => result_to_single(
                    ecx.consider_builtin_array_unsize(goal, a_elem_ty, b_elem_ty),
                    BuiltinImplSource::Misc,
                ),

                // `Struct<T>` -> `Struct<U>` where `T: Unsize<U>`
                (&ty::Adt(a_def, a_args), &ty::Adt(b_def, b_args))
                    if a_def.is_struct() && a_def == b_def =>
                {
                    result_to_single(
                        ecx.consider_builtin_struct_unsize(goal, a_def, a_args, b_args),
                        BuiltinImplSource::Misc,
                    )
                }

                //  `(A, B, T)` -> `(A, B, U)` where `T: Unsize<U>`
                (&ty::Tuple(a_tys), &ty::Tuple(b_tys))
                    if a_tys.len() == b_tys.len() && !a_tys.is_empty() =>
                {
                    result_to_single(
                        ecx.consider_builtin_tuple_unsize(goal, a_tys, b_tys),
                        BuiltinImplSource::TupleUnsizing,
                    )
                }

                _ => vec![],
            }
        })
    }
}

impl<'tcx> EvalCtxt<'_, 'tcx> {
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
        goal: Goal<'tcx, (Ty<'tcx>, Ty<'tcx>)>,
        a_data: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
        a_region: ty::Region<'tcx>,
        b_data: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
        b_region: ty::Region<'tcx>,
    ) -> Vec<(CanonicalResponse<'tcx>, BuiltinImplSource)> {
        let tcx = self.tcx();
        let Goal { predicate: (a_ty, _b_ty), .. } = goal;

        let mut responses = vec![];
        // If the principal def ids match (or are both none), then we're not doing
        // trait upcasting. We're just removing auto traits (or shortening the lifetime).
        if a_data.principal_def_id() == b_data.principal_def_id() {
            if let Ok(resp) = self.consider_builtin_upcast_to_principal(
                goal,
                a_data,
                a_region,
                b_data,
                b_region,
                a_data.principal(),
            ) {
                responses.push((resp, BuiltinImplSource::Misc));
            }
        } else if let Some(a_principal) = a_data.principal() {
            self.walk_vtable(
                a_principal.with_self_ty(tcx, a_ty),
                |ecx, new_a_principal, _, vtable_vptr_slot| {
                    if let Ok(resp) = ecx.probe_misc_candidate("dyn upcast").enter(|ecx| {
                        ecx.consider_builtin_upcast_to_principal(
                            goal,
                            a_data,
                            a_region,
                            b_data,
                            b_region,
                            Some(new_a_principal.map_bound(|trait_ref| {
                                ty::ExistentialTraitRef::erase_self_ty(tcx, trait_ref)
                            })),
                        )
                    }) {
                        responses
                            .push((resp, BuiltinImplSource::TraitUpcasting { vtable_vptr_slot }));
                    }
                },
            );
        }

        responses
    }

    fn consider_builtin_unsize_to_dyn_candidate(
        &mut self,
        goal: Goal<'tcx, (Ty<'tcx>, Ty<'tcx>)>,
        b_data: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
        b_region: ty::Region<'tcx>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();
        let Goal { predicate: (a_ty, _), .. } = goal;

        // Can only unsize to an object-safe trait.
        if b_data.principal_def_id().is_some_and(|def_id| !tcx.check_is_object_safe(def_id)) {
            return Err(NoSolution);
        }

        // Check that the type implements all of the predicates of the trait object.
        // (i.e. the principal, all of the associated types match, and any auto traits)
        self.add_goals(
            GoalSource::ImplWhereBound,
            b_data.iter().map(|pred| goal.with(tcx, pred.with_self_ty(tcx, a_ty))),
        );

        // The type must be `Sized` to be unsized.
        if let Some(sized_def_id) = tcx.lang_items().sized_trait() {
            self.add_goal(
                GoalSource::ImplWhereBound,
                goal.with(tcx, ty::TraitRef::new(tcx, sized_def_id, [a_ty])),
            );
        } else {
            return Err(NoSolution);
        }

        // The type must outlive the lifetime of the `dyn` we're unsizing into.
        self.add_goal(GoalSource::Misc, goal.with(tcx, ty::OutlivesPredicate(a_ty, b_region)));
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    fn consider_builtin_upcast_to_principal(
        &mut self,
        goal: Goal<'tcx, (Ty<'tcx>, Ty<'tcx>)>,
        a_data: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
        a_region: ty::Region<'tcx>,
        b_data: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
        b_region: ty::Region<'tcx>,
        upcast_principal: Option<ty::PolyExistentialTraitRef<'tcx>>,
    ) -> QueryResult<'tcx> {
        let param_env = goal.param_env;

        // We may upcast to auto traits that are either explicitly listed in
        // the object type's bounds, or implied by the principal trait ref's
        // supertraits.
        let a_auto_traits: FxIndexSet<DefId> = a_data
            .auto_traits()
            .chain(a_data.principal_def_id().into_iter().flat_map(|principal_def_id| {
                supertrait_def_ids(self.tcx(), principal_def_id)
                    .filter(|def_id| self.tcx().trait_is_auto(*def_id))
            }))
            .collect();

        // More than one projection in a_ty's bounds may match the projection
        // in b_ty's bound. Use this to first determine *which* apply without
        // having any inference side-effects. We process obligations because
        // unification may initially succeed due to deferred projection equality.
        let projection_may_match =
            |ecx: &mut Self,
             source_projection: ty::PolyExistentialProjection<'tcx>,
             target_projection: ty::PolyExistentialProjection<'tcx>| {
                source_projection.item_def_id() == target_projection.item_def_id()
                    && ecx
                        .probe(|_| ProbeKind::UpcastProjectionCompatibility)
                        .enter(|ecx| -> Result<(), NoSolution> {
                            ecx.eq(param_env, source_projection, target_projection)?;
                            let _ = ecx.try_evaluate_added_goals()?;
                            Ok(())
                        })
                        .is_ok()
            };

        for bound in b_data {
            match bound.skip_binder() {
                // Check that a's supertrait (upcast_principal) is compatible
                // with the target (b_ty).
                ty::ExistentialPredicate::Trait(target_principal) => {
                    self.eq(param_env, upcast_principal.unwrap(), bound.rebind(target_principal))?;
                }
                // Check that b_ty's projection is satisfied by exactly one of
                // a_ty's projections. First, we look through the list to see if
                // any match. If not, error. Then, if *more* than one matches, we
                // return ambiguity. Otherwise, if exactly one matches, equate
                // it with b_ty's projection.
                ty::ExistentialPredicate::Projection(target_projection) => {
                    let target_projection = bound.rebind(target_projection);
                    let mut matching_projections =
                        a_data.projection_bounds().filter(|source_projection| {
                            projection_may_match(self, *source_projection, target_projection)
                        });
                    let Some(source_projection) = matching_projections.next() else {
                        return Err(NoSolution);
                    };
                    if matching_projections.next().is_some() {
                        return self.evaluate_added_goals_and_make_canonical_response(
                            Certainty::AMBIGUOUS,
                        );
                    }
                    self.eq(param_env, source_projection, target_projection)?;
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
        self.add_goal(
            GoalSource::ImplWhereBound,
            Goal::new(
                self.tcx(),
                param_env,
                ty::Binder::dummy(ty::OutlivesPredicate(a_region, b_region)),
            ),
        );

        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
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
        goal: Goal<'tcx, (Ty<'tcx>, Ty<'tcx>)>,
        a_elem_ty: Ty<'tcx>,
        b_elem_ty: Ty<'tcx>,
    ) -> QueryResult<'tcx> {
        self.eq(goal.param_env, a_elem_ty, b_elem_ty)?;
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
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
        goal: Goal<'tcx, (Ty<'tcx>, Ty<'tcx>)>,
        def: ty::AdtDef<'tcx>,
        a_args: ty::GenericArgsRef<'tcx>,
        b_args: ty::GenericArgsRef<'tcx>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();
        let Goal { predicate: (_a_ty, b_ty), .. } = goal;

        let unsizing_params = tcx.unsizing_params_for_adt(def.did());
        // We must be unsizing some type parameters. This also implies
        // that the struct has a tail field.
        if unsizing_params.is_empty() {
            return Err(NoSolution);
        }

        let tail_field = def.non_enum_variant().tail();
        let tail_field_ty = tcx.type_of(tail_field.did);

        let a_tail_ty = tail_field_ty.instantiate(tcx, a_args);
        let b_tail_ty = tail_field_ty.instantiate(tcx, b_args);

        // Instantiate just the unsizing params from B into A. The type after
        // this instantiation must be equal to B. This is so we don't unsize
        // unrelated type parameters.
        let new_a_args = tcx.mk_args_from_iter(
            a_args
                .iter()
                .enumerate()
                .map(|(i, a)| if unsizing_params.contains(i as u32) { b_args[i] } else { a }),
        );
        let unsized_a_ty = Ty::new_adt(tcx, def, new_a_args);

        // Finally, we require that `TailA: Unsize<TailB>` for the tail field
        // types.
        self.eq(goal.param_env, unsized_a_ty, b_ty)?;
        self.add_goal(
            GoalSource::ImplWhereBound,
            goal.with(
                tcx,
                ty::TraitRef::new(
                    tcx,
                    tcx.lang_items().unsize_trait().unwrap(),
                    [a_tail_ty, b_tail_ty],
                ),
            ),
        );
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
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
        goal: Goal<'tcx, (Ty<'tcx>, Ty<'tcx>)>,
        a_tys: &'tcx ty::List<Ty<'tcx>>,
        b_tys: &'tcx ty::List<Ty<'tcx>>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();
        let Goal { predicate: (_a_ty, b_ty), .. } = goal;

        let (&a_last_ty, a_rest_tys) = a_tys.split_last().unwrap();
        let &b_last_ty = b_tys.last().unwrap();

        // Instantiate just the tail field of B., and require that they're equal.
        let unsized_a_ty =
            Ty::new_tup_from_iter(tcx, a_rest_tys.iter().copied().chain([b_last_ty]));
        self.eq(goal.param_env, unsized_a_ty, b_ty)?;

        // Similar to ADTs, require that we can unsize the tail.
        self.add_goal(
            GoalSource::ImplWhereBound,
            goal.with(
                tcx,
                ty::TraitRef::new(
                    tcx,
                    tcx.lang_items().unsize_trait().unwrap(),
                    [a_last_ty, b_last_ty],
                ),
            ),
        );
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    // Return `Some` if there is an impl (built-in or user provided) that may
    // hold for the self type of the goal, which for coherence and soundness
    // purposes must disqualify the built-in auto impl assembled by considering
    // the type's constituent types.
    fn disqualify_auto_trait_candidate_due_to_possible_impl(
        &mut self,
        goal: Goal<'tcx, TraitPredicate<'tcx>>,
    ) -> Option<QueryResult<'tcx>> {
        let self_ty = goal.predicate.self_ty();
        match *self_ty.kind() {
            // Stall int and float vars until they are resolved to a concrete
            // numerical type. That's because the check for impls below treats
            // int vars as matching any impl. Even if we filtered such impls,
            // we probably don't want to treat an `impl !AutoTrait for i32` as
            // disqualifying the built-in auto impl for `i64: AutoTrait` either.
            ty::Infer(ty::IntVar(_) | ty::FloatVar(_)) => {
                Some(self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS))
            }

            // These types cannot be structurally decomposed into constituent
            // types, and therefore have no built-in auto impl.
            ty::Dynamic(..)
            | ty::Param(..)
            | ty::Foreign(..)
            | ty::Alias(ty::Projection | ty::Weak | ty::Inherent, ..)
            | ty::Placeholder(..) => Some(Err(NoSolution)),

            ty::Infer(_) | ty::Bound(_, _) => bug!("unexpected type `{self_ty}`"),

            // Coroutines have one special built-in candidate, `Unpin`, which
            // takes precedence over the structural auto trait candidate being
            // assembled.
            ty::Coroutine(def_id, _)
                if Some(goal.predicate.def_id()) == self.tcx().lang_items().unpin_trait() =>
            {
                match self.tcx().coroutine_movability(def_id) {
                    Movability::Static => Some(Err(NoSolution)),
                    Movability::Movable => {
                        Some(self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes))
                    }
                }
            }

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
            | ty::Slice(_)
            | ty::RawPtr(_)
            | ty::Ref(_, _, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(_)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(_, _)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::Tuple(_)
            | ty::Adt(_, _)
            // FIXME: Handling opaques here is kinda sus. Especially because we
            // simplify them to SimplifiedType::Placeholder.
            | ty::Alias(ty::Opaque, _) => {
                let mut disqualifying_impl = None;
                self.tcx().for_each_relevant_impl_treating_projections(
                    goal.predicate.def_id(),
                    goal.predicate.self_ty(),
                    TreatProjections::NextSolverLookup,
                    |impl_def_id| {
                        disqualifying_impl = Some(impl_def_id);
                    },
                );
                if let Some(def_id) = disqualifying_impl {
                    debug!(?def_id, ?goal, "disqualified auto-trait implementation");
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
        goal: Goal<'tcx, TraitPredicate<'tcx>>,
        constituent_tys: impl Fn(
            &EvalCtxt<'_, 'tcx>,
            Ty<'tcx>,
        ) -> Result<Vec<ty::Binder<'tcx, Ty<'tcx>>>, NoSolution>,
    ) -> QueryResult<'tcx> {
        self.probe_misc_candidate("constituent tys").enter(|ecx| {
            ecx.add_goals(
                GoalSource::ImplWhereBound,
                constituent_tys(ecx, goal.predicate.self_ty())?
                    .into_iter()
                    .map(|ty| {
                        ecx.enter_forall(ty, |ty| {
                            goal.with(ecx.tcx(), goal.predicate.with_self_ty(ecx.tcx(), ty))
                        })
                    })
                    .collect::<Vec<_>>(),
            );
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    #[instrument(level = "debug", skip(self))]
    pub(super) fn compute_trait_goal(
        &mut self,
        goal: Goal<'tcx, TraitPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        let candidates = self.assemble_and_evaluate_candidates(goal);
        self.merge_candidates(candidates)
    }
}
