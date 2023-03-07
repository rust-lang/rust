//! Dealing with trait goals, i.e. `T: Trait<'a, U>`.

use super::{assembly, EvalCtxt, SolverMode};
use rustc_hir::def_id::DefId;
use rustc_hir::LangItem;
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::util::supertraits;
use rustc_middle::traits::solve::{CanonicalResponse, Certainty, Goal, QueryResult};
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams, TreatProjections};
use rustc_middle::ty::{self, ToPredicate, Ty, TyCtxt};
use rustc_middle::ty::{TraitPredicate, TypeVisitableExt};
use rustc_span::DUMMY_SP;

pub mod structural_traits;

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
    ) -> QueryResult<'tcx> {
        let tcx = ecx.tcx();

        let impl_trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
        let drcx = DeepRejectCtxt { treat_obligation_params: TreatParams::ForLookup };
        if !drcx.substs_refs_may_unify(
            goal.predicate.trait_ref.substs,
            impl_trait_ref.skip_binder().substs,
        ) {
            return Err(NoSolution);
        }

        let impl_polarity = tcx.impl_polarity(impl_def_id);
        // An upper bound of the certainty of this goal, used to lower the certainty
        // of reservation impl to ambiguous during coherence.
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

        ecx.probe(|ecx| {
            let impl_substs = ecx.fresh_substs_for_item(impl_def_id);
            let impl_trait_ref = impl_trait_ref.subst(tcx, impl_substs);

            ecx.eq(goal.param_env, goal.predicate.trait_ref, impl_trait_ref)?;
            let where_clause_bounds = tcx
                .predicates_of(impl_def_id)
                .instantiate(tcx, impl_substs)
                .predicates
                .into_iter()
                .map(|pred| goal.with(tcx, pred));
            ecx.add_goals(where_clause_bounds);

            ecx.evaluate_added_goals_and_make_canonical_response(maximal_certainty)
        })
    }

    fn consider_implied_clause(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        assumption: ty::Predicate<'tcx>,
        requirements: impl IntoIterator<Item = Goal<'tcx, ty::Predicate<'tcx>>>,
    ) -> QueryResult<'tcx> {
        if let Some(poly_trait_pred) = assumption.to_opt_poly_trait_pred()
            && poly_trait_pred.def_id() == goal.predicate.def_id()
        {
            // FIXME: Constness and polarity
            ecx.probe(|ecx| {
                let assumption_trait_pred =
                    ecx.instantiate_binder_with_infer(poly_trait_pred);
                ecx.eq(
                    goal.param_env,
                    goal.predicate.trait_ref,
                    assumption_trait_pred.trait_ref,
                )?;
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
        if let Some(poly_trait_pred) = assumption.to_opt_poly_trait_pred()
            && poly_trait_pred.def_id() == goal.predicate.def_id()
        {
            // FIXME: Constness and polarity
            ecx.probe(|ecx| {
                let assumption_trait_pred =
                    ecx.instantiate_binder_with_infer(poly_trait_pred);
                ecx.eq(
                    goal.param_env,
                    goal.predicate.trait_ref,
                    assumption_trait_pred.trait_ref,
                )?;

                let tcx = ecx.tcx();
                let ty::Dynamic(bounds, _, _) = *goal.predicate.self_ty().kind() else {
                    bug!("expected object type in `consider_object_bound_candidate`");
                };
                ecx.add_goals(
                    structural_traits::predicates_for_object_candidate(
                        &ecx,
                        goal.param_env,
                        goal.predicate.trait_ref,
                        bounds,
                    )
                    .into_iter()
                    .map(|pred| goal.with(tcx, pred)),
                );
                ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            })
        } else {
            Err(NoSolution)
        }
    }

    fn consider_auto_trait_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        // This differs from the current stable behavior and
        // fixes #84857. Due to breakage found via crater, we
        // currently instead lint patterns which can be used to
        // exploit this unsoundness on stable, see #93367 for
        // more details.
        //
        // Using `TreatProjections::NextSolverLookup` is fine here because
        // `instantiate_constituent_tys_for_auto_trait` returns nothing for
        // projection types anyways. So it doesn't really matter what we do
        // here, and this is faster.
        if let Some(def_id) = ecx.tcx().find_map_relevant_impl(
            goal.predicate.def_id(),
            goal.predicate.self_ty(),
            TreatProjections::NextSolverLookup,
            Some,
        ) {
            debug!(?def_id, ?goal, "disqualified auto-trait implementation");
            return Err(NoSolution);
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
        let tcx = ecx.tcx();

        ecx.probe(|ecx| {
            let nested_obligations = tcx
                .predicates_of(goal.predicate.def_id())
                .instantiate(tcx, goal.predicate.trait_ref.substs);
            ecx.add_goals(nested_obligations.predicates.into_iter().map(|p| goal.with(tcx, p)));
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
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

    fn consider_builtin_pointer_like_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.self_ty().has_non_region_infer() {
            return ecx.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS);
        }

        let tcx = ecx.tcx();
        let self_ty = tcx.erase_regions(goal.predicate.self_ty());

        if let Ok(layout) = tcx.layout_of(goal.param_env.and(self_ty))
            && layout.layout.size() == tcx.data_layout.pointer_size
            && layout.layout.align().abi == tcx.data_layout.pointer_align.abi
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
        if let ty::FnPtr(..) = goal.predicate.self_ty().kind() {
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        } else {
            Err(NoSolution)
        }
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
                Some(a) => a,
                None => {
                    return ecx
                        .evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS);
                }
            };
        let output_is_sized_pred = tupled_inputs_and_output
            .map_bound(|(_, output)| tcx.at(DUMMY_SP).mk_trait_ref(LangItem::Sized, [output]));

        let pred = tupled_inputs_and_output
            .map_bound(|(inputs, _)| {
                tcx.mk_trait_ref(goal.predicate.def_id(), [goal.predicate.self_ty(), inputs])
            })
            .to_predicate(tcx);
        // A built-in `Fn` impl only holds if the output is sized.
        // (FIXME: technically we only need to check this if the type is a fn ptr...)
        Self::consider_implied_clause(ecx, goal, pred, [goal.with(tcx, output_is_sized_pred)])
    }

    fn consider_builtin_tuple_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if let ty::Tuple(..) = goal.predicate.self_ty().kind() {
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        } else {
            Err(NoSolution)
        }
    }

    fn consider_builtin_pointee_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        _goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
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
        // Technically, we need to check that the future output type is Sized,
        // but that's already proven by the generator being WF.
        ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
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
        Self::consider_implied_clause(
            ecx,
            goal,
            ty::Binder::dummy(
                tcx.mk_trait_ref(goal.predicate.def_id(), [self_ty, generator.resume_ty()]),
            )
            .to_predicate(tcx),
            // Technically, we need to check that the generator types are Sized,
            // but that's already proven by the generator being WF.
            [],
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
            return ecx.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS);
        }
        ecx.probe(|ecx| {
            match (a_ty.kind(), b_ty.kind()) {
                // Trait upcasting, or `dyn Trait + Auto + 'a` -> `dyn Trait + 'b`
                (&ty::Dynamic(_, _, ty::Dyn), &ty::Dynamic(_, _, ty::Dyn)) => {
                    // Dyn upcasting is handled separately, since due to upcasting,
                    // when there are two supertraits that differ by substs, we
                    // may return more than one query response.
                    Err(NoSolution)
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
                    // Check that the type implements all of the predicates of the def-id.
                    // (i.e. the principal, all of the associated types match, and any auto traits)
                    ecx.add_goals(
                        data.iter().map(|pred| goal.with(tcx, pred.with_self_ty(tcx, a_ty))),
                    );
                    // The type must be Sized to be unsized.
                    ecx.add_goal(
                        goal.with(tcx, ty::Binder::dummy(tcx.mk_trait_ref(sized_def_id, [a_ty]))),
                    );
                    // The type must outlive the lifetime of the `dyn` we're unsizing into.
                    ecx.add_goal(
                        goal.with(tcx, ty::Binder::dummy(ty::OutlivesPredicate(a_ty, region))),
                    );
                    ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                }
                // `[T; n]` -> `[T]` unsizing
                (&ty::Array(a_elem_ty, ..), &ty::Slice(b_elem_ty)) => {
                    // We just require that the element type stays the same
                    ecx.eq(goal.param_env, a_elem_ty, b_elem_ty)?;
                    ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
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
                    let tail_field_ty = tcx.type_of(tail_field.did);

                    let a_tail_ty = tail_field_ty.subst(tcx, a_substs);
                    let b_tail_ty = tail_field_ty.subst(tcx, b_substs);

                    // Substitute just the unsizing params from B into A. The type after
                    // this substitution must be equal to B. This is so we don't unsize
                    // unrelated type parameters.
                    let new_a_substs =
                        tcx.mk_substs_from_iter(a_substs.iter().enumerate().map(|(i, a)| {
                            if unsizing_params.contains(i as u32) { b_substs[i] } else { a }
                        }));
                    let unsized_a_ty = tcx.mk_adt(a_def, new_a_substs);

                    // Finally, we require that `TailA: Unsize<TailB>` for the tail field
                    // types.
                    ecx.eq(goal.param_env, unsized_a_ty, b_ty)?;
                    ecx.add_goal(goal.with(
                        tcx,
                        ty::Binder::dummy(
                            tcx.mk_trait_ref(goal.predicate.def_id(), [a_tail_ty, b_tail_ty]),
                        ),
                    ));
                    ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                }
                // Tuple unsizing `(.., T)` -> `(.., U)` where `T: Unsize<U>`
                (&ty::Tuple(a_tys), &ty::Tuple(b_tys))
                    if a_tys.len() == b_tys.len() && !a_tys.is_empty() =>
                {
                    let (a_last_ty, a_rest_tys) = a_tys.split_last().unwrap();
                    let b_last_ty = b_tys.last().unwrap();

                    // Substitute just the tail field of B., and require that they're equal.
                    let unsized_a_ty =
                        tcx.mk_tup_from_iter(a_rest_tys.iter().chain([b_last_ty]).copied());
                    ecx.eq(goal.param_env, unsized_a_ty, b_ty)?;

                    // Similar to ADTs, require that the rest of the fields are equal.
                    ecx.add_goal(goal.with(
                        tcx,
                        ty::Binder::dummy(
                            tcx.mk_trait_ref(goal.predicate.def_id(), [*a_last_ty, *b_last_ty]),
                        ),
                    ));
                    ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
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
            ecx.probe(|ecx| -> Result<_, NoSolution> {
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
                let new_a_data = tcx.mk_poly_existential_predicates_from_iter(new_a_data);
                let new_a_ty = tcx.mk_dynamic(new_a_data, b_region, ty::Dyn);

                // We also require that A's lifetime outlives B's lifetime.
                ecx.eq(goal.param_env, new_a_ty, b_ty)?;
                ecx.add_goal(
                    goal.with(tcx, ty::Binder::dummy(ty::OutlivesPredicate(a_region, b_region))),
                );
                ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
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

    fn consider_builtin_discriminant_kind_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        _goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        // `DiscriminantKind` is automatically implemented for every type.
        ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    fn consider_builtin_destruct_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx> {
        if !goal.param_env.is_const() {
            // `Destruct` is automatically implemented for every type in
            // non-const environments.
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        } else {
            // FIXME(-Ztrait-solver=next): Implement this when we get const working in the new solver
            Err(NoSolution)
        }
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
        constituent_tys: impl Fn(&EvalCtxt<'_, 'tcx>, Ty<'tcx>) -> Result<Vec<Ty<'tcx>>, NoSolution>,
    ) -> QueryResult<'tcx> {
        self.probe(|ecx| {
            ecx.add_goals(
                constituent_tys(ecx, goal.predicate.self_ty())?
                    .into_iter()
                    .map(|ty| {
                        goal.with(
                            ecx.tcx(),
                            ty::Binder::dummy(goal.predicate.with_self_ty(ecx.tcx(), ty)),
                        )
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
