use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::{DefineOpaqueTypes, InferCtxt};
use rustc_infer::traits::{
    Obligation, PolyTraitObligation, PredicateObligation, Selection, SelectionResult, TraitEngine,
};
use rustc_middle::traits::solve::{CanonicalInput, Certainty, Goal};
use rustc_middle::traits::{
    BuiltinImplSource, ImplSource, ImplSourceUserDefinedData, ObligationCause, SelectionError,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::DUMMY_SP;

use crate::solve::assembly::{Candidate, CandidateSource};
use crate::solve::eval_ctxt::{EvalCtxt, GenerateProofTree};
use crate::solve::inspect::ProofTreeBuilder;
use crate::solve::search_graph::OverflowHandler;
use crate::traits::StructurallyNormalizeExt;
use crate::traits::TraitEngineExt;

pub trait InferCtxtSelectExt<'tcx> {
    fn select_in_new_trait_solver(
        &self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> SelectionResult<'tcx, Selection<'tcx>>;
}

impl<'tcx> InferCtxtSelectExt<'tcx> for InferCtxt<'tcx> {
    fn select_in_new_trait_solver(
        &self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> SelectionResult<'tcx, Selection<'tcx>> {
        assert!(self.next_trait_solver());

        let trait_goal = Goal::new(
            self.tcx,
            obligation.param_env,
            self.instantiate_binder_with_placeholders(obligation.predicate),
        );

        let (result, _) = EvalCtxt::enter_root(self, GenerateProofTree::Never, |ecx| {
            let goal = Goal::new(ecx.tcx(), trait_goal.param_env, trait_goal.predicate);
            let (orig_values, canonical_goal) = ecx.canonicalize_goal(goal);
            let mut candidates = ecx.compute_canonical_trait_candidates(canonical_goal);

            // pseudo-winnow
            if candidates.len() == 0 {
                return Err(SelectionError::Unimplemented);
            } else if candidates.len() > 1 {
                let mut i = 0;
                while i < candidates.len() {
                    let should_drop_i = (0..candidates.len()).filter(|&j| i != j).any(|j| {
                        candidate_should_be_dropped_in_favor_of(
                            ecx.tcx(),
                            &candidates[i],
                            &candidates[j],
                        )
                    });
                    if should_drop_i {
                        candidates.swap_remove(i);
                    } else {
                        i += 1;
                        if i > 1 {
                            return Ok(None);
                        }
                    }
                }
            }

            let candidate = candidates.pop().unwrap();
            let (certainty, nested_goals) = ecx
                .instantiate_and_apply_query_response(
                    trait_goal.param_env,
                    orig_values,
                    candidate.result,
                )
                .map_err(|_| SelectionError::Unimplemented)?;

            Ok(Some((candidate, certainty, nested_goals)))
        });

        let (candidate, certainty, nested_goals) = match result {
            Ok(Some((candidate, certainty, nested_goals))) => (candidate, certainty, nested_goals),
            Ok(None) => return Ok(None),
            Err(e) => return Err(e),
        };

        let nested_obligations: Vec<_> = nested_goals
            .into_iter()
            .map(|goal| {
                Obligation::new(self.tcx, ObligationCause::dummy(), goal.param_env, goal.predicate)
            })
            .collect();

        let goal = self.resolve_vars_if_possible(trait_goal);
        match (certainty, candidate.source) {
            // Rematching the implementation will instantiate the same nested goals that
            // would have caused the ambiguity, so we can still make progress here regardless.
            (_, CandidateSource::Impl(def_id)) => {
                rematch_impl(self, goal, def_id, nested_obligations)
            }

            (Certainty::Maybe(_), CandidateSource::BuiltinImpl(src))
                if self.tcx.lang_items().unsize_trait() == Some(goal.predicate.def_id()) =>
            {
                rematch_unsize(self, goal, nested_obligations, src)
            }

            // Technically some builtin impls have nested obligations, but if
            // `Certainty::Yes`, then they should've all been verified and don't
            // need re-checking.
            (Certainty::Yes, CandidateSource::BuiltinImpl(src)) => {
                Ok(Some(ImplSource::Builtin(src, nested_obligations)))
            }

            // It's fine not to do anything to rematch these, since there are no
            // nested obligations.
            (Certainty::Yes, CandidateSource::ParamEnv(_) | CandidateSource::AliasBound) => {
                Ok(Some(ImplSource::Param(ty::BoundConstness::NotConst, nested_obligations)))
            }

            (Certainty::Maybe(_), _) => Ok(None),
        }
    }
}

impl<'tcx> EvalCtxt<'_, 'tcx> {
    fn compute_canonical_trait_candidates(
        &mut self,
        canonical_input: CanonicalInput<'tcx>,
    ) -> Vec<Candidate<'tcx>> {
        // This doesn't record the canonical goal on the stack during the
        // candidate assembly step, but that's fine. Selection is conceptually
        // outside of the solver, and if there were any cycles, we'd encounter
        // the cycle anyways one step later.
        EvalCtxt::enter_canonical(
            self.tcx(),
            self.search_graph(),
            canonical_input,
            // FIXME: This is wrong, idk if we even want to track stuff here.
            &mut ProofTreeBuilder::new_noop(),
            |ecx, goal| {
                let trait_goal = Goal {
                    param_env: goal.param_env,
                    predicate: goal
                        .predicate
                        .to_opt_poly_trait_pred()
                        .expect("we canonicalized a trait goal")
                        .no_bound_vars()
                        .expect("we instantiated all bound vars"),
                };
                ecx.assemble_and_evaluate_candidates(trait_goal)
            },
        )
    }
}

fn candidate_should_be_dropped_in_favor_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    victim: &Candidate<'tcx>,
    other: &Candidate<'tcx>,
) -> bool {
    match (victim.source, other.source) {
        (CandidateSource::ParamEnv(victim_idx), CandidateSource::ParamEnv(other_idx)) => {
            victim_idx >= other_idx
        }
        (_, CandidateSource::ParamEnv(_)) => true,

        // FIXME: we could prefer earlier vtable bases perhaps...
        (
            CandidateSource::BuiltinImpl(BuiltinImplSource::Object { .. }),
            CandidateSource::BuiltinImpl(BuiltinImplSource::Object { .. }),
        ) => false,
        (_, CandidateSource::BuiltinImpl(BuiltinImplSource::Object { .. })) => true,

        (CandidateSource::Impl(victim_def_id), CandidateSource::Impl(other_def_id)) => {
            tcx.specializes((other_def_id, victim_def_id))
                && other.result.value.certainty == Certainty::Yes
        }

        _ => false,
    }
}

fn rematch_impl<'tcx>(
    infcx: &InferCtxt<'tcx>,
    goal: Goal<'tcx, ty::TraitPredicate<'tcx>>,
    impl_def_id: DefId,
    mut nested: Vec<PredicateObligation<'tcx>>,
) -> SelectionResult<'tcx, Selection<'tcx>> {
    let args = infcx.fresh_args_for_item(DUMMY_SP, impl_def_id);
    let impl_trait_ref =
        infcx.tcx.impl_trait_ref(impl_def_id).unwrap().instantiate(infcx.tcx, args);

    nested.extend(
        infcx
            .at(&ObligationCause::dummy(), goal.param_env)
            .eq(DefineOpaqueTypes::No, goal.predicate.trait_ref, impl_trait_ref)
            .map_err(|_| SelectionError::Unimplemented)?
            .into_obligations(),
    );

    nested.extend(
        infcx.tcx.predicates_of(impl_def_id).instantiate(infcx.tcx, args).into_iter().map(
            |(pred, _)| Obligation::new(infcx.tcx, ObligationCause::dummy(), goal.param_env, pred),
        ),
    );

    Ok(Some(ImplSource::UserDefined(ImplSourceUserDefinedData { impl_def_id, args, nested })))
}

/// The `Unsize` trait is particularly important to coercion, so we try rematch it.
/// NOTE: This must stay in sync with `consider_builtin_unsize_candidate` in trait
/// goal assembly in the solver, both for soundness and in order to avoid ICEs.
fn rematch_unsize<'tcx>(
    infcx: &InferCtxt<'tcx>,
    goal: Goal<'tcx, ty::TraitPredicate<'tcx>>,
    mut nested: Vec<PredicateObligation<'tcx>>,
    source: BuiltinImplSource,
) -> SelectionResult<'tcx, Selection<'tcx>> {
    let tcx = infcx.tcx;
    let a_ty = structurally_normalize(goal.predicate.self_ty(), infcx, goal.param_env, &mut nested);
    let b_ty = structurally_normalize(
        goal.predicate.trait_ref.args.type_at(1),
        infcx,
        goal.param_env,
        &mut nested,
    );
    match (a_ty.kind(), b_ty.kind()) {
        (_, &ty::Dynamic(data, region, ty::Dyn)) => {
            // Check that the type implements all of the predicates of the def-id.
            // (i.e. the principal, all of the associated types match, and any auto traits)
            nested.extend(data.iter().map(|pred| {
                Obligation::new(
                    infcx.tcx,
                    ObligationCause::dummy(),
                    goal.param_env,
                    pred.with_self_ty(tcx, a_ty),
                )
            }));
            // The type must be Sized to be unsized.
            let sized_def_id = tcx.require_lang_item(hir::LangItem::Sized, None);
            nested.push(Obligation::new(
                infcx.tcx,
                ObligationCause::dummy(),
                goal.param_env,
                ty::TraitRef::new(tcx, sized_def_id, [a_ty]),
            ));
            // The type must outlive the lifetime of the `dyn` we're unsizing into.
            nested.push(Obligation::new(
                infcx.tcx,
                ObligationCause::dummy(),
                goal.param_env,
                ty::Binder::dummy(ty::OutlivesPredicate(a_ty, region)),
            ));

            Ok(Some(ImplSource::Builtin(source, nested)))
        }
        // `[T; n]` -> `[T]` unsizing
        (&ty::Array(a_elem_ty, ..), &ty::Slice(b_elem_ty)) => {
            nested.extend(
                infcx
                    .at(&ObligationCause::dummy(), goal.param_env)
                    .eq(DefineOpaqueTypes::No, a_elem_ty, b_elem_ty)
                    .expect("expected rematch to succeed")
                    .into_obligations(),
            );

            Ok(Some(ImplSource::Builtin(source, nested)))
        }
        // Struct unsizing `Struct<T>` -> `Struct<U>` where `T: Unsize<U>`
        (&ty::Adt(a_def, a_args), &ty::Adt(b_def, b_args))
            if a_def.is_struct() && a_def.did() == b_def.did() =>
        {
            let unsizing_params = tcx.unsizing_params_for_adt(a_def.did());
            // We must be unsizing some type parameters. This also implies
            // that the struct has a tail field.
            if unsizing_params.is_empty() {
                bug!("expected rematch to succeed")
            }

            let tail_field = a_def
                .non_enum_variant()
                .fields
                .raw
                .last()
                .expect("expected unsized ADT to have a tail field");
            let tail_field_ty = tcx.type_of(tail_field.did);

            let a_tail_ty = tail_field_ty.instantiate(tcx, a_args);
            let b_tail_ty = tail_field_ty.instantiate(tcx, b_args);

            // Substitute just the unsizing params from B into A. The type after
            // this substitution must be equal to B. This is so we don't unsize
            // unrelated type parameters.
            let new_a_args = tcx.mk_args_from_iter(
                a_args
                    .iter()
                    .enumerate()
                    .map(|(i, a)| if unsizing_params.contains(i as u32) { b_args[i] } else { a }),
            );
            let unsized_a_ty = Ty::new_adt(tcx, a_def, new_a_args);

            nested.extend(
                infcx
                    .at(&ObligationCause::dummy(), goal.param_env)
                    .eq(DefineOpaqueTypes::No, unsized_a_ty, b_ty)
                    .expect("expected rematch to succeed")
                    .into_obligations(),
            );

            // Finally, we require that `TailA: Unsize<TailB>` for the tail field
            // types.
            nested.push(Obligation::new(
                tcx,
                ObligationCause::dummy(),
                goal.param_env,
                ty::TraitRef::new(tcx, goal.predicate.def_id(), [a_tail_ty, b_tail_ty]),
            ));

            Ok(Some(ImplSource::Builtin(source, nested)))
        }
        // Tuple unsizing `(.., T)` -> `(.., U)` where `T: Unsize<U>`
        (&ty::Tuple(a_tys), &ty::Tuple(b_tys))
            if a_tys.len() == b_tys.len() && !a_tys.is_empty() =>
        {
            let (a_last_ty, a_rest_tys) = a_tys.split_last().unwrap();
            let b_last_ty = b_tys.last().unwrap();

            // Substitute just the tail field of B., and require that they're equal.
            let unsized_a_ty =
                Ty::new_tup_from_iter(tcx, a_rest_tys.iter().chain([b_last_ty]).copied());
            nested.extend(
                infcx
                    .at(&ObligationCause::dummy(), goal.param_env)
                    .eq(DefineOpaqueTypes::No, unsized_a_ty, b_ty)
                    .expect("expected rematch to succeed")
                    .into_obligations(),
            );

            // Similar to ADTs, require that the rest of the fields are equal.
            nested.push(Obligation::new(
                tcx,
                ObligationCause::dummy(),
                goal.param_env,
                ty::TraitRef::new(tcx, goal.predicate.def_id(), [*a_last_ty, *b_last_ty]),
            ));

            // We need to be able to detect tuple unsizing to require its feature gate.
            assert_eq!(
                source,
                BuiltinImplSource::TupleUnsizing,
                "compiler-errors wants to know if this can ever be triggered..."
            );
            Ok(Some(ImplSource::Builtin(source, nested)))
        }
        // FIXME: We *could* ICE here if either:
        // 1. the certainty is `Certainty::Yes`,
        // 2. we're in codegen (which should mean `Certainty::Yes`).
        _ => Ok(None),
    }
}

fn structurally_normalize<'tcx>(
    ty: Ty<'tcx>,
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    nested: &mut Vec<PredicateObligation<'tcx>>,
) -> Ty<'tcx> {
    if matches!(ty.kind(), ty::Alias(..)) {
        let mut engine = <dyn TraitEngine<'tcx>>::new(infcx);
        let normalized_ty = infcx
            .at(&ObligationCause::dummy(), param_env)
            .structurally_normalize(ty, &mut *engine)
            .expect("normalization shouldn't fail if we got to here");
        nested.extend(engine.pending_obligations());
        normalized_ty
    } else {
        ty
    }
}
