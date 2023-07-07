use std::ops::ControlFlow;

use rustc_hir::def_id::DefId;
use rustc_infer::infer::{DefineOpaqueTypes, InferCtxt, InferOk};
use rustc_infer::traits::util::supertraits;
use rustc_infer::traits::{
    Obligation, PolyTraitObligation, PredicateObligation, Selection, SelectionResult,
};
use rustc_middle::traits::solve::{CanonicalInput, Certainty, Goal};
use rustc_middle::traits::{
    ImplSource, ImplSourceObjectData, ImplSourceTraitUpcastingData, ImplSourceUserDefinedData,
    ObligationCause, SelectionError,
};
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::DUMMY_SP;

use crate::solve::assembly::{BuiltinImplSource, Candidate, CandidateSource};
use crate::solve::eval_ctxt::{EvalCtxt, GenerateProofTree};
use crate::solve::inspect::ProofTreeBuilder;
use crate::solve::search_graph::OverflowHandler;
use crate::traits::vtable::{count_own_vtable_entries, prepare_vtable_segments, VtblSegment};

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

        let (result, _) = EvalCtxt::enter_root(self, GenerateProofTree::No, |ecx| {
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

            // Rematching the dyn upcast or object goal will instantiate the same nested
            // goals that would have caused the ambiguity, so we can still make progress here
            // regardless.
            // FIXME: This doesn't actually check the object bounds hold here.
            (
                _,
                CandidateSource::BuiltinImpl(
                    BuiltinImplSource::Object | BuiltinImplSource::TraitUpcasting,
                ),
            ) => rematch_object(self, goal, nested_obligations),

            // Technically some builtin impls have nested obligations, but if
            // `Certainty::Yes`, then they should've all been verified and don't
            // need re-checking.
            (Certainty::Yes, CandidateSource::BuiltinImpl(BuiltinImplSource::Misc)) => {
                Ok(Some(ImplSource::Builtin(nested_obligations)))
            }

            // It's fine not to do anything to rematch these, since there are no
            // nested obligations.
            (Certainty::Yes, CandidateSource::ParamEnv(_) | CandidateSource::AliasBound) => {
                Ok(Some(ImplSource::Param(nested_obligations, ty::BoundConstness::NotConst)))
            }

            (_, CandidateSource::BuiltinImpl(BuiltinImplSource::Ambiguity))
            | (Certainty::Maybe(_), _) => Ok(None),
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

        (
            CandidateSource::BuiltinImpl(BuiltinImplSource::Object),
            CandidateSource::BuiltinImpl(BuiltinImplSource::Object),
        ) => false,
        (_, CandidateSource::BuiltinImpl(BuiltinImplSource::Object)) => true,

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
    let substs = infcx.fresh_substs_for_item(DUMMY_SP, impl_def_id);
    let impl_trait_ref = infcx.tcx.impl_trait_ref(impl_def_id).unwrap().subst(infcx.tcx, substs);

    nested.extend(
        infcx
            .at(&ObligationCause::dummy(), goal.param_env)
            .eq(DefineOpaqueTypes::No, goal.predicate.trait_ref, impl_trait_ref)
            .map_err(|_| SelectionError::Unimplemented)?
            .into_obligations(),
    );

    nested.extend(
        infcx.tcx.predicates_of(impl_def_id).instantiate(infcx.tcx, substs).into_iter().map(
            |(pred, _)| Obligation::new(infcx.tcx, ObligationCause::dummy(), goal.param_env, pred),
        ),
    );

    Ok(Some(ImplSource::UserDefined(ImplSourceUserDefinedData { impl_def_id, substs, nested })))
}

fn rematch_object<'tcx>(
    infcx: &InferCtxt<'tcx>,
    goal: Goal<'tcx, ty::TraitPredicate<'tcx>>,
    mut nested: Vec<PredicateObligation<'tcx>>,
) -> SelectionResult<'tcx, Selection<'tcx>> {
    let self_ty = goal.predicate.self_ty();
    let ty::Dynamic(data, _, source_kind) = *self_ty.kind()
    else {
        bug!()
    };
    let source_trait_ref = data.principal().unwrap().with_self_ty(infcx.tcx, self_ty);

    let (is_upcasting, target_trait_ref_unnormalized) = if Some(goal.predicate.def_id())
        == infcx.tcx.lang_items().unsize_trait()
    {
        assert_eq!(source_kind, ty::Dyn, "cannot upcast dyn*");
        if let ty::Dynamic(data, _, ty::Dyn) = goal.predicate.trait_ref.substs.type_at(1).kind() {
            (true, data.principal().unwrap().with_self_ty(infcx.tcx, self_ty))
        } else {
            bug!()
        }
    } else {
        (false, ty::Binder::dummy(goal.predicate.trait_ref))
    };

    let mut target_trait_ref = None;
    for candidate_trait_ref in supertraits(infcx.tcx, source_trait_ref) {
        let result = infcx.commit_if_ok(|_| {
            infcx.at(&ObligationCause::dummy(), goal.param_env).eq(
                DefineOpaqueTypes::No,
                target_trait_ref_unnormalized,
                candidate_trait_ref,
            )

            // FIXME: We probably should at least shallowly verify these...
        });

        match result {
            Ok(InferOk { value: (), obligations }) => {
                target_trait_ref = Some(candidate_trait_ref);
                nested.extend(obligations);
                break;
            }
            Err(_) => continue,
        }
    }

    let target_trait_ref = target_trait_ref.unwrap();

    let mut offset = 0;
    let Some((vtable_base, vtable_vptr_slot)) =
        prepare_vtable_segments(infcx.tcx, source_trait_ref, |segment| {
            match segment {
                VtblSegment::MetadataDSA => {
                    offset += TyCtxt::COMMON_VTABLE_ENTRIES.len();
                }
                VtblSegment::TraitOwnEntries { trait_ref, emit_vptr } => {
                    let own_vtable_entries = count_own_vtable_entries(infcx.tcx, trait_ref);

                    if trait_ref == target_trait_ref {
                        if emit_vptr {
                            return ControlFlow::Break((
                                offset,
                                Some(offset + count_own_vtable_entries(infcx.tcx, trait_ref)),
                            ));
                        } else {
                            return ControlFlow::Break((offset, None));
                        }
                    }

                    offset += own_vtable_entries;
                    if emit_vptr {
                        offset += 1;
                    }
                }
            }
            ControlFlow::Continue(())
        })
    else {
        bug!();
    };

    // If we're upcasting, get the offset of the vtable pointer, otherwise get
    // the base of the vtable.
    Ok(Some(if is_upcasting {
        ImplSource::TraitUpcasting(ImplSourceTraitUpcastingData { vtable_vptr_slot, nested })
    } else {
        ImplSource::Object(ImplSourceObjectData { vtable_base, nested })
    }))
}
