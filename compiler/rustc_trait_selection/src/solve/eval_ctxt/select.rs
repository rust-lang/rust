use std::ops::ControlFlow;

use rustc_hir::def_id::DefId;
use rustc_infer::infer::{DefineOpaqueTypes, InferCtxt, InferOk, TyCtxtInferExt};
use rustc_infer::traits::util::supertraits;
use rustc_infer::traits::{
    Obligation, PredicateObligation, Selection, SelectionResult, TraitObligation,
};
use rustc_middle::infer::canonical::{Canonical, CanonicalVarValues};
use rustc_middle::traits::solve::{Certainty, Goal, PredefinedOpaquesData, QueryInput};
use rustc_middle::traits::{
    DefiningAnchor, ImplSource, ImplSourceObjectData, ImplSourceTraitUpcastingData,
    ImplSourceUserDefinedData, ObligationCause, SelectionError,
};
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::DUMMY_SP;

use crate::solve::assembly::{BuiltinImplSource, Candidate, CandidateSource};
use crate::solve::eval_ctxt::{EvalCtxt, NestedGoals};
use crate::solve::inspect::ProofTreeBuilder;
use crate::solve::search_graph::SearchGraph;
use crate::solve::SolverMode;
use crate::traits::vtable::{count_own_vtable_entries, prepare_vtable_segments, VtblSegment};

pub trait InferCtxtSelectExt<'tcx> {
    fn select_in_new_trait_solver(
        &self,
        obligation: &TraitObligation<'tcx>,
    ) -> SelectionResult<'tcx, Selection<'tcx>>;
}

impl<'tcx> InferCtxtSelectExt<'tcx> for InferCtxt<'tcx> {
    fn select_in_new_trait_solver(
        &self,
        obligation: &TraitObligation<'tcx>,
    ) -> SelectionResult<'tcx, Selection<'tcx>> {
        assert!(self.next_trait_solver());

        let goal = Goal::new(
            self.tcx,
            obligation.param_env,
            self.instantiate_binder_with_placeholders(obligation.predicate),
        );

        let mode = if self.intercrate { SolverMode::Coherence } else { SolverMode::Normal };
        let mut search_graph = SearchGraph::new(self.tcx, mode);
        let mut ecx = EvalCtxt {
            search_graph: &mut search_graph,
            infcx: self,
            // Only relevant when canonicalizing the response,
            // which we don't do within this evaluation context.
            predefined_opaques_in_body: self
                .tcx
                .mk_predefined_opaques_in_body(PredefinedOpaquesData::default()),
            // Only relevant when canonicalizing the response.
            max_input_universe: ty::UniverseIndex::ROOT,
            var_values: CanonicalVarValues::dummy(),
            nested_goals: NestedGoals::new(),
            tainted: Ok(()),
            inspect: ProofTreeBuilder::new_noop(),
        };

        let (orig_values, canonical_goal) = ecx.canonicalize_goal(goal);
        let mut candidates = ecx.compute_canonical_trait_candidates(canonical_goal);

        // pseudo-winnow
        if candidates.len() == 0 {
            return Err(SelectionError::Unimplemented);
        } else if candidates.len() > 1 {
            let mut i = 0;
            while i < candidates.len() {
                let should_drop_i = (0..candidates.len()).filter(|&j| i != j).any(|j| {
                    candidate_should_be_dropped_in_favor_of(&candidates[i], &candidates[j])
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
            .instantiate_and_apply_query_response(goal.param_env, orig_values, candidate.result)
            .map_err(|_| SelectionError::Unimplemented)?;

        let goal = self.resolve_vars_if_possible(goal);

        let nested_obligations: Vec<_> = nested_goals
            .into_iter()
            .map(|goal| {
                Obligation::new(self.tcx, ObligationCause::dummy(), goal.param_env, goal.predicate)
            })
            .collect();

        if let Certainty::Maybe(_) = certainty {
            return Ok(None);
        }

        match (certainty, candidate.source) {
            (_, CandidateSource::Impl(def_id)) => {
                rematch_impl(self, goal, def_id, nested_obligations)
            }

            (
                _,
                CandidateSource::BuiltinImpl(
                    BuiltinImplSource::Object | BuiltinImplSource::TraitUpcasting,
                ),
            ) => rematch_object(self, goal, nested_obligations),

            (Certainty::Yes, CandidateSource::BuiltinImpl(BuiltinImplSource::Misc)) => {
                // technically some builtin impls have nested obligations, but if
                // `Certainty::Yes`, then they should've all been verified by the
                // evaluation above.
                Ok(Some(ImplSource::Builtin(nested_obligations)))
            }

            (Certainty::Yes, CandidateSource::ParamEnv(_) | CandidateSource::AliasBound) => {
                // It's fine not to do anything to rematch these, since there are no
                // nested obligations.
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
        canonical_input: Canonical<'tcx, QueryInput<'tcx, ty::TraitPredicate<'tcx>>>,
    ) -> Vec<Candidate<'tcx>> {
        let intercrate = match self.search_graph.solver_mode() {
            SolverMode::Normal => false,
            SolverMode::Coherence => true,
        };
        let (canonical_infcx, input, var_values) = self
            .tcx()
            .infer_ctxt()
            .intercrate(intercrate)
            .with_next_trait_solver(true)
            .with_opaque_type_inference(canonical_input.value.anchor)
            .build_with_canonical(DUMMY_SP, &canonical_input);

        let mut ecx = EvalCtxt {
            infcx: &canonical_infcx,
            var_values,
            predefined_opaques_in_body: input.predefined_opaques_in_body,
            max_input_universe: canonical_input.max_universe,
            search_graph: &mut self.search_graph,
            nested_goals: NestedGoals::new(),
            tainted: Ok(()),
            inspect: ProofTreeBuilder::new_noop(),
        };

        for &(key, ty) in &input.predefined_opaques_in_body.opaque_types {
            ecx.insert_hidden_type(key, input.goal.param_env, ty)
                .expect("failed to prepopulate opaque types");
        }

        let candidates = ecx.assemble_and_evaluate_candidates(input.goal);

        // We don't need the canonicalized context anymore
        if input.anchor != DefiningAnchor::Error {
            let _ = canonical_infcx.take_opaque_types();
        }

        candidates
    }
}

fn candidate_should_be_dropped_in_favor_of<'tcx>(
    victim: &Candidate<'tcx>,
    other: &Candidate<'tcx>,
) -> bool {
    match (victim.source, other.source) {
        (CandidateSource::ParamEnv(i), CandidateSource::ParamEnv(j)) => i >= j,
        (_, CandidateSource::ParamEnv(_)) => true,
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
    let source_trait_ref = if let ty::Dynamic(data, _, ty::Dyn) = self_ty.kind() {
        data.principal().unwrap().with_self_ty(infcx.tcx, self_ty)
    } else {
        bug!()
    };

    let (is_upcasting, target_trait_ref_unnormalized) = if Some(goal.predicate.def_id())
        == infcx.tcx.lang_items().unsize_trait()
    {
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

    // If we're upcasting, get the offset of the vtable pointer, which is
    Ok(Some(if is_upcasting {
        ImplSource::TraitUpcasting(ImplSourceTraitUpcastingData { vtable_vptr_slot, nested })
    } else {
        ImplSource::Object(ImplSourceObjectData { vtable_base, nested })
    }))
}
