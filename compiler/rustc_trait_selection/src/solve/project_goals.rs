use crate::traits::{specialization_graph, translate_substs};

use super::infcx_ext::InferCtxtExt;
use super::{
    fixme_instantiate_canonical_query_response, CanonicalGoal, CanonicalResponse, Certainty,
    EvalCtxt, Goal, QueryResult,
};
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::canonical::{CanonicalVarValues, OriginalQueryValues};
use rustc_infer::infer::{InferCtxt, InferOk, TyCtxtInferExt};
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::specialization_graph::LeafDef;
use rustc_infer::traits::{ObligationCause, Reveal};
use rustc_middle::ty;
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams};
use rustc_middle::ty::ProjectionPredicate;
use rustc_middle::ty::TypeVisitable;
use rustc_span::DUMMY_SP;
use std::iter;

// FIXME: Deduplicate the candidate code between projection and trait goal.

/// Similar to [super::trait_goals::Candidate] but for `Projection` goals.
#[derive(Debug, Clone)]
struct Candidate<'tcx> {
    source: CandidateSource,
    result: CanonicalResponse<'tcx>,
}

#[allow(dead_code)] // FIXME: implement and use all variants.
#[derive(Debug, Clone, Copy)]
enum CandidateSource {
    Impl(DefId),
    ParamEnv(usize),
    Builtin,
}

impl<'tcx> EvalCtxt<'tcx> {
    pub(super) fn compute_projection_goal(
        &mut self,
        goal: CanonicalGoal<'tcx, ProjectionPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        let candidates = self.assemble_and_evaluate_project_candidates(goal);
        self.merge_project_candidates(candidates)
    }

    fn assemble_and_evaluate_project_candidates(
        &mut self,
        goal: CanonicalGoal<'tcx, ProjectionPredicate<'tcx>>,
    ) -> Vec<Candidate<'tcx>> {
        let (ref infcx, goal, var_values) =
            self.tcx.infer_ctxt().build_with_canonical(DUMMY_SP, &goal);
        let mut acx = AssemblyCtxt { cx: self, infcx, var_values, candidates: Vec::new() };

        acx.assemble_candidates_after_normalizing_self_ty(goal);
        acx.assemble_impl_candidates(goal);
        acx.candidates
    }

    fn merge_project_candidates(
        &mut self,
        mut candidates: Vec<Candidate<'tcx>>,
    ) -> QueryResult<'tcx> {
        match candidates.len() {
            0 => return Err(NoSolution),
            1 => return Ok(candidates.pop().unwrap().result),
            _ => {}
        }

        if candidates.len() > 1 {
            let mut i = 0;
            'outer: while i < candidates.len() {
                for j in (0..candidates.len()).filter(|&j| i != j) {
                    if self.project_candidate_should_be_dropped_in_favor_of(
                        &candidates[i],
                        &candidates[j],
                    ) {
                        debug!(candidate = ?candidates[i], "Dropping candidate #{}/{}", i, candidates.len());
                        candidates.swap_remove(i);
                        continue 'outer;
                    }
                }

                debug!(candidate = ?candidates[i], "Retaining candidate #{}/{}", i, candidates.len());
                // If there are *STILL* multiple candidates, give up
                // and report ambiguity.
                i += 1;
                if i > 1 {
                    debug!("multiple matches, ambig");
                    // FIXME: return overflow if all candidates overflow, otherwise return ambiguity.
                    unimplemented!();
                }
            }
        }

        Ok(candidates.pop().unwrap().result)
    }

    fn project_candidate_should_be_dropped_in_favor_of(
        &self,
        candidate: &Candidate<'tcx>,
        other: &Candidate<'tcx>,
    ) -> bool {
        // FIXME: implement this
        match (candidate.source, other.source) {
            (CandidateSource::Impl(_), _)
            | (CandidateSource::ParamEnv(_), _)
            | (CandidateSource::Builtin, _) => unimplemented!(),
        }
    }
}

/// Similar to [super::trait_goals::AssemblyCtxt] but for `Projection` goals.
struct AssemblyCtxt<'a, 'tcx> {
    cx: &'a mut EvalCtxt<'tcx>,
    infcx: &'a InferCtxt<'tcx>,
    var_values: CanonicalVarValues<'tcx>,
    candidates: Vec<Candidate<'tcx>>,
}

impl<'tcx> AssemblyCtxt<'_, 'tcx> {
    fn try_insert_candidate(&mut self, source: CandidateSource, certainty: Certainty) {
        match self.infcx.make_canonical_response(self.var_values.clone(), certainty) {
            Ok(result) => self.candidates.push(Candidate { source, result }),
            Err(NoSolution) => debug!(?source, ?certainty, "failed leakcheck"),
        }
    }

    fn assemble_candidates_after_normalizing_self_ty(
        &mut self,
        goal: Goal<'tcx, ProjectionPredicate<'tcx>>,
    ) {
        let tcx = self.cx.tcx;
        let &ty::Alias(ty::Projection, projection_ty) = goal.predicate.projection_ty.self_ty().kind() else {
            return
        };
        self.infcx.probe(|_| {
            let normalized_ty = self.infcx.next_ty_infer();
            let normalizes_to_goal = goal.with(
                tcx,
                ty::Binder::dummy(ty::ProjectionPredicate {
                    projection_ty,
                    term: normalized_ty.into(),
                }),
            );
            let normalization_certainty =
                match self.cx.evaluate_goal(&self.infcx, normalizes_to_goal) {
                    Ok((_, certainty)) => certainty,
                    Err(NoSolution) => return,
                };

            // NOTE: Alternatively we could call `evaluate_goal` here and only have a `Normalized` candidate.
            // This doesn't work as long as we use `CandidateSource` in both winnowing and to resolve associated items.
            let goal = goal.with(tcx, goal.predicate.with_self_ty(tcx, normalized_ty));
            let mut orig_values = OriginalQueryValues::default();
            let goal = self.infcx.canonicalize_query(goal, &mut orig_values);
            let normalized_candidates = self.cx.assemble_and_evaluate_project_candidates(goal);
            // Map each candidate from being canonical wrt the current inference context to being
            // canonical wrt the caller.
            for Candidate { source, result } in normalized_candidates {
                self.infcx.probe(|_| {
                    let candidate_certainty = fixme_instantiate_canonical_query_response(
                        self.infcx,
                        &orig_values,
                        result,
                    );
                    self.try_insert_candidate(
                        source,
                        normalization_certainty.unify_and(candidate_certainty),
                    )
                })
            }
        })
    }

    fn assemble_impl_candidates(&mut self, goal: Goal<'tcx, ProjectionPredicate<'tcx>>) {
        self.cx.tcx.for_each_relevant_impl(
            goal.predicate.trait_def_id(self.cx.tcx),
            goal.predicate.self_ty(),
            |impl_def_id| self.consider_impl_candidate(goal, impl_def_id),
        );
    }

    fn consider_impl_candidate(
        &mut self,
        goal: Goal<'tcx, ProjectionPredicate<'tcx>>,
        impl_def_id: DefId,
    ) {
        let tcx = self.cx.tcx;
        let goal_trait_ref = goal.predicate.projection_ty.trait_ref(tcx);
        let impl_trait_ref = tcx.bound_impl_trait_ref(impl_def_id).unwrap();
        let drcx = DeepRejectCtxt { treat_obligation_params: TreatParams::AsPlaceholder };
        if iter::zip(goal_trait_ref.substs, impl_trait_ref.skip_binder().substs)
            .any(|(goal, imp)| !drcx.generic_args_may_unify(goal, imp))
        {
            return;
        }

        self.infcx.probe(|_| {
            let impl_substs = self.infcx.fresh_substs_for_item(DUMMY_SP, impl_def_id);
            let impl_trait_ref = impl_trait_ref.subst(tcx, impl_substs);

            let Ok(InferOk { obligations, .. }) = self
                .infcx
                .at(&ObligationCause::dummy(), goal.param_env)
                .define_opaque_types(false)
                .eq(goal_trait_ref, impl_trait_ref)
                .map_err(|e| debug!("failed to equate trait refs: {e:?}"))
            else {
                return
            };

            let nested_goals = obligations.into_iter().map(|o| o.into()).collect();
            let Ok(trait_ref_certainty) = self.cx.evaluate_all(self.infcx, nested_goals) else { return };

            let Some(assoc_def) = self.fetch_eligible_assoc_item_def(
                goal.param_env,
                goal_trait_ref,
                goal.predicate.def_id(),
                impl_def_id
            ) else {
                return
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
                impl_trait_ref.substs,
            );
            let substs = translate_substs(
                self.infcx,
                goal.param_env,
                impl_def_id,
                impl_substs_with_gat,
                assoc_def.defining_node,
            );

            // Finally we construct the actual value of the associated type.
            let is_const = matches!(tcx.def_kind(assoc_def.item.def_id), DefKind::AssocConst);
            let ty = tcx.bound_type_of(assoc_def.item.def_id);
            let term: ty::EarlyBinder<ty::Term<'tcx>> = if is_const {
                let identity_substs = ty::InternalSubsts::identity_for_item(tcx, assoc_def.item.def_id);
                let did = ty::WithOptConstParam::unknown(assoc_def.item.def_id);
                let kind =
                    ty::ConstKind::Unevaluated(ty::UnevaluatedConst::new(did, identity_substs));
                ty.map_bound(|ty| tcx.mk_const(kind, ty).into())
            } else {
                ty.map_bound(|ty| ty.into())
            };

            let Ok(InferOk { obligations, .. }) = self
                .infcx
                .at(&ObligationCause::dummy(), goal.param_env)
                .define_opaque_types(false)
                .eq(goal.predicate.term,  term.subst(tcx, substs))
                .map_err(|e| debug!("failed to equate trait refs: {e:?}"))
            else {
                return
            };

            let nested_goals = obligations.into_iter().map(|o| o.into()).collect();
            let Ok(rhs_certainty) = self.cx.evaluate_all(self.infcx, nested_goals) else { return };

            let certainty = trait_ref_certainty.unify_and(rhs_certainty);
            self.try_insert_candidate(CandidateSource::Impl(impl_def_id), certainty);
        })
    }

    /// This behavior is also implemented in `rustc_ty_utils` and in the old `project` code.
    ///
    /// FIXME: We should merge these 3 implementations as it's likely that they otherwise
    /// diverge.
    #[instrument(level = "debug", skip(self, param_env), ret)]
    fn fetch_eligible_assoc_item_def(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        goal_trait_ref: ty::TraitRef<'tcx>,
        trait_assoc_def_id: DefId,
        impl_def_id: DefId,
    ) -> Option<LeafDef> {
        let node_item =
            specialization_graph::assoc_def(self.cx.tcx, impl_def_id, trait_assoc_def_id)
                .map_err(|ErrorGuaranteed { .. }| ())
                .ok()?;

        let eligible = if node_item.is_final() {
            // Non-specializable items are always projectable.
            true
        } else {
            // Only reveal a specializable default if we're past type-checking
            // and the obligation is monomorphic, otherwise passes such as
            // transmute checking and polymorphic MIR optimizations could
            // get a result which isn't correct for all monomorphizations.
            if param_env.reveal() == Reveal::All {
                let poly_trait_ref = self.infcx.resolve_vars_if_possible(goal_trait_ref);
                !poly_trait_ref.still_further_specializable()
            } else {
                debug!(?node_item.item.def_id, "not eligible due to default");
                false
            }
        };

        if eligible { Some(node_item) } else { None }
    }
}
