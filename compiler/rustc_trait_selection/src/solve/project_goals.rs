use crate::traits::{specialization_graph, translate_substs};

use super::assembly::{self, AssemblyCtxt};
use super::{CanonicalGoal, EvalCtxt, Goal, QueryResult};
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::{InferCtxt, InferOk};
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::specialization_graph::LeafDef;
use rustc_infer::traits::{ObligationCause, Reveal};
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams};
use rustc_middle::ty::ProjectionPredicate;
use rustc_middle::ty::TypeVisitable;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::DUMMY_SP;
use std::iter;

#[allow(dead_code)] // FIXME: implement and use all variants.
#[derive(Debug, Clone, Copy)]
pub(super) enum CandidateSource {
    Impl(DefId),
    ParamEnv(usize),
    Builtin,
}

type Candidate<'tcx> = assembly::Candidate<'tcx, ProjectionPredicate<'tcx>>;

impl<'tcx> EvalCtxt<'tcx> {
    pub(super) fn compute_projection_goal(
        &mut self,
        goal: CanonicalGoal<'tcx, ProjectionPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        let candidates = AssemblyCtxt::assemble_and_evaluate_candidates(self, goal);
        self.merge_project_candidates(candidates)
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

impl<'tcx> assembly::GoalKind<'tcx> for ProjectionPredicate<'tcx> {
    type CandidateSource = CandidateSource;

    fn self_ty(self) -> Ty<'tcx> {
        self.self_ty()
    }

    fn with_self_ty(self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> Self {
        self.with_self_ty(tcx, self_ty)
    }

    fn trait_def_id(self, tcx: TyCtxt<'tcx>) -> DefId {
        self.trait_def_id(tcx)
    }

    fn consider_impl_candidate(
        acx: &mut AssemblyCtxt<'_, 'tcx, ProjectionPredicate<'tcx>>,
        goal: Goal<'tcx, ProjectionPredicate<'tcx>>,
        impl_def_id: DefId,
    ) {
        let tcx = acx.cx.tcx;
        let goal_trait_ref = goal.predicate.projection_ty.trait_ref(tcx);
        let impl_trait_ref = tcx.bound_impl_trait_ref(impl_def_id).unwrap();
        let drcx = DeepRejectCtxt { treat_obligation_params: TreatParams::AsPlaceholder };
        if iter::zip(goal_trait_ref.substs, impl_trait_ref.skip_binder().substs)
            .any(|(goal, imp)| !drcx.generic_args_may_unify(goal, imp))
        {
            return;
        }

        acx.infcx.probe(|_| {
            let impl_substs = acx.infcx.fresh_substs_for_item(DUMMY_SP, impl_def_id);
            let impl_trait_ref = impl_trait_ref.subst(tcx, impl_substs);

            let Ok(InferOk { obligations, .. }) = acx
                .infcx
                .at(&ObligationCause::dummy(), goal.param_env)
                .define_opaque_types(false)
                .eq(goal_trait_ref, impl_trait_ref)
                .map_err(|e| debug!("failed to equate trait refs: {e:?}"))
            else {
                return
            };

            let nested_goals = obligations.into_iter().map(|o| o.into()).collect();
            let Ok(trait_ref_certainty) = acx.cx.evaluate_all(acx.infcx, nested_goals) else { return };

            let Some(assoc_def) = fetch_eligible_assoc_item_def(
                acx.infcx,
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
                acx.infcx,
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

            let Ok(InferOk { obligations, .. }) = acx
                .infcx
                .at(&ObligationCause::dummy(), goal.param_env)
                .define_opaque_types(false)
                .eq(goal.predicate.term,  term.subst(tcx, substs))
                .map_err(|e| debug!("failed to equate trait refs: {e:?}"))
            else {
                return
            };

            let nested_goals = obligations.into_iter().map(|o| o.into()).collect();
            let Ok(rhs_certainty) = acx.cx.evaluate_all(acx.infcx, nested_goals) else { return };

            let certainty = trait_ref_certainty.unify_and(rhs_certainty);
            acx.try_insert_candidate(CandidateSource::Impl(impl_def_id), certainty);
        })
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
) -> Option<LeafDef> {
    let node_item = specialization_graph::assoc_def(infcx.tcx, impl_def_id, trait_assoc_def_id)
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
            let poly_trait_ref = infcx.resolve_vars_if_possible(goal_trait_ref);
            !poly_trait_ref.still_further_specializable()
        } else {
            debug!(?node_item.item.def_id, "not eligible due to default");
            false
        }
    };

    if eligible { Some(node_item) } else { None }
}
