//! Code shared by trait and projection goals for candidate assembly.

use super::infcx_ext::InferCtxtExt;
use super::{
    fixme_instantiate_canonical_query_response, CanonicalGoal, CanonicalResponse, Certainty,
    EvalCtxt, Goal,
};
use rustc_hir::def_id::DefId;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::infer::{
    canonical::{CanonicalVarValues, OriginalQueryValues},
    InferCtxt,
};
use rustc_infer::traits::query::NoSolution;
use rustc_middle::ty::TypeFoldable;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::DUMMY_SP;
use std::fmt::Debug;

/// A candidate is a possible way to prove a goal.
///
/// It consists of both the `source`, which describes how that goal would be proven,
/// and the `result` when using the given `source`.
///
/// For the list of possible candidates, please look at the documentation of
/// [super::trait_goals::CandidateSource] and [super::project_goals::CandidateSource].
#[derive(Debug, Clone)]
pub(super) struct Candidate<'tcx, G: GoalKind<'tcx>> {
    pub(super) source: G::CandidateSource,
    pub(super) result: CanonicalResponse<'tcx>,
}

pub(super) trait GoalKind<'tcx>: TypeFoldable<'tcx> + Copy {
    type CandidateSource: Debug + Copy;

    fn self_ty(self) -> Ty<'tcx>;

    fn with_self_ty(self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> Self;

    fn trait_def_id(self, tcx: TyCtxt<'tcx>) -> DefId;

    fn consider_impl_candidate(
        acx: &mut AssemblyCtxt<'_, 'tcx, Self>,
        goal: Goal<'tcx, Self>,
        impl_def_id: DefId,
    );
}

/// An abstraction which correctly deals with the canonical results for candidates.
///
/// It also deduplicates the behavior between trait and projection predicates.
pub(super) struct AssemblyCtxt<'a, 'tcx, G: GoalKind<'tcx>> {
    pub(super) cx: &'a mut EvalCtxt<'tcx>,
    pub(super) infcx: &'a InferCtxt<'tcx>,
    var_values: CanonicalVarValues<'tcx>,
    candidates: Vec<Candidate<'tcx, G>>,
}

impl<'a, 'tcx, G: GoalKind<'tcx>> AssemblyCtxt<'a, 'tcx, G> {
    pub(super) fn assemble_and_evaluate_candidates(
        cx: &'a mut EvalCtxt<'tcx>,
        goal: CanonicalGoal<'tcx, G>,
    ) -> Vec<Candidate<'tcx, G>> {
        let (ref infcx, goal, var_values) =
            cx.tcx.infer_ctxt().build_with_canonical(DUMMY_SP, &goal);
        let mut acx = AssemblyCtxt { cx, infcx, var_values, candidates: Vec::new() };

        acx.assemble_candidates_after_normalizing_self_ty(goal);

        acx.assemble_impl_candidates(goal);

        acx.candidates
    }

    pub(super) fn try_insert_candidate(
        &mut self,
        source: G::CandidateSource,
        certainty: Certainty,
    ) {
        match self.infcx.make_canonical_response(self.var_values.clone(), certainty) {
            Ok(result) => self.candidates.push(Candidate { source, result }),
            Err(NoSolution) => debug!(?source, ?certainty, "failed leakcheck"),
        }
    }

    /// If the self type of a goal is a projection, computing the relevant candidates is difficult.
    ///
    /// To deal with this, we first try to normalize the self type and add the candidates for the normalized
    /// self type to the list of candidates in case that succeeds. Note that we can't just eagerly return in
    /// this case as projections as self types add `
    fn assemble_candidates_after_normalizing_self_ty(&mut self, goal: Goal<'tcx, G>) {
        let tcx = self.cx.tcx;
        // FIXME: We also have to normalize opaque types, not sure where to best fit that in.
        let &ty::Alias(ty::Projection, projection_ty) = goal.predicate.self_ty().kind() else {
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
            let normalized_candidates =
                AssemblyCtxt::assemble_and_evaluate_candidates(self.cx, goal);

            // Map each candidate from being canonical wrt the current inference context to being
            // canonical wrt the caller.
            for Candidate { source, result } in normalized_candidates {
                self.infcx.probe(|_| {
                    let candidate_certainty = fixme_instantiate_canonical_query_response(
                        &self.infcx,
                        &orig_values,
                        result,
                    );

                    // FIXME: This is a bit scary if the `normalizes_to_goal` overflows.
                    //
                    // If we have an ambiguous candidate it hides that normalization
                    // caused an overflow which may cause issues.
                    self.try_insert_candidate(
                        source,
                        normalization_certainty.unify_and(candidate_certainty),
                    )
                })
            }
        })
    }

    fn assemble_impl_candidates(&mut self, goal: Goal<'tcx, G>) {
        self.cx.tcx.for_each_relevant_impl(
            goal.predicate.trait_def_id(self.cx.tcx),
            goal.predicate.self_ty(),
            |impl_def_id| G::consider_impl_candidate(self, goal, impl_def_id),
        );
    }
}
