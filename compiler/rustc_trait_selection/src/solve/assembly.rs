//! Code shared by trait and projection goals for candidate assembly.

use super::infcx_ext::InferCtxtExt;
use super::{CanonicalResponse, Certainty, EvalCtxt, Goal};
use rustc_hir::def_id::DefId;
use rustc_infer::traits::query::NoSolution;
use rustc_middle::ty::TypeFoldable;
use rustc_middle::ty::{self, Ty, TyCtxt};
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
        acx: &mut AssemblyCtxt<'_, '_, 'tcx, Self>,
        goal: Goal<'tcx, Self>,
        impl_def_id: DefId,
    );
}

/// An abstraction which correctly deals with the canonical results for candidates.
///
/// It also deduplicates the behavior between trait and projection predicates.
pub(super) struct AssemblyCtxt<'a, 'b, 'tcx, G: GoalKind<'tcx>> {
    pub(super) cx: &'a mut EvalCtxt<'b, 'tcx>,
    candidates: Vec<Candidate<'tcx, G>>,
}

impl<'a, 'b, 'tcx, G: GoalKind<'tcx>> AssemblyCtxt<'a, 'b, 'tcx, G> {
    pub(super) fn assemble_and_evaluate_candidates(
        cx: &'a mut EvalCtxt<'b, 'tcx>,
        goal: Goal<'tcx, G>,
    ) -> Vec<Candidate<'tcx, G>> {
        let mut acx = AssemblyCtxt { cx, candidates: Vec::new() };

        acx.assemble_candidates_after_normalizing_self_ty(goal);

        acx.assemble_impl_candidates(goal);

        acx.candidates
    }

    pub(super) fn try_insert_candidate(
        &mut self,
        source: G::CandidateSource,
        certainty: Certainty,
    ) {
        match self.cx.make_canonical_response(certainty) {
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
        let tcx = self.cx.tcx();
        let infcx = self.cx.infcx;
        // FIXME: We also have to normalize opaque types, not sure where to best fit that in.
        let &ty::Alias(ty::Projection, projection_ty) = goal.predicate.self_ty().kind() else {
            return
        };
        infcx.probe(|_| {
            let normalized_ty = infcx.next_ty_infer();
            let normalizes_to_goal = goal.with(
                tcx,
                ty::Binder::dummy(ty::ProjectionPredicate {
                    projection_ty,
                    term: normalized_ty.into(),
                }),
            );
            let normalization_certainty = match self.cx.evaluate_goal(normalizes_to_goal) {
                Ok((_, certainty)) => certainty,
                Err(NoSolution) => return,
            };

            // NOTE: Alternatively we could call `evaluate_goal` here and only have a `Normalized` candidate.
            // This doesn't work as long as we use `CandidateSource` in both winnowing and to resolve associated items.
            let goal = goal.with(tcx, goal.predicate.with_self_ty(tcx, normalized_ty));
            let normalized_candidates =
                AssemblyCtxt::assemble_and_evaluate_candidates(self.cx, goal);
            for mut normalized_candidate in normalized_candidates {
                normalized_candidate.result =
                    normalized_candidate.result.unchecked_map(|mut response| {
                        response.certainty = response.certainty.unify_and(normalization_certainty);
                        response
                    });
                self.candidates.push(normalized_candidate);
            }
        })
    }

    fn assemble_impl_candidates(&mut self, goal: Goal<'tcx, G>) {
        let tcx = self.cx.tcx();
        tcx.for_each_relevant_impl(
            goal.predicate.trait_def_id(tcx),
            goal.predicate.self_ty(),
            |impl_def_id| G::consider_impl_candidate(self, goal, impl_def_id),
        );
    }
}
