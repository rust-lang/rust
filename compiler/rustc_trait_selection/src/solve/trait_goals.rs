//! Dealing with trait goals, i.e. `T: Trait<'a, U>`.

use std::iter;

use super::infcx_ext::InferCtxtExt;
use super::{
    fixme_instantiate_canonical_query_response, CanonicalGoal, CanonicalResponse, Certainty,
    EvalCtxt, Goal, QueryResult,
};
use rustc_hir::def_id::DefId;
use rustc_infer::infer::canonical::{CanonicalVarValues, OriginalQueryValues};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::infer::{InferCtxt, InferOk};
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::ObligationCause;
use rustc_middle::ty;
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams};
use rustc_middle::ty::TraitPredicate;
use rustc_span::DUMMY_SP;

/// A candidate is a possible way to prove a goal.
///
/// It consists of both the `source`, which describes how that goal
/// would be proven, and the `result` when using the given `source`.
///
/// For the list of possible candidates, please look at the documentation
/// of [CandidateSource].
#[derive(Debug, Clone)]
pub(super) struct Candidate<'tcx> {
    source: CandidateSource,
    result: CanonicalResponse<'tcx>,
}

#[allow(dead_code)] // FIXME: implement and use all variants.
#[derive(Debug, Clone, Copy)]
pub(super) enum CandidateSource {
    /// Some user-defined impl with the given `DefId`.
    Impl(DefId),
    /// The n-th caller bound in the `param_env` of our goal.
    ///
    /// This is pretty much always a bound from the `where`-clauses of the
    /// currently checked item.
    ParamEnv(usize),
    /// A bound on the `self_ty` in case it is a projection or an opaque type.
    ///
    /// # Examples
    ///
    /// ```ignore (for syntax highlighting)
    /// trait Trait {
    ///     type Assoc: OtherTrait;
    /// }
    /// ```
    ///
    /// We know that `<Whatever as Trait>::Assoc: OtherTrait` holds by looking at
    /// the bounds on `Trait::Assoc`.
    AliasBound(usize),
    /// A builtin implementation for some specific traits, used in cases
    /// where we cannot rely an ordinary library implementations.
    ///
    /// The most notable examples are `Sized`, `Copy` and `Clone`. This is also
    /// used for the `DiscriminantKind` and `Pointee` trait, both of which have
    /// an associated type.
    Builtin,
    /// An automatic impl for an auto trait, e.g. `Send`. These impls recursively look
    /// at the constituent types of the `self_ty` to check whether the auto trait
    /// is implemented for those.
    AutoImpl,
}

struct AssemblyCtxt<'a, 'tcx> {
    cx: &'a mut EvalCtxt<'tcx>,
    infcx: &'a InferCtxt<'tcx>,
    var_values: CanonicalVarValues<'tcx>,
    candidates: Vec<Candidate<'tcx>>,
}

impl<'tcx> EvalCtxt<'tcx> {
    pub(super) fn compute_trait_goal(
        &mut self,
        goal: CanonicalGoal<'tcx, TraitPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        let candidates = self.assemble_and_evaluate_trait_candidates(goal);
        self.merge_trait_candidates_discard_reservation_impls(candidates)
    }

    pub(super) fn assemble_and_evaluate_trait_candidates(
        &mut self,
        goal: CanonicalGoal<'tcx, TraitPredicate<'tcx>>,
    ) -> Vec<Candidate<'tcx>> {
        let (ref infcx, goal, var_values) =
            self.tcx.infer_ctxt().build_with_canonical(DUMMY_SP, &goal);
        let mut acx = AssemblyCtxt { cx: self, infcx, var_values, candidates: Vec::new() };

        acx.assemble_candidates_after_normalizing_self_ty(goal);
        acx.assemble_impl_candidates(goal);

        // FIXME: Remaining candidates
        acx.candidates
    }

    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn merge_trait_candidates_discard_reservation_impls(
        &mut self,
        mut candidates: Vec<Candidate<'tcx>>,
    ) -> QueryResult<'tcx> {
        match candidates.len() {
            0 => return Err(NoSolution),
            1 => return Ok(self.discard_reservation_impl(candidates.pop().unwrap()).result),
            _ => {}
        }

        if candidates.len() > 1 {
            let mut i = 0;
            'outer: while i < candidates.len() {
                for j in (0..candidates.len()).filter(|&j| i != j) {
                    if self.trait_candidate_should_be_dropped_in_favor_of(
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

        Ok(self.discard_reservation_impl(candidates.pop().unwrap()).result)
    }

    fn trait_candidate_should_be_dropped_in_favor_of(
        &self,
        candidate: &Candidate<'tcx>,
        other: &Candidate<'tcx>,
    ) -> bool {
        // FIXME: implement this
        match (candidate.source, other.source) {
            (CandidateSource::Impl(_), _)
            | (CandidateSource::ParamEnv(_), _)
            | (CandidateSource::AliasBound(_), _)
            | (CandidateSource::Builtin, _)
            | (CandidateSource::AutoImpl, _) => unimplemented!(),
        }
    }

    fn discard_reservation_impl(&self, candidate: Candidate<'tcx>) -> Candidate<'tcx> {
        if let CandidateSource::Impl(def_id) = candidate.source {
            if let ty::ImplPolarity::Reservation = self.tcx.impl_polarity(def_id) {
                debug!("Selected reservation impl");
                // FIXME: reduce candidate to ambiguous
                // FIXME: replace `var_values` with identity, yeet external constraints.
                unimplemented!()
            }
        }

        candidate
    }
}

impl<'tcx> AssemblyCtxt<'_, 'tcx> {
    /// Adds a new candidate using the current state of the inference context.
    ///
    /// This does require each assembly method to correctly use `probe` to not taint
    /// the results of other candidates.
    fn try_insert_candidate(&mut self, source: CandidateSource, certainty: Certainty) {
        match self.infcx.make_canonical_response(self.var_values.clone(), certainty) {
            Ok(result) => self.candidates.push(Candidate { source, result }),
            Err(NoSolution) => debug!(?source, ?certainty, "failed leakcheck"),
        }
    }

    /// If the self type of a trait goal is a projection, computing the relevant candidates is difficult.
    ///
    /// To deal with this, we first try to normalize the self type and add the candidates for the normalized
    /// self type to the list of candidates in case that succeeds. Note that we can't just eagerly return in
    /// this case as projections as self types add `
    fn assemble_candidates_after_normalizing_self_ty(
        &mut self,
        goal: Goal<'tcx, TraitPredicate<'tcx>>,
    ) {
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
            let goal = goal.with(tcx, goal.predicate.with_self_type(tcx, normalized_ty));
            let mut orig_values = OriginalQueryValues::default();
            let goal = self.infcx.canonicalize_query(goal, &mut orig_values);
            let normalized_candidates = self.cx.assemble_and_evaluate_trait_candidates(goal);

            // Map each candidate from being canonical wrt the current inference context to being
            // canonical wrt the caller.
            for Candidate { source, result } in normalized_candidates {
                self.infcx.probe(|_| {
                    let candidate_certainty = fixme_instantiate_canonical_query_response(
                        self.infcx,
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

    fn assemble_impl_candidates(&mut self, goal: Goal<'tcx, TraitPredicate<'tcx>>) {
        self.cx.tcx.for_each_relevant_impl(
            goal.predicate.def_id(),
            goal.predicate.self_ty(),
            |impl_def_id| self.consider_impl_candidate(goal, impl_def_id),
        );
    }

    fn consider_impl_candidate(
        &mut self,
        goal: Goal<'tcx, TraitPredicate<'tcx>>,
        impl_def_id: DefId,
    ) {
        let impl_trait_ref = self.cx.tcx.bound_impl_trait_ref(impl_def_id).unwrap();
        let drcx = DeepRejectCtxt { treat_obligation_params: TreatParams::AsPlaceholder };
        if iter::zip(goal.predicate.trait_ref.substs, impl_trait_ref.skip_binder().substs)
            .any(|(goal, imp)| !drcx.generic_args_may_unify(goal, imp))
        {
            return;
        }

        self.infcx.probe(|_| {
            let impl_substs = self.infcx.fresh_substs_for_item(DUMMY_SP, impl_def_id);
            let impl_trait_ref = impl_trait_ref.subst(self.cx.tcx, impl_substs);

            let Ok(InferOk { obligations, .. }) = self
                .infcx
                .at(&ObligationCause::dummy(), goal.param_env)
                .define_opaque_types(false)
                .eq(goal.predicate.trait_ref, impl_trait_ref)
                .map_err(|e| debug!("failed to equate trait refs: {e:?}"))
            else {
                return
            };

            let nested_goals = obligations.into_iter().map(|o| o.into()).collect();

            let Ok(certainty) = self.cx.evaluate_all(self.infcx, nested_goals) else { return };
            self.try_insert_candidate(CandidateSource::Impl(impl_def_id), certainty);
        })
    }
}
