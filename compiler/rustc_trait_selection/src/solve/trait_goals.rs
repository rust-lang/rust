//! Dealing with trait goals, i.e. `T: Trait<'a, U>`.

use std::iter;

use super::assembly::{self, AssemblyCtxt};
use super::{CanonicalGoal, Certainty, EvalCtxt, Goal, QueryResult};
use rustc_hir::def_id::DefId;
use rustc_infer::infer::{InferOk, LateBoundRegionConversionTime};
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::util::supertraits;
use rustc_infer::traits::ObligationCause;
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams};
use rustc_middle::ty::TraitPredicate;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::DUMMY_SP;

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
    /// Implementation of `Trait` or its supertraits for a `dyn Trait + Send + Sync`.
    ObjectBound(usize),
    /// Implementation of `Send` or other explicitly listed *auto* traits for
    /// a `dyn Trait + Send + Sync`
    ObjectAutoBound,
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

type Candidate<'tcx> = assembly::Candidate<'tcx, TraitPredicate<'tcx>>;

impl<'tcx> assembly::GoalKind<'tcx> for TraitPredicate<'tcx> {
    type CandidateSource = CandidateSource;

    fn self_ty(self) -> Ty<'tcx> {
        self.self_ty()
    }

    fn with_self_ty(self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> Self {
        self.with_self_ty(tcx, self_ty)
    }

    fn trait_def_id(self, _: TyCtxt<'tcx>) -> DefId {
        self.def_id()
    }

    fn consider_impl_candidate(
        acx: &mut AssemblyCtxt<'_, 'tcx, Self>,
        goal: Goal<'tcx, Self>,
        impl_def_id: DefId,
    ) {
        let tcx = acx.cx.tcx;

        let impl_trait_ref = tcx.bound_impl_trait_ref(impl_def_id).unwrap();
        let drcx = DeepRejectCtxt { treat_obligation_params: TreatParams::AsPlaceholder };
        if iter::zip(goal.predicate.trait_ref.substs, impl_trait_ref.skip_binder().substs)
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
                .eq(goal.predicate.trait_ref, impl_trait_ref)
                .map_err(|e| debug!("failed to equate trait refs: {e:?}"))
            else {
                return
            };
            let where_clause_bounds = tcx
                .predicates_of(impl_def_id)
                .instantiate(tcx, impl_substs)
                .predicates
                .into_iter()
                .map(|pred| goal.with(tcx, pred));

            let nested_goals =
                obligations.into_iter().map(|o| o.into()).chain(where_clause_bounds).collect();

            let Ok(certainty) = acx.cx.evaluate_all(acx.infcx, nested_goals) else { return };
            acx.try_insert_candidate(CandidateSource::Impl(impl_def_id), certainty);
        })
    }

    fn consider_alias_bound_candidates(
        acx: &mut AssemblyCtxt<'_, 'tcx, Self>,
        goal: Goal<'tcx, Self>,
        alias_ty: ty::AliasTy<'tcx>,
    ) {
        for (idx, (predicate, _)) in acx
            .cx
            .tcx
            .bound_explicit_item_bounds(alias_ty.def_id)
            .subst_iter_copied(acx.cx.tcx, alias_ty.substs)
            .enumerate()
        {
            let Some(poly_trait_pred) = predicate.to_opt_poly_trait_pred() else { continue };
            if poly_trait_pred.skip_binder().def_id() != goal.predicate.def_id() {
                continue;
            };
            // FIXME: constness? polarity?
            let poly_trait_ref = poly_trait_pred.map_bound(|trait_pred| trait_pred.trait_ref);
            // FIXME: Faster to do a filter first with a rejection context?
            match_poly_trait_ref_against_goal(
                acx,
                goal,
                poly_trait_ref,
                CandidateSource::AliasBound(idx),
            );
        }
    }

    fn consider_object_bound_candidates(
        acx: &mut AssemblyCtxt<'_, 'tcx, Self>,
        goal: Goal<'tcx, Self>,
        object_bounds: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    ) {
        if let Some(principal_trait_ref) = object_bounds.principal() {
            let principal_trait_ref =
                principal_trait_ref.with_self_ty(acx.cx.tcx, goal.predicate.self_ty());

            for (idx, poly_trait_ref) in supertraits(acx.cx.tcx, principal_trait_ref).enumerate() {
                if poly_trait_ref.skip_binder().def_id != goal.predicate.def_id() {
                    continue;
                };
                match_poly_trait_ref_against_goal(
                    acx,
                    goal,
                    poly_trait_ref,
                    CandidateSource::ObjectBound(idx),
                );
            }
        }

        if object_bounds.auto_traits().any(|def_id| def_id == goal.predicate.def_id()) {
            acx.try_insert_candidate(CandidateSource::ObjectAutoBound, Certainty::Yes);
        }
    }
}

fn match_poly_trait_ref_against_goal<'tcx>(
    acx: &mut AssemblyCtxt<'_, 'tcx, TraitPredicate<'tcx>>,
    goal: Goal<'tcx, TraitPredicate<'tcx>>,
    trait_ref: ty::PolyTraitRef<'tcx>,
    candidate: impl FnOnce() -> CandidateSource,
) {
    acx.infcx.probe(|_| {
        let trait_ref = acx.infcx.replace_bound_vars_with_fresh_vars(
            DUMMY_SP,
            LateBoundRegionConversionTime::HigherRankedType,
            trait_ref,
        );

        let Ok(InferOk { obligations, .. }) = acx
            .infcx
            .at(&ObligationCause::dummy(), goal.param_env)
            .define_opaque_types(false)
            .sup(goal.predicate.trait_ref, trait_ref)
            .map_err(|e| debug!("failed to equate trait refs: {e:?}"))
        else {
            return
        };

        let nested_goals = obligations.into_iter().map(|o| o.into()).collect();

        let Ok(certainty) = acx.cx.evaluate_all(acx.infcx, nested_goals) else { return };
        acx.try_insert_candidate(candidate(), certainty);
    })
}

impl<'tcx> EvalCtxt<'tcx> {
    pub(super) fn compute_trait_goal(
        &mut self,
        goal: CanonicalGoal<'tcx, TraitPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        let candidates = AssemblyCtxt::assemble_and_evaluate_candidates(self, goal);
        self.merge_trait_candidates_discard_reservation_impls(candidates)
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
            | (CandidateSource::ObjectBound(_), _)
            | (CandidateSource::ObjectAutoBound, _)
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
