use crate::traits::{specialization_graph, translate_substs};

use super::assembly::{self, Candidate, CandidateSource};
use super::infcx_ext::InferCtxtExt;
use super::{Certainty, EvalCtxt, Goal, MaybeCause, QueryResult};
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::specialization_graph::LeafDef;
use rustc_infer::traits::Reveal;
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams};
use rustc_middle::ty::TypeVisitable;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{ProjectionPredicate, TypeSuperVisitable, TypeVisitor};
use rustc_span::DUMMY_SP;
use std::iter;
use std::ops::ControlFlow;

impl<'tcx> EvalCtxt<'_, 'tcx> {
    pub(super) fn compute_projection_goal(
        &mut self,
        goal: Goal<'tcx, ProjectionPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        // To only compute normalization ones for each projection we only
        // normalize if the expected term is an unconstrained inference variable.
        //
        // E.g. for `<T as Trait>::Assoc = u32` we recursively compute the goal
        // `exists<U> <T as Trait>::Assoc = U` and then take the resulting type for
        // `U` and equate it with `u32`. This means that we don't need a separate
        // projection cache in the solver.
        if self.term_is_fully_unconstrained(goal) {
            let candidates = self.assemble_and_evaluate_candidates(goal);
            self.merge_project_candidates(candidates)
        } else {
            let predicate = goal.predicate;
            let unconstrained_rhs = match predicate.term.unpack() {
                ty::TermKind::Ty(_) => self.infcx.next_ty_infer().into(),
                ty::TermKind::Const(ct) => self.infcx.next_const_infer(ct.ty()).into(),
            };
            let unconstrained_predicate = ty::Clause::Projection(ProjectionPredicate {
                projection_ty: goal.predicate.projection_ty,
                term: unconstrained_rhs,
            });
            let (_has_changed, normalize_certainty) =
                self.evaluate_goal(goal.with(self.tcx(), unconstrained_predicate))?;

            let nested_eq_goals =
                self.infcx.eq(goal.param_env, unconstrained_rhs, predicate.term)?;
            let eval_certainty = self.evaluate_all(nested_eq_goals)?;
            self.make_canonical_response(normalize_certainty.unify_and(eval_certainty))
        }
    }

    /// Is the projection predicate is of the form `exists<T> <Ty as Trait>::Assoc = T`.
    ///
    /// This is the case if the `term` is an inference variable in the innermost universe
    /// and does not occur in any other part of the predicate.
    fn term_is_fully_unconstrained(&self, goal: Goal<'tcx, ProjectionPredicate<'tcx>>) -> bool {
        let infcx = self.infcx;
        let term_is_infer = match goal.predicate.term.unpack() {
            ty::TermKind::Ty(ty) => {
                if let &ty::Infer(ty::TyVar(vid)) = ty.kind() {
                    match infcx.probe_ty_var(vid) {
                        Ok(value) => bug!("resolved var in query: {goal:?} {value:?}"),
                        Err(universe) => universe == infcx.universe(),
                    }
                } else {
                    false
                }
            }
            ty::TermKind::Const(ct) => {
                if let ty::ConstKind::Infer(ty::InferConst::Var(vid)) = ct.kind() {
                    match self.infcx.probe_const_var(vid) {
                        Ok(value) => bug!("resolved var in query: {goal:?} {value:?}"),
                        Err(universe) => universe == infcx.universe(),
                    }
                } else {
                    false
                }
            }
        };

        // Guard against `<T as Trait<?0>>::Assoc = ?0>`.
        struct ContainsTerm<'tcx> {
            term: ty::Term<'tcx>,
        }
        impl<'tcx> TypeVisitor<'tcx> for ContainsTerm<'tcx> {
            type BreakTy = ();
            fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
                if t.needs_infer() {
                    if ty::Term::from(t) == self.term {
                        ControlFlow::BREAK
                    } else {
                        t.super_visit_with(self)
                    }
                } else {
                    ControlFlow::CONTINUE
                }
            }

            fn visit_const(&mut self, c: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
                if c.needs_infer() {
                    if ty::Term::from(c) == self.term {
                        ControlFlow::BREAK
                    } else {
                        c.super_visit_with(self)
                    }
                } else {
                    ControlFlow::CONTINUE
                }
            }
        }

        let mut visitor = ContainsTerm { term: goal.predicate.term };

        term_is_infer
            && goal.predicate.projection_ty.visit_with(&mut visitor).is_continue()
            && goal.param_env.visit_with(&mut visitor).is_continue()
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
            | (CandidateSource::BuiltinImpl, _)
            | (CandidateSource::AliasBound(_), _) => unimplemented!(),
        }
    }
}

impl<'tcx> assembly::GoalKind<'tcx> for ProjectionPredicate<'tcx> {
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
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, ProjectionPredicate<'tcx>>,
        impl_def_id: DefId,
    ) -> Result<Certainty, NoSolution> {
        let tcx = ecx.tcx();

        let goal_trait_ref = goal.predicate.projection_ty.trait_ref(tcx);
        let impl_trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
        let drcx = DeepRejectCtxt { treat_obligation_params: TreatParams::AsPlaceholder };
        if iter::zip(goal_trait_ref.substs, impl_trait_ref.skip_binder().substs)
            .any(|(goal, imp)| !drcx.generic_args_may_unify(goal, imp))
        {
            return Err(NoSolution);
        }

        ecx.infcx.probe(|_| {
            let impl_substs = ecx.infcx.fresh_substs_for_item(DUMMY_SP, impl_def_id);
            let impl_trait_ref = impl_trait_ref.subst(tcx, impl_substs);

            let mut nested_goals = ecx.infcx.eq(goal.param_env, goal_trait_ref, impl_trait_ref)?;
            let where_clause_bounds = tcx
                .predicates_of(impl_def_id)
                .instantiate(tcx, impl_substs)
                .predicates
                .into_iter()
                .map(|pred| goal.with(tcx, pred));

            nested_goals.extend(where_clause_bounds);
            let trait_ref_certainty = ecx.evaluate_all(nested_goals)?;

            // In case the associated item is hidden due to specialization, we have to
            // return ambiguity this would otherwise be incomplete, resulting in
            // unsoundness during coherence (#105782).
            let Some(assoc_def) = fetch_eligible_assoc_item_def(
                ecx.infcx,
                goal.param_env,
                goal_trait_ref,
                goal.predicate.def_id(),
                impl_def_id
            )? else {
                let certainty = Certainty::Maybe(MaybeCause::Ambiguity);
                return Ok(trait_ref_certainty.unify_and(certainty));
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
                impl_substs,
            );
            let substs = translate_substs(
                ecx.infcx,
                goal.param_env,
                impl_def_id,
                impl_substs_with_gat,
                assoc_def.defining_node,
            );

            // Finally we construct the actual value of the associated type.
            let is_const = matches!(tcx.def_kind(assoc_def.item.def_id), DefKind::AssocConst);
            let ty = tcx.bound_type_of(assoc_def.item.def_id);
            let term: ty::EarlyBinder<ty::Term<'tcx>> = if is_const {
                let identity_substs =
                    ty::InternalSubsts::identity_for_item(tcx, assoc_def.item.def_id);
                let did = ty::WithOptConstParam::unknown(assoc_def.item.def_id);
                let kind =
                    ty::ConstKind::Unevaluated(ty::UnevaluatedConst::new(did, identity_substs));
                ty.map_bound(|ty| tcx.mk_const(kind, ty).into())
            } else {
                ty.map_bound(|ty| ty.into())
            };

            // The term of our goal should be fully unconstrained, so this should never fail.
            //
            // It can however be ambiguous when the resolved type is a projection.
            let nested_goals = ecx
                .infcx
                .eq(goal.param_env, goal.predicate.term, term.subst(tcx, substs))
                .expect("failed to unify with unconstrained term");
            let rhs_certainty =
                ecx.evaluate_all(nested_goals).expect("failed to unify with unconstrained term");

            Ok(trait_ref_certainty.unify_and(rhs_certainty))
        })
    }

    fn consider_builtin_sized_candidate(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> Result<Certainty, NoSolution> {
        bug!("`Sized` does not have an associated type: {:?}", goal);
    }

    fn consider_assumption(
        _ecx: &mut EvalCtxt<'_, 'tcx>,
        _goal: Goal<'tcx, Self>,
        assumption: ty::Predicate<'tcx>,
    ) -> Result<Certainty, NoSolution> {
        if let Some(_poly_projection_pred) = assumption.to_opt_poly_projection_pred() {
            unimplemented!()
        } else {
            Err(NoSolution)
        }
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
) -> Result<Option<LeafDef>, NoSolution> {
    let node_item = specialization_graph::assoc_def(infcx.tcx, impl_def_id, trait_assoc_def_id)
        .map_err(|ErrorGuaranteed { .. }| NoSolution)?;

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

    if eligible { Ok(Some(node_item)) } else { Ok(None) }
}
