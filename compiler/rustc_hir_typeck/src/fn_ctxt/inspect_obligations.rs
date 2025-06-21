//! A utility module to inspect currently ambiguous obligations in the current context.

use rustc_data_structures::unord::UnordSet;
use rustc_hir::def_id::DefId;
use rustc_infer::traits::{self, ObligationCause, PredicateObligations};
use rustc_middle::traits::solve::GoalSource;
use rustc_middle::ty::{self, Ty, TypeVisitableExt};
use rustc_span::Span;
use rustc_trait_selection::solve::Certainty;
use rustc_trait_selection::solve::inspect::{
    InspectConfig, InspectGoal, ProofTreeInferCtxtExt, ProofTreeVisitor,
};
use tracing::{debug, instrument, trace};

use crate::FnCtxt;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Returns a list of all obligations whose self type has been unified
    /// with the unconstrained type `self_ty`.
    #[instrument(skip(self), level = "debug")]
    pub(crate) fn obligations_for_self_ty(&self, self_ty: ty::TyVid) -> PredicateObligations<'tcx> {
        if self.next_trait_solver() {
            self.obligations_for_self_ty_next(self_ty)
        } else {
            let ty_var_root = self.root_var(self_ty);
            let mut obligations = self.fulfillment_cx.borrow().pending_obligations();
            trace!("pending_obligations = {:#?}", obligations);
            obligations
                .retain(|obligation| self.predicate_has_self_ty(obligation.predicate, ty_var_root));
            obligations
        }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn predicate_has_self_ty(
        &self,
        predicate: ty::Predicate<'tcx>,
        expected_vid: ty::TyVid,
    ) -> bool {
        match predicate.kind().skip_binder() {
            ty::PredicateKind::Clause(ty::ClauseKind::Trait(data)) => {
                self.type_matches_expected_vid(expected_vid, data.self_ty())
            }
            ty::PredicateKind::Clause(ty::ClauseKind::Projection(data)) => {
                self.type_matches_expected_vid(expected_vid, data.projection_term.self_ty())
            }
            ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(..))
            | ty::PredicateKind::Subtype(..)
            | ty::PredicateKind::Coerce(..)
            | ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(..))
            | ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(..))
            | ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(..))
            | ty::PredicateKind::DynCompatible(..)
            | ty::PredicateKind::NormalizesTo(..)
            | ty::PredicateKind::AliasRelate(..)
            | ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(..))
            | ty::PredicateKind::ConstEquate(..)
            | ty::PredicateKind::Clause(ty::ClauseKind::HostEffect(..))
            | ty::PredicateKind::Ambiguous => false,
        }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn type_matches_expected_vid(&self, expected_vid: ty::TyVid, ty: Ty<'tcx>) -> bool {
        let ty = self.shallow_resolve(ty);
        debug!(?ty);

        match *ty.kind() {
            ty::Infer(ty::TyVar(found_vid)) => {
                self.root_var(expected_vid) == self.root_var(found_vid)
            }
            _ => false,
        }
    }

    pub(crate) fn obligations_for_self_ty_next(
        &self,
        self_ty: ty::TyVid,
    ) -> PredicateObligations<'tcx> {
        let obligations = self.fulfillment_cx.borrow().pending_obligations();
        debug!(?obligations);
        let mut obligations_for_self_ty = PredicateObligations::new();
        for obligation in obligations {
            let mut visitor = NestedObligationsForSelfTy {
                fcx: self,
                self_ty,
                obligations_for_self_ty: &mut obligations_for_self_ty,
                root_cause: &obligation.cause,
            };

            let goal = obligation.as_goal();
            self.visit_proof_tree(goal, &mut visitor);
        }

        obligations_for_self_ty.retain_mut(|obligation| {
            obligation.predicate = self.resolve_vars_if_possible(obligation.predicate);
            !obligation.predicate.has_placeholders()
        });
        obligations_for_self_ty
    }

    /// Only needed for the `From<{float}>` for `f32` type fallback.
    #[instrument(skip(self), level = "debug")]
    pub(crate) fn from_float_for_f32_root_vids(&self) -> UnordSet<ty::FloatVid> {
        if self.next_trait_solver() {
            self.from_float_for_f32_root_vids_next()
        } else {
            let Some(from_trait) = self.tcx.lang_items().from_trait() else {
                return UnordSet::new();
            };
            self.fulfillment_cx
                .borrow_mut()
                .pending_obligations()
                .into_iter()
                .filter_map(|obligation| {
                    self.predicate_from_float_for_f32_root_vid(from_trait, obligation.predicate)
                })
                .collect()
        }
    }

    fn predicate_from_float_for_f32_root_vid(
        &self,
        from_trait: DefId,
        predicate: ty::Predicate<'tcx>,
    ) -> Option<ty::FloatVid> {
        // The predicates we are looking for look like
        // `TraitPredicate(<f32 as std::convert::From<{float}>>, polarity:Positive)`.
        // They will have no bound variables.
        match predicate.kind().no_bound_vars() {
            Some(ty::PredicateKind::Clause(ty::ClauseKind::Trait(ty::TraitPredicate {
                polarity: ty::PredicatePolarity::Positive,
                trait_ref,
            }))) if trait_ref.def_id == from_trait
                && self.shallow_resolve(trait_ref.self_ty()).kind()
                    == &ty::Float(ty::FloatTy::F32) =>
            {
                self.root_float_vid(trait_ref.args.type_at(1))
            }
            _ => None,
        }
    }

    fn from_float_for_f32_root_vids_next(&self) -> UnordSet<ty::FloatVid> {
        let Some(from_trait) = self.tcx.lang_items().from_trait() else {
            return UnordSet::new();
        };
        let obligations = self.fulfillment_cx.borrow().pending_obligations();
        debug!(?obligations);
        let mut vids = UnordSet::new();
        for obligation in obligations {
            let mut visitor = FindFromFloatForF32RootVids {
                fcx: self,
                from_trait,
                vids: &mut vids,
                span: obligation.cause.span,
            };

            let goal = obligation.as_goal();
            self.visit_proof_tree(goal, &mut visitor);
        }
        vids
    }
}

struct NestedObligationsForSelfTy<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    self_ty: ty::TyVid,
    root_cause: &'a ObligationCause<'tcx>,
    obligations_for_self_ty: &'a mut PredicateObligations<'tcx>,
}

impl<'tcx> ProofTreeVisitor<'tcx> for NestedObligationsForSelfTy<'_, 'tcx> {
    fn span(&self) -> Span {
        self.root_cause.span
    }

    fn config(&self) -> InspectConfig {
        // Using an intentionally low depth to minimize the chance of future
        // breaking changes in case we adapt the approach later on. This also
        // avoids any hangs for exponentially growing proof trees.
        InspectConfig { max_depth: 5 }
    }

    fn visit_goal(&mut self, inspect_goal: &InspectGoal<'_, 'tcx>) {
        // No need to walk into goal subtrees that certainly hold, since they
        // wouldn't then be stalled on an infer var.
        if inspect_goal.result() == Ok(Certainty::Yes) {
            return;
        }

        let tcx = self.fcx.tcx;
        let goal = inspect_goal.goal();
        if self.fcx.predicate_has_self_ty(goal.predicate, self.self_ty)
            // We do not push the instantiated forms of goals as it would cause any
            // aliases referencing bound vars to go from having escaping bound vars to
            // being able to be normalized to an inference variable.
            //
            // This is mostly just a hack as arbitrary nested goals could still contain
            // such aliases while having a different `GoalSource`. Closure signature inference
            // however can't really handle *every* higher ranked `Fn` goal also being present
            // in the form of `?c: Fn<(<?x as Trait<'!a>>::Assoc)`.
            //
            // This also just better matches the behaviour of the old solver where we do not
            // encounter instantiated forms of goals, only nested goals that referred to bound
            // vars from instantiated goals.
            && !matches!(inspect_goal.source(), GoalSource::InstantiateHigherRanked)
        {
            self.obligations_for_self_ty.push(traits::Obligation::new(
                tcx,
                self.root_cause.clone(),
                goal.param_env,
                goal.predicate,
            ));
        }

        // If there's a unique way to prove a given goal, recurse into
        // that candidate. This means that for `impl<F: FnOnce(u32)> Trait<F> for () {}`
        // and a `(): Trait<?0>` goal we recurse into the impl and look at
        // the nested `?0: FnOnce(u32)` goal.
        if let Some(candidate) = inspect_goal.unique_applicable_candidate() {
            candidate.visit_nested_no_probe(self)
        }
    }
}

struct FindFromFloatForF32RootVids<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    from_trait: DefId,
    vids: &'a mut UnordSet<ty::FloatVid>,
    span: Span,
}

impl<'tcx> ProofTreeVisitor<'tcx> for FindFromFloatForF32RootVids<'_, 'tcx> {
    fn span(&self) -> Span {
        self.span
    }

    fn config(&self) -> InspectConfig {
        // Avoid hang from exponentially growing proof trees (see `cycle-modulo-ambig-aliases.rs`).
        // 3 is more than enough for all occurences in practice (a.k.a. `Into`).
        InspectConfig { max_depth: 3 }
    }

    fn visit_goal(&mut self, inspect_goal: &InspectGoal<'_, 'tcx>) {
        if let Some(vid) = self
            .fcx
            .predicate_from_float_for_f32_root_vid(self.from_trait, inspect_goal.goal().predicate)
        {
            self.vids.insert(vid);
        } else if let Some(candidate) = inspect_goal.unique_applicable_candidate() {
            let start_len = self.vids.len();
            let _ = candidate.goal().infcx().commit_if_ok(|_| {
                candidate.visit_nested_no_probe(self);
                if self.vids.len() > start_len { Ok(()) } else { Err(()) }
            });
        }
    }
}
