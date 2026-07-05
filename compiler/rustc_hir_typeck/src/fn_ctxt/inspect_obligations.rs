//! A utility module to inspect currently ambiguous obligations in the current context.

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::unord::UnordSet;
use rustc_hir::def_id::DefId;
use rustc_infer::traits::{self, ObligationCause, PredicateObligations};
use rustc_middle::ty::{self, Ty, TypeVisitableExt};
use rustc_span::Span;
use rustc_trait_selection::solve::Certainty;
use rustc_trait_selection::solve::inspect::{
    InferCtxtProofTreeExt, InspectConfig, InspectGoal, ProofTreeVisitor,
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
                self.type_matches_expected_vid(data.self_ty(), expected_vid)
            }
            ty::PredicateKind::Clause(ty::ClauseKind::Projection(data)) => {
                if data.projection_term.kind.is_trait_projection() {
                    self.type_matches_expected_vid(data.self_ty(), expected_vid)
                } else {
                    false
                }
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
            | ty::PredicateKind::Clause(ty::ClauseKind::UnstableFeature(_))
            | ty::PredicateKind::Ambiguous => false,
        }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn type_matches_expected_vid(&self, ty: Ty<'tcx>, expected_vid: ty::TyVid) -> bool {
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
        // We only look at obligations which may reference the self type.
        // This lookup uses the `sub_root` instead of the inference variable
        // itself as that's slightly nicer to implement. It shouldn't really
        // matter.
        //
        // This is really impactful when typechecking functions with a lot of
        // stalled obligations, e.g. in the `wg-grammar` benchmark.
        let sub_root_var = self.sub_unification_table_root_var(self_ty);
        let obligations = self
            .fulfillment_cx
            .borrow()
            .pending_obligations_potentially_referencing_sub_root(&self.infcx, sub_root_var);
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

    /// When we're about to emit `E0282` for a bare type inference variable `ty`,
    /// find a better variable to blame: the unresolved input of a stalled
    /// associated-type projection whose output is (equivalent to) `ty`.
    ///
    /// E.g. in `v.get(x.into())` with `v: Vec<Foo>`, `get` returns
    /// `Option<&<?I as SliceIndex<[Foo]>>::Output>`. If `?I` (the `.into()` target)
    /// is unknown we get stuck on the output `?Out` at a later use. `?Out` has no
    /// annotatable source but `?I` does, so we blame `?I` instead. See #146126.
    ///
    /// Returns `None` unless `ty` is the output of such a stalled *trait* projection
    /// with a bare inference-variable input. Type variables only (not consts).
    #[instrument(level = "debug", skip(self), ret)]
    pub(crate) fn ambiguous_projection_input(&self, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
        // Follow chains of stalled projections (e.g. `x.into().into()`) for a few
        // hops. The bound keeps this cheap and guarantees termination; every known
        // case needs a single hop.
        const MAX_HOPS: usize = 4;

        let mut current = self.shallow_resolve(ty);
        let mut result = None;

        for _ in 0..MAX_HOPS {
            let ty::Infer(ty::TyVar(cur_vid)) = *current.kind() else { break };

            let obligations = self.fulfillment_cx.borrow().pending_obligations();

            // The stuck variable is often only a `Subtype`/`Coerce` sibling of the
            // projection output, not the output itself. Follow those relations to a
            // fixpoint to collect every variable holding the same value.
            let mut related: FxHashSet<ty::TyVid> = FxHashSet::default();
            related.insert(self.root_var(cur_vid));
            loop {
                let mut changed = false;
                for obligation in &obligations {
                    let predicate = self.resolve_vars_if_possible(obligation.predicate);
                    let Some(kind) = predicate.kind().no_bound_vars() else { continue };
                    let (a, b) = match kind {
                        ty::PredicateKind::Subtype(p) => (p.a, p.b),
                        ty::PredicateKind::Coerce(p) => (p.a, p.b),
                        _ => continue,
                    };
                    if let (ty::Infer(ty::TyVar(av)), ty::Infer(ty::TyVar(bv))) =
                        (self.shallow_resolve(a).kind(), self.shallow_resolve(b).kind())
                    {
                        let (ar, br) = (self.root_var(*av), self.root_var(*bv));
                        if related.contains(&ar) != related.contains(&br) {
                            related.insert(ar);
                            related.insert(br);
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }

            // Find stalled *trait* projections whose output is in `related` and
            // collect their distinct unresolved inputs. (Only trait projections:
            // their input is well-defined; inherent/opaque/free aliases aren't.)
            // Retarget only if exactly one input turns up; otherwise blaming a
            // single one would be arbitrary, so leave the diagnostic unchanged.
            let mut candidate_roots: FxHashSet<ty::TyVid> = FxHashSet::default();
            let mut candidate = None;
            for obligation in &obligations {
                let predicate = self.resolve_vars_if_possible(obligation.predicate);
                let Some(kind) = predicate.kind().no_bound_vars() else { continue };

                let (alias_args, term) = match kind {
                    ty::PredicateKind::Clause(ty::ClauseKind::Projection(pred))
                        if pred.projection_term.kind.is_trait_projection() =>
                    {
                        (pred.projection_term.args, pred.term)
                    }
                    ty::PredicateKind::NormalizesTo(pred)
                        if pred.alias.kind.is_trait_projection() =>
                    {
                        (pred.alias.args, pred.term)
                    }
                    ty::PredicateKind::AliasRelate(lhs, rhs, _) => {
                        if let Some(lhs_ty) = lhs.as_type()
                            && let ty::Alias(_, alias) = *self.shallow_resolve(lhs_ty).kind()
                            && matches!(alias.kind, ty::AliasTyKind::Projection { .. })
                        {
                            (alias.args, rhs)
                        } else if let Some(rhs_ty) = rhs.as_type()
                            && let ty::Alias(_, alias) = *self.shallow_resolve(rhs_ty).kind()
                            && matches!(alias.kind, ty::AliasTyKind::Projection { .. })
                        {
                            (alias.args, lhs)
                        } else {
                            continue;
                        }
                    }
                    _ => continue,
                };

                // The output must be (equivalent to) the stuck variable.
                let Some(term_ty) = term.as_type() else { continue };
                let ty::Infer(ty::TyVar(term_vid)) = *self.shallow_resolve(term_ty).kind() else {
                    continue;
                };
                if !related.contains(&self.root_var(term_vid)) {
                    continue;
                }

                // Blame the first input that is still an unresolved variable. Args
                // are `[Self, trait args.., assoc args..]`, so this prefers `Self`,
                // e.g. `?input` of `<?input as SliceIndex<_>>::Output`.
                for arg in alias_args {
                    if let Some(arg_ty) = arg.as_type() {
                        let arg_ty = self.shallow_resolve(arg_ty);
                        if let ty::Infer(ty::TyVar(vid)) = arg_ty.kind() {
                            if candidate_roots.insert(self.root_var(*vid)) {
                                candidate = Some(arg_ty);
                            }
                            break;
                        }
                    }
                }
            }

            match candidate {
                Some(input) if candidate_roots.len() == 1 => {
                    result = Some(input);
                    current = input;
                }
                _ => break,
            }
        }
        result
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

        // We don't care about any pending goals which don't actually
        // use the self type.
        if !inspect_goal
            .orig_values()
            .iter()
            .filter_map(|arg| arg.as_type())
            .any(|ty| self.fcx.type_matches_expected_vid(ty, self.self_ty))
        {
            debug!(goal = ?inspect_goal.goal(), "goal does not mention self type");
            return;
        }

        let tcx = self.fcx.tcx;
        let goal = inspect_goal.goal();
        if self.fcx.predicate_has_self_ty(goal.predicate, self.self_ty) {
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
        // 3 is more than enough for all occurrences in practice (a.k.a. `Into`).
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
