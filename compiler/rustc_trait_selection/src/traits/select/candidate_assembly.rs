//! Candidate assembly.
//!
//! The selection process begins by examining all in-scope impls,
//! caller obligations, and so forth and assembling a list of
//! candidates. See the [rustc dev guide] for more details.
//!
//! [rustc dev guide]:https://rustc-dev-guide.rust-lang.org/traits/resolution.html#candidate-assembly
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_infer::traits::TraitEngine;
use rustc_infer::traits::{Obligation, SelectionError, TraitObligation};
use rustc_lint_defs::builtin::DEREF_INTO_DYN_SUPERTRAIT;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, ToPredicate, Ty, TypeFoldable, WithConstness};
use rustc_target::spec::abi::Abi;

use crate::traits;
use crate::traits::coherence::Conflict;
use crate::traits::query::evaluate_obligation::InferCtxtExt;
use crate::traits::{util, SelectionResult};
use crate::traits::{Ambiguous, ErrorReporting, Overflow, Unimplemented};

use super::BuiltinImplConditions;
use super::IntercrateAmbiguityCause;
use super::OverflowError;
use super::SelectionCandidate::{self, *};
use super::{EvaluatedCandidate, SelectionCandidateSet, SelectionContext, TraitObligationStack};

impl<'cx, 'tcx> SelectionContext<'cx, 'tcx> {
    #[instrument(level = "debug", skip(self))]
    pub(super) fn candidate_from_obligation<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
    ) -> SelectionResult<'tcx, SelectionCandidate<'tcx>> {
        // Watch out for overflow. This intentionally bypasses (and does
        // not update) the cache.
        self.check_recursion_limit(&stack.obligation, &stack.obligation)?;

        // Check the cache. Note that we freshen the trait-ref
        // separately rather than using `stack.fresh_trait_ref` --
        // this is because we want the unbound variables to be
        // replaced with fresh types starting from index 0.
        let cache_fresh_trait_pred = self.infcx.freshen(stack.obligation.predicate);
        debug!(?cache_fresh_trait_pred);
        debug_assert!(!stack.obligation.predicate.has_escaping_bound_vars());

        if let Some(c) =
            self.check_candidate_cache(stack.obligation.param_env, cache_fresh_trait_pred)
        {
            debug!(candidate = ?c, "CACHE HIT");
            return c;
        }

        // If no match, compute result and insert into cache.
        //
        // FIXME(nikomatsakis) -- this cache is not taking into
        // account cycles that may have occurred in forming the
        // candidate. I don't know of any specific problems that
        // result but it seems awfully suspicious.
        let (candidate, dep_node) =
            self.in_task(|this| this.candidate_from_obligation_no_cache(stack));

        debug!(?candidate, "CACHE MISS");
        self.insert_candidate_cache(
            stack.obligation.param_env,
            cache_fresh_trait_pred,
            dep_node,
            candidate.clone(),
        );
        candidate
    }

    fn candidate_from_obligation_no_cache<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
    ) -> SelectionResult<'tcx, SelectionCandidate<'tcx>> {
        if let Some(conflict) = self.is_knowable(stack) {
            debug!("coherence stage: not knowable");
            if self.intercrate_ambiguity_causes.is_some() {
                debug!("evaluate_stack: intercrate_ambiguity_causes is some");
                // Heuristics: show the diagnostics when there are no candidates in crate.
                if let Ok(candidate_set) = self.assemble_candidates(stack) {
                    let mut no_candidates_apply = true;

                    for c in candidate_set.vec.iter() {
                        if self.evaluate_candidate(stack, &c)?.may_apply() {
                            no_candidates_apply = false;
                            break;
                        }
                    }

                    if !candidate_set.ambiguous && no_candidates_apply {
                        let trait_ref = stack.obligation.predicate.skip_binder().trait_ref;
                        let self_ty = trait_ref.self_ty();
                        let (trait_desc, self_desc) = with_no_trimmed_paths(|| {
                            let trait_desc = trait_ref.print_only_trait_path().to_string();
                            let self_desc = if self_ty.has_concrete_skeleton() {
                                Some(self_ty.to_string())
                            } else {
                                None
                            };
                            (trait_desc, self_desc)
                        });
                        let cause = if let Conflict::Upstream = conflict {
                            IntercrateAmbiguityCause::UpstreamCrateUpdate { trait_desc, self_desc }
                        } else {
                            IntercrateAmbiguityCause::DownstreamCrate { trait_desc, self_desc }
                        };
                        debug!(?cause, "evaluate_stack: pushing cause");
                        self.intercrate_ambiguity_causes.as_mut().unwrap().push(cause);
                    }
                }
            }
            return Ok(None);
        }

        let candidate_set = self.assemble_candidates(stack)?;

        if candidate_set.ambiguous {
            debug!("candidate set contains ambig");
            return Ok(None);
        }

        let candidates = candidate_set.vec;

        debug!(?stack, ?candidates, "assembled {} candidates", candidates.len());

        // At this point, we know that each of the entries in the
        // candidate set is *individually* applicable. Now we have to
        // figure out if they contain mutual incompatibilities. This
        // frequently arises if we have an unconstrained input type --
        // for example, we are looking for `$0: Eq` where `$0` is some
        // unconstrained type variable. In that case, we'll get a
        // candidate which assumes $0 == int, one that assumes `$0 ==
        // usize`, etc. This spells an ambiguity.

        let mut candidates = self.filter_impls(candidates, stack.obligation);

        // If there is more than one candidate, first winnow them down
        // by considering extra conditions (nested obligations and so
        // forth). We don't winnow if there is exactly one
        // candidate. This is a relatively minor distinction but it
        // can lead to better inference and error-reporting. An
        // example would be if there was an impl:
        //
        //     impl<T:Clone> Vec<T> { fn push_clone(...) { ... } }
        //
        // and we were to see some code `foo.push_clone()` where `boo`
        // is a `Vec<Bar>` and `Bar` does not implement `Clone`.  If
        // we were to winnow, we'd wind up with zero candidates.
        // Instead, we select the right impl now but report "`Bar` does
        // not implement `Clone`".
        if candidates.len() == 1 {
            return self.filter_reservation_impls(candidates.pop().unwrap(), stack.obligation);
        }

        // Winnow, but record the exact outcome of evaluation, which
        // is needed for specialization. Propagate overflow if it occurs.
        let mut candidates = candidates
            .into_iter()
            .map(|c| match self.evaluate_candidate(stack, &c) {
                Ok(eval) if eval.may_apply() => {
                    Ok(Some(EvaluatedCandidate { candidate: c, evaluation: eval }))
                }
                Ok(_) => Ok(None),
                Err(OverflowError::Canonical) => Err(Overflow),
                Err(OverflowError::ErrorReporting) => Err(ErrorReporting),
            })
            .flat_map(Result::transpose)
            .collect::<Result<Vec<_>, _>>()?;

        debug!(?stack, ?candidates, "winnowed to {} candidates", candidates.len());

        let needs_infer = stack.obligation.predicate.has_infer_types_or_consts();

        // If there are STILL multiple candidates, we can further
        // reduce the list by dropping duplicates -- including
        // resolving specializations.
        if candidates.len() > 1 {
            let mut i = 0;
            while i < candidates.len() {
                let is_dup = (0..candidates.len()).filter(|&j| i != j).any(|j| {
                    self.candidate_should_be_dropped_in_favor_of(
                        &candidates[i],
                        &candidates[j],
                        needs_infer,
                    )
                });
                if is_dup {
                    debug!(candidate = ?candidates[i], "Dropping candidate #{}/{}", i, candidates.len());
                    candidates.swap_remove(i);
                } else {
                    debug!(candidate = ?candidates[i], "Retaining candidate #{}/{}", i, candidates.len());
                    i += 1;

                    // If there are *STILL* multiple candidates, give up
                    // and report ambiguity.
                    if i > 1 {
                        debug!("multiple matches, ambig");
                        return Err(Ambiguous(
                            candidates
                                .into_iter()
                                .filter_map(|c| match c.candidate {
                                    SelectionCandidate::ImplCandidate(def_id) => Some(def_id),
                                    _ => None,
                                })
                                .collect(),
                        ));
                    }
                }
            }
        }

        // If there are *NO* candidates, then there are no impls --
        // that we know of, anyway. Note that in the case where there
        // are unbound type variables within the obligation, it might
        // be the case that you could still satisfy the obligation
        // from another crate by instantiating the type variables with
        // a type from another crate that does have an impl. This case
        // is checked for in `evaluate_stack` (and hence users
        // who might care about this case, like coherence, should use
        // that function).
        if candidates.is_empty() {
            // If there's an error type, 'downgrade' our result from
            // `Err(Unimplemented)` to `Ok(None)`. This helps us avoid
            // emitting additional spurious errors, since we're guaranteed
            // to have emitted at least one.
            if stack.obligation.references_error() {
                debug!("no results for error type, treating as ambiguous");
                return Ok(None);
            }
            return Err(Unimplemented);
        }

        // Just one candidate left.
        self.filter_reservation_impls(candidates.pop().unwrap().candidate, stack.obligation)
    }

    #[instrument(skip(self, stack), level = "debug")]
    pub(super) fn assemble_candidates<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
    ) -> Result<SelectionCandidateSet<'tcx>, SelectionError<'tcx>> {
        let TraitObligationStack { obligation, .. } = *stack;
        let obligation = &Obligation {
            param_env: obligation.param_env,
            cause: obligation.cause.clone(),
            recursion_depth: obligation.recursion_depth,
            predicate: self.infcx().resolve_vars_if_possible(obligation.predicate),
        };

        if obligation.predicate.skip_binder().self_ty().is_ty_var() {
            // Self is a type variable (e.g., `_: AsRef<str>`).
            //
            // This is somewhat problematic, as the current scheme can't really
            // handle it turning to be a projection. This does end up as truly
            // ambiguous in most cases anyway.
            //
            // Take the fast path out - this also improves
            // performance by preventing assemble_candidates_from_impls from
            // matching every impl for this trait.
            return Ok(SelectionCandidateSet { vec: vec![], ambiguous: true });
        }

        let mut candidates = SelectionCandidateSet { vec: Vec::new(), ambiguous: false };

        // The only way to prove a NotImplemented(T: Foo) predicate is via a negative impl.
        // There are no compiler built-in rules for this.
        if obligation.polarity() == ty::ImplPolarity::Negative {
            self.assemble_candidates_for_trait_alias(obligation, &mut candidates);
            self.assemble_candidates_from_impls(obligation, &mut candidates);
        } else {
            self.assemble_candidates_for_trait_alias(obligation, &mut candidates);

            // Other bounds. Consider both in-scope bounds from fn decl
            // and applicable impls. There is a certain set of precedence rules here.
            let def_id = obligation.predicate.def_id();
            let lang_items = self.tcx().lang_items();

            if lang_items.copy_trait() == Some(def_id) {
                debug!(obligation_self_ty = ?obligation.predicate.skip_binder().self_ty());

                // User-defined copy impls are permitted, but only for
                // structs and enums.
                self.assemble_candidates_from_impls(obligation, &mut candidates);

                // For other types, we'll use the builtin rules.
                let copy_conditions = self.copy_clone_conditions(obligation);
                self.assemble_builtin_bound_candidates(copy_conditions, &mut candidates);
            } else if lang_items.discriminant_kind_trait() == Some(def_id) {
                // `DiscriminantKind` is automatically implemented for every type.
                candidates.vec.push(DiscriminantKindCandidate);
            } else if lang_items.pointee_trait() == Some(def_id) {
                // `Pointee` is automatically implemented for every type.
                candidates.vec.push(PointeeCandidate);
            } else if lang_items.sized_trait() == Some(def_id) {
                // Sized is never implementable by end-users, it is
                // always automatically computed.
                let sized_conditions = self.sized_conditions(obligation);
                self.assemble_builtin_bound_candidates(sized_conditions, &mut candidates);
            } else if lang_items.unsize_trait() == Some(def_id) {
                self.assemble_candidates_for_unsizing(obligation, &mut candidates);
            } else if lang_items.drop_trait() == Some(def_id)
                && obligation.predicate.skip_binder().constness == ty::BoundConstness::ConstIfConst
            {
                if self.is_in_const_context {
                    self.assemble_const_drop_candidates(obligation, stack, &mut candidates)?;
                } else {
                    debug!("passing ~const Drop bound; in non-const context");
                    // `~const Drop` when we are not in a const context has no effect.
                    candidates.vec.push(ConstDropCandidate)
                }
            } else {
                if lang_items.clone_trait() == Some(def_id) {
                    // Same builtin conditions as `Copy`, i.e., every type which has builtin support
                    // for `Copy` also has builtin support for `Clone`, and tuples/arrays of `Clone`
                    // types have builtin support for `Clone`.
                    let clone_conditions = self.copy_clone_conditions(obligation);
                    self.assemble_builtin_bound_candidates(clone_conditions, &mut candidates);
                }

                self.assemble_generator_candidates(obligation, &mut candidates);
                self.assemble_closure_candidates(obligation, &mut candidates);
                self.assemble_fn_pointer_candidates(obligation, &mut candidates);
                self.assemble_candidates_from_impls(obligation, &mut candidates);
                self.assemble_candidates_from_object_ty(obligation, &mut candidates);
            }

            self.assemble_candidates_from_projected_tys(obligation, &mut candidates);
            self.assemble_candidates_from_caller_bounds(stack, &mut candidates)?;
            // Auto implementations have lower priority, so we only
            // consider triggering a default if there is no other impl that can apply.
            if candidates.vec.is_empty() {
                self.assemble_candidates_from_auto_impls(obligation, &mut candidates);
            }
        }
        debug!("candidate list size: {}", candidates.vec.len());
        Ok(candidates)
    }

    fn assemble_candidates_from_projected_tys(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        debug!(?obligation, "assemble_candidates_from_projected_tys");

        // Before we go into the whole placeholder thing, just
        // quickly check if the self-type is a projection at all.
        match obligation.predicate.skip_binder().trait_ref.self_ty().kind() {
            ty::Projection(_) | ty::Opaque(..) => {}
            ty::Infer(ty::TyVar(_)) => {
                span_bug!(
                    obligation.cause.span,
                    "Self=_ should have been handled by assemble_candidates"
                );
            }
            _ => return,
        }

        let result = self
            .infcx
            .probe(|_| self.match_projection_obligation_against_definition_bounds(obligation));

        for predicate_index in result {
            candidates.vec.push(ProjectionCandidate(predicate_index));
        }
    }

    /// Given an obligation like `<SomeTrait for T>`, searches the obligations that the caller
    /// supplied to find out whether it is listed among them.
    ///
    /// Never affects the inference environment.
    fn assemble_candidates_from_caller_bounds<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        debug!(?stack.obligation, "assemble_candidates_from_caller_bounds");

        let all_bounds = stack
            .obligation
            .param_env
            .caller_bounds()
            .iter()
            .filter_map(|o| o.to_opt_poly_trait_ref());

        // Micro-optimization: filter out predicates relating to different traits.
        let matching_bounds =
            all_bounds.filter(|p| p.value.def_id() == stack.obligation.predicate.def_id());

        // Keep only those bounds which may apply, and propagate overflow if it occurs.
        for bound in matching_bounds {
            let wc = self.evaluate_where_clause(stack, bound.value)?;
            if wc.may_apply() {
                candidates.vec.push(ParamCandidate((bound, stack.obligation.polarity())));
            }
        }

        Ok(())
    }

    fn assemble_generator_candidates(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        if self.tcx().lang_items().gen_trait() != Some(obligation.predicate.def_id()) {
            return;
        }

        // Okay to skip binder because the substs on generator types never
        // touch bound regions, they just capture the in-scope
        // type/region parameters.
        let self_ty = obligation.self_ty().skip_binder();
        match self_ty.kind() {
            ty::Generator(..) => {
                debug!(?self_ty, ?obligation, "assemble_generator_candidates",);

                candidates.vec.push(GeneratorCandidate);
            }
            ty::Infer(ty::TyVar(_)) => {
                debug!("assemble_generator_candidates: ambiguous self-type");
                candidates.ambiguous = true;
            }
            _ => {}
        }
    }

    /// Checks for the artificial impl that the compiler will create for an obligation like `X :
    /// FnMut<..>` where `X` is a closure type.
    ///
    /// Note: the type parameters on a closure candidate are modeled as *output* type
    /// parameters and hence do not affect whether this trait is a match or not. They will be
    /// unified during the confirmation step.
    fn assemble_closure_candidates(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        let kind = match self.tcx().fn_trait_kind_from_lang_item(obligation.predicate.def_id()) {
            Some(k) => k,
            None => {
                return;
            }
        };

        // Okay to skip binder because the substs on closure types never
        // touch bound regions, they just capture the in-scope
        // type/region parameters
        match *obligation.self_ty().skip_binder().kind() {
            ty::Closure(_, closure_substs) => {
                debug!(?kind, ?obligation, "assemble_unboxed_candidates");
                match self.infcx.closure_kind(closure_substs) {
                    Some(closure_kind) => {
                        debug!(?closure_kind, "assemble_unboxed_candidates");
                        if closure_kind.extends(kind) {
                            candidates.vec.push(ClosureCandidate);
                        }
                    }
                    None => {
                        debug!("assemble_unboxed_candidates: closure_kind not yet known");
                        candidates.vec.push(ClosureCandidate);
                    }
                }
            }
            ty::Infer(ty::TyVar(_)) => {
                debug!("assemble_unboxed_closure_candidates: ambiguous self-type");
                candidates.ambiguous = true;
            }
            _ => {}
        }
    }

    /// Implements one of the `Fn()` family for a fn pointer.
    fn assemble_fn_pointer_candidates(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        // We provide impl of all fn traits for fn pointers.
        if self.tcx().fn_trait_kind_from_lang_item(obligation.predicate.def_id()).is_none() {
            return;
        }

        // Okay to skip binder because what we are inspecting doesn't involve bound regions.
        let self_ty = obligation.self_ty().skip_binder();
        match *self_ty.kind() {
            ty::Infer(ty::TyVar(_)) => {
                debug!("assemble_fn_pointer_candidates: ambiguous self-type");
                candidates.ambiguous = true; // Could wind up being a fn() type.
            }
            // Provide an impl, but only for suitable `fn` pointers.
            ty::FnPtr(_) => {
                if let ty::FnSig {
                    unsafety: hir::Unsafety::Normal,
                    abi: Abi::Rust,
                    c_variadic: false,
                    ..
                } = self_ty.fn_sig(self.tcx()).skip_binder()
                {
                    candidates.vec.push(FnPointerCandidate { is_const: false });
                }
            }
            // Provide an impl for suitable functions, rejecting `#[target_feature]` functions (RFC 2396).
            ty::FnDef(def_id, _) => {
                if let ty::FnSig {
                    unsafety: hir::Unsafety::Normal,
                    abi: Abi::Rust,
                    c_variadic: false,
                    ..
                } = self_ty.fn_sig(self.tcx()).skip_binder()
                {
                    if self.tcx().codegen_fn_attrs(def_id).target_features.is_empty() {
                        candidates
                            .vec
                            .push(FnPointerCandidate { is_const: self.tcx().is_const_fn(def_id) });
                    }
                }
            }
            _ => {}
        }
    }

    /// Searches for impls that might apply to `obligation`.
    fn assemble_candidates_from_impls(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        debug!(?obligation, "assemble_candidates_from_impls");

        // Essentially any user-written impl will match with an error type,
        // so creating `ImplCandidates` isn't useful. However, we might
        // end up finding a candidate elsewhere (e.g. a `BuiltinCandidate` for `Sized)
        // This helps us avoid overflow: see issue #72839
        // Since compilation is already guaranteed to fail, this is just
        // to try to show the 'nicest' possible errors to the user.
        // We don't check for errors in the `ParamEnv` - in practice,
        // it seems to cause us to be overly aggressive in deciding
        // to give up searching for candidates, leading to spurious errors.
        if obligation.predicate.references_error() {
            return;
        }

        self.tcx().for_each_relevant_impl(
            obligation.predicate.def_id(),
            obligation.predicate.skip_binder().trait_ref.self_ty(),
            |impl_def_id| {
                self.infcx.probe(|_| {
                    if let Ok(_substs) = self.match_impl(impl_def_id, obligation) {
                        candidates.vec.push(ImplCandidate(impl_def_id));
                    }
                });
            },
        );
    }

    fn assemble_candidates_from_auto_impls(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        // Okay to skip binder here because the tests we do below do not involve bound regions.
        let self_ty = obligation.self_ty().skip_binder();
        debug!(?self_ty, "assemble_candidates_from_auto_impls");

        let def_id = obligation.predicate.def_id();

        if self.tcx().trait_is_auto(def_id) {
            match self_ty.kind() {
                ty::Dynamic(..) => {
                    // For object types, we don't know what the closed
                    // over types are. This means we conservatively
                    // say nothing; a candidate may be added by
                    // `assemble_candidates_from_object_ty`.
                }
                ty::Foreign(..) => {
                    // Since the contents of foreign types is unknown,
                    // we don't add any `..` impl. Default traits could
                    // still be provided by a manual implementation for
                    // this trait and type.
                }
                ty::Param(..) | ty::Projection(..) => {
                    // In these cases, we don't know what the actual
                    // type is.  Therefore, we cannot break it down
                    // into its constituent types. So we don't
                    // consider the `..` impl but instead just add no
                    // candidates: this means that typeck will only
                    // succeed if there is another reason to believe
                    // that this obligation holds. That could be a
                    // where-clause or, in the case of an object type,
                    // it could be that the object type lists the
                    // trait (e.g., `Foo+Send : Send`). See
                    // `ui/typeck/typeck-default-trait-impl-send-param.rs`
                    // for an example of a test case that exercises
                    // this path.
                }
                ty::Infer(ty::TyVar(_)) => {
                    // The auto impl might apply; we don't know.
                    candidates.ambiguous = true;
                }
                ty::Generator(_, _, movability)
                    if self.tcx().lang_items().unpin_trait() == Some(def_id) =>
                {
                    match movability {
                        hir::Movability::Static => {
                            // Immovable generators are never `Unpin`, so
                            // suppress the normal auto-impl candidate for it.
                        }
                        hir::Movability::Movable => {
                            // Movable generators are always `Unpin`, so add an
                            // unconditional builtin candidate.
                            candidates.vec.push(BuiltinCandidate { has_nested: false });
                        }
                    }
                }

                _ => candidates.vec.push(AutoImplCandidate(def_id)),
            }
        }
    }

    /// Searches for impls that might apply to `obligation`.
    fn assemble_candidates_from_object_ty(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        debug!(
            self_ty = ?obligation.self_ty().skip_binder(),
            "assemble_candidates_from_object_ty",
        );

        self.infcx.probe(|_snapshot| {
            // The code below doesn't care about regions, and the
            // self-ty here doesn't escape this probe, so just erase
            // any LBR.
            let self_ty = self.tcx().erase_late_bound_regions(obligation.self_ty());
            let poly_trait_ref = match self_ty.kind() {
                ty::Dynamic(ref data, ..) => {
                    if data.auto_traits().any(|did| did == obligation.predicate.def_id()) {
                        debug!(
                            "assemble_candidates_from_object_ty: matched builtin bound, \
                             pushing candidate"
                        );
                        candidates.vec.push(BuiltinObjectCandidate);
                        return;
                    }

                    if let Some(principal) = data.principal() {
                        if !self.infcx.tcx.features().object_safe_for_dispatch {
                            principal.with_self_ty(self.tcx(), self_ty)
                        } else if self.tcx().is_object_safe(principal.def_id()) {
                            principal.with_self_ty(self.tcx(), self_ty)
                        } else {
                            return;
                        }
                    } else {
                        // Only auto trait bounds exist.
                        return;
                    }
                }
                ty::Infer(ty::TyVar(_)) => {
                    debug!("assemble_candidates_from_object_ty: ambiguous");
                    candidates.ambiguous = true; // could wind up being an object type
                    return;
                }
                _ => return,
            };

            debug!(?poly_trait_ref, "assemble_candidates_from_object_ty");

            let poly_trait_predicate = self.infcx().resolve_vars_if_possible(obligation.predicate);
            let placeholder_trait_predicate =
                self.infcx().replace_bound_vars_with_placeholders(poly_trait_predicate);

            // Count only those upcast versions that match the trait-ref
            // we are looking for. Specifically, do not only check for the
            // correct trait, but also the correct type parameters.
            // For example, we may be trying to upcast `Foo` to `Bar<i32>`,
            // but `Foo` is declared as `trait Foo: Bar<u32>`.
            let candidate_supertraits = util::supertraits(self.tcx(), poly_trait_ref)
                .enumerate()
                .filter(|&(_, upcast_trait_ref)| {
                    self.infcx.probe(|_| {
                        self.match_normalize_trait_ref(
                            obligation,
                            upcast_trait_ref,
                            placeholder_trait_predicate.trait_ref,
                        )
                        .is_ok()
                    })
                })
                .map(|(idx, _)| ObjectCandidate(idx));

            candidates.vec.extend(candidate_supertraits);
        })
    }

    /// Temporary migration for #89190
    fn need_migrate_deref_output_trait_object(
        &mut self,
        ty: Ty<'tcx>,
        cause: &traits::ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Option<(Ty<'tcx>, DefId)> {
        let tcx = self.tcx();
        if tcx.features().trait_upcasting {
            return None;
        }

        // <ty as Deref>
        let trait_ref = ty::TraitRef {
            def_id: tcx.lang_items().deref_trait()?,
            substs: tcx.mk_substs_trait(ty, &[]),
        };

        let obligation = traits::Obligation::new(
            cause.clone(),
            param_env,
            ty::Binder::dummy(trait_ref).without_const().to_predicate(tcx),
        );
        if !self.infcx.predicate_may_hold(&obligation) {
            return None;
        }

        let mut fulfillcx = traits::FulfillmentContext::new_in_snapshot();
        let normalized_ty = fulfillcx.normalize_projection_type(
            &self.infcx,
            param_env,
            ty::ProjectionTy {
                item_def_id: tcx.lang_items().deref_target()?,
                substs: trait_ref.substs,
            },
            cause.clone(),
        );

        let ty::Dynamic(data, ..) = normalized_ty.kind() else {
            return None;
        };

        let def_id = data.principal_def_id()?;

        return Some((normalized_ty, def_id));
    }

    /// Searches for unsizing that might apply to `obligation`.
    fn assemble_candidates_for_unsizing(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        // We currently never consider higher-ranked obligations e.g.
        // `for<'a> &'a T: Unsize<Trait+'a>` to be implemented. This is not
        // because they are a priori invalid, and we could potentially add support
        // for them later, it's just that there isn't really a strong need for it.
        // A `T: Unsize<U>` obligation is always used as part of a `T: CoerceUnsize<U>`
        // impl, and those are generally applied to concrete types.
        //
        // That said, one might try to write a fn with a where clause like
        //     for<'a> Foo<'a, T>: Unsize<Foo<'a, Trait>>
        // where the `'a` is kind of orthogonal to the relevant part of the `Unsize`.
        // Still, you'd be more likely to write that where clause as
        //     T: Trait
        // so it seems ok if we (conservatively) fail to accept that `Unsize`
        // obligation above. Should be possible to extend this in the future.
        let source = match obligation.self_ty().no_bound_vars() {
            Some(t) => t,
            None => {
                // Don't add any candidates if there are bound regions.
                return;
            }
        };
        let target = obligation.predicate.skip_binder().trait_ref.substs.type_at(1);

        debug!(?source, ?target, "assemble_candidates_for_unsizing");

        match (source.kind(), target.kind()) {
            // Trait+Kx+'a -> Trait+Ky+'b (upcasts).
            (&ty::Dynamic(ref data_a, ..), &ty::Dynamic(ref data_b, ..)) => {
                // Upcast coercions permit several things:
                //
                // 1. Dropping auto traits, e.g., `Foo + Send` to `Foo`
                // 2. Tightening the region bound, e.g., `Foo + 'a` to `Foo + 'b` if `'a: 'b`
                // 3. Tightening trait to its super traits, eg. `Foo` to `Bar` if `Foo: Bar`
                //
                // Note that neither of the first two of these changes requires any
                // change at runtime. The third needs to change pointer metadata at runtime.
                //
                // We always perform upcasting coercions when we can because of reason
                // #2 (region bounds).
                let auto_traits_compatible = data_b
                    .auto_traits()
                    // All of a's auto traits need to be in b's auto traits.
                    .all(|b| data_a.auto_traits().any(|a| a == b));
                if auto_traits_compatible {
                    let principal_def_id_a = data_a.principal_def_id();
                    let principal_def_id_b = data_b.principal_def_id();
                    if principal_def_id_a == principal_def_id_b {
                        // no cyclic
                        candidates.vec.push(BuiltinUnsizeCandidate);
                    } else if principal_def_id_a.is_some() && principal_def_id_b.is_some() {
                        // not casual unsizing, now check whether this is trait upcasting coercion.
                        let principal_a = data_a.principal().unwrap();
                        let target_trait_did = principal_def_id_b.unwrap();
                        let source_trait_ref = principal_a.with_self_ty(self.tcx(), source);
                        if let Some((deref_output_ty, deref_output_trait_did)) = self
                            .need_migrate_deref_output_trait_object(
                                source,
                                &obligation.cause,
                                obligation.param_env,
                            )
                        {
                            if deref_output_trait_did == target_trait_did {
                                self.tcx().struct_span_lint_hir(
                                    DEREF_INTO_DYN_SUPERTRAIT,
                                    obligation.cause.body_id,
                                    obligation.cause.span,
                                    |lint| {
                                        lint.build(&format!(
                                            "`{}` implements `Deref` with supertrait `{}` as output",
                                            source,
                                            deref_output_ty
                                        )).emit();
                                    },
                                );
                                return;
                            }
                        }

                        for (idx, upcast_trait_ref) in
                            util::supertraits(self.tcx(), source_trait_ref).enumerate()
                        {
                            if upcast_trait_ref.def_id() == target_trait_did {
                                candidates.vec.push(TraitUpcastingUnsizeCandidate(idx));
                            }
                        }
                    }
                }
            }

            // `T` -> `Trait`
            (_, &ty::Dynamic(..)) => {
                candidates.vec.push(BuiltinUnsizeCandidate);
            }

            // Ambiguous handling is below `T` -> `Trait`, because inference
            // variables can still implement `Unsize<Trait>` and nested
            // obligations will have the final say (likely deferred).
            (&ty::Infer(ty::TyVar(_)), _) | (_, &ty::Infer(ty::TyVar(_))) => {
                debug!("assemble_candidates_for_unsizing: ambiguous");
                candidates.ambiguous = true;
            }

            // `[T; n]` -> `[T]`
            (&ty::Array(..), &ty::Slice(_)) => {
                candidates.vec.push(BuiltinUnsizeCandidate);
            }

            // `Struct<T>` -> `Struct<U>`
            (&ty::Adt(def_id_a, _), &ty::Adt(def_id_b, _)) if def_id_a.is_struct() => {
                if def_id_a == def_id_b {
                    candidates.vec.push(BuiltinUnsizeCandidate);
                }
            }

            // `(.., T)` -> `(.., U)`
            (&ty::Tuple(tys_a), &ty::Tuple(tys_b)) => {
                if tys_a.len() == tys_b.len() {
                    candidates.vec.push(BuiltinUnsizeCandidate);
                }
            }

            _ => {}
        };
    }

    fn assemble_candidates_for_trait_alias(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        // Okay to skip binder here because the tests we do below do not involve bound regions.
        let self_ty = obligation.self_ty().skip_binder();
        debug!(?self_ty, "assemble_candidates_for_trait_alias");

        let def_id = obligation.predicate.def_id();

        if self.tcx().is_trait_alias(def_id) {
            candidates.vec.push(TraitAliasCandidate(def_id));
        }
    }

    /// Assembles the trait which are built-in to the language itself:
    /// `Copy`, `Clone` and `Sized`.
    fn assemble_builtin_bound_candidates(
        &mut self,
        conditions: BuiltinImplConditions<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        match conditions {
            BuiltinImplConditions::Where(nested) => {
                debug!(?nested, "builtin_bound");
                candidates
                    .vec
                    .push(BuiltinCandidate { has_nested: !nested.skip_binder().is_empty() });
            }
            BuiltinImplConditions::None => {}
            BuiltinImplConditions::Ambiguous => {
                debug!("assemble_builtin_bound_candidates: ambiguous builtin");
                candidates.ambiguous = true;
            }
        }
    }

    fn assemble_const_drop_candidates<'a>(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        obligation_stack: &TraitObligationStack<'a, 'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        let mut stack: Vec<(Ty<'tcx>, usize)> = vec![(obligation.self_ty().skip_binder(), 0)];

        while let Some((ty, depth)) = stack.pop() {
            let mut noreturn = false;

            self.check_recursion_depth(depth, obligation)?;
            let mut new_candidates = SelectionCandidateSet { vec: Vec::new(), ambiguous: false };
            let mut copy_obligation =
                obligation.with(obligation.predicate.rebind(ty::TraitPredicate {
                    trait_ref: ty::TraitRef {
                        def_id: self.tcx().require_lang_item(hir::LangItem::Copy, None),
                        substs: self.tcx().mk_substs_trait(ty, &[]),
                    },
                    constness: ty::BoundConstness::NotConst,
                    polarity: ty::ImplPolarity::Positive,
                }));
            copy_obligation.recursion_depth = depth + 1;
            self.assemble_candidates_from_impls(&copy_obligation, &mut new_candidates);
            let copy_conditions = self.copy_clone_conditions(&copy_obligation);
            self.assemble_builtin_bound_candidates(copy_conditions, &mut new_candidates);
            let copy_stack = self.push_stack(obligation_stack.list(), &copy_obligation);
            self.assemble_candidates_from_caller_bounds(&copy_stack, &mut new_candidates)?;

            let const_drop_obligation =
                obligation.with(obligation.predicate.rebind(ty::TraitPredicate {
                    trait_ref: ty::TraitRef {
                        def_id: self.tcx().require_lang_item(hir::LangItem::Drop, None),
                        substs: self.tcx().mk_substs_trait(ty, &[]),
                    },
                    constness: ty::BoundConstness::ConstIfConst,
                    polarity: ty::ImplPolarity::Positive,
                }));

            let const_drop_stack = self.push_stack(obligation_stack.list(), &const_drop_obligation);
            self.assemble_candidates_from_caller_bounds(&const_drop_stack, &mut new_candidates)?;

            if !new_candidates.vec.is_empty() {
                noreturn = true;
            }
            debug!(?new_candidates.vec, "assemble_const_drop_candidates");

            match ty.kind() {
                ty::Int(_)
                | ty::Uint(_)
                | ty::Float(_)
                | ty::Infer(ty::IntVar(_))
                | ty::Infer(ty::FloatVar(_))
                | ty::FnPtr(_)
                | ty::Never
                | ty::Ref(..)
                | ty::FnDef(..)
                | ty::RawPtr(_)
                | ty::Bool
                | ty::Char
                | ty::Str
                | ty::Foreign(_) => {} // Do nothing. These types satisfy `const Drop`.

                ty::Adt(def, subst) => {
                    let mut set = SelectionCandidateSet { vec: Vec::new(), ambiguous: false };
                    self.assemble_candidates_from_impls(
                        &obligation.with(obligation.predicate.map_bound(|mut pred| {
                            pred.trait_ref.substs = self.tcx().mk_substs_trait(ty, &[]);
                            pred
                        })),
                        &mut set,
                    );
                    stack.extend(def.all_fields().map(|f| (f.ty(self.tcx(), subst), depth + 1)));

                    debug!(?set.vec, "assemble_const_drop_candidates - ty::Adt");
                    if set.vec.into_iter().any(|candidate| {
                        if let SelectionCandidate::ImplCandidate(did) = candidate {
                            matches!(self.tcx().impl_constness(did), hir::Constness::NotConst)
                        } else {
                            false
                        }
                    }) {
                        if !noreturn {
                            // has non-const Drop
                            return Ok(());
                        }
                        debug!("not returning");
                    }
                }

                ty::Array(ty, _) => stack.push((ty, depth + 1)),

                ty::Tuple(_) => stack.extend(ty.tuple_fields().map(|t| (t, depth + 1))),

                ty::Closure(_, substs) => {
                    let substs = substs.as_closure();
                    let ty = self.infcx.shallow_resolve(substs.tupled_upvars_ty());
                    stack.push((ty, depth + 1));
                }

                ty::Generator(_, substs, _) => {
                    let substs = substs.as_generator();
                    let ty = self.infcx.shallow_resolve(substs.tupled_upvars_ty());

                    stack.push((ty, depth + 1));
                    stack.push((substs.witness(), depth + 1));
                }

                ty::GeneratorWitness(tys) => stack.extend(
                    self.tcx().erase_late_bound_regions(*tys).iter().map(|t| (t, depth + 1)),
                ),

                ty::Slice(ty) => stack.push((ty, depth + 1)),

                ty::Opaque(..)
                | ty::Dynamic(..)
                | ty::Error(_)
                | ty::Bound(..)
                | ty::Infer(_)
                | ty::Placeholder(_)
                | ty::Projection(..)
                | ty::Param(..) => {
                    if !noreturn {
                        return Ok(());
                    }
                    debug!("not returning");
                }
            }
            debug!(?stack, "assemble_const_drop_candidates - in loop");
        }
        // all types have passed.
        candidates.vec.push(ConstDropCandidate);

        Ok(())
    }
}
