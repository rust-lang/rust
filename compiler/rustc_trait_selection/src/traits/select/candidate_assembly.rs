//! Candidate assembly.
//!
//! The selection process begins by examining all in-scope impls,
//! caller obligations, and so forth and assembling a list of
//! candidates. See the [rustc dev guide] for more details.
//!
//! [rustc dev guide]:https://rustc-dev-guide.rust-lang.org/traits/resolution.html#candidate-assembly
use rustc_hir as hir;
use rustc_infer::traits::{Obligation, SelectionError, TraitObligation};
use rustc_middle::ty::{self, TypeFoldable};
use rustc_target::spec::abi::Abi;

use crate::traits::{util, SelectionResult};

use super::BuiltinImplConditions;
use super::SelectionCandidate::{self, *};
use super::{SelectionCandidateSet, SelectionContext, TraitObligationStack};

impl<'cx, 'tcx> SelectionContext<'cx, 'tcx> {
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
        debug!(
            "candidate_from_obligation(cache_fresh_trait_pred={:?}, obligation={:?})",
            cache_fresh_trait_pred, stack
        );
        debug_assert!(!stack.obligation.predicate.has_escaping_bound_vars());

        if let Some(c) =
            self.check_candidate_cache(stack.obligation.param_env, cache_fresh_trait_pred)
        {
            debug!("CACHE HIT: SELECT({:?})={:?}", cache_fresh_trait_pred, c);
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

        debug!("CACHE MISS: SELECT({:?})={:?}", cache_fresh_trait_pred, candidate);
        self.insert_candidate_cache(
            stack.obligation.param_env,
            cache_fresh_trait_pred,
            dep_node,
            candidate.clone(),
        );
        candidate
    }

    pub(super) fn assemble_candidates<'o>(
        &mut self,
        stack: &TraitObligationStack<'o, 'tcx>,
    ) -> Result<SelectionCandidateSet<'tcx>, SelectionError<'tcx>> {
        let TraitObligationStack { obligation, .. } = *stack;
        let obligation = &Obligation {
            param_env: obligation.param_env,
            cause: obligation.cause.clone(),
            recursion_depth: obligation.recursion_depth,
            predicate: self.infcx().resolve_vars_if_possible(&obligation.predicate),
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

        self.assemble_candidates_for_trait_alias(obligation, &mut candidates)?;

        // Other bounds. Consider both in-scope bounds from fn decl
        // and applicable impls. There is a certain set of precedence rules here.
        let def_id = obligation.predicate.def_id();
        let lang_items = self.tcx().lang_items();

        if lang_items.copy_trait() == Some(def_id) {
            debug!("obligation self ty is {:?}", obligation.predicate.skip_binder().self_ty());

            // User-defined copy impls are permitted, but only for
            // structs and enums.
            self.assemble_candidates_from_impls(obligation, &mut candidates)?;

            // For other types, we'll use the builtin rules.
            let copy_conditions = self.copy_clone_conditions(obligation);
            self.assemble_builtin_bound_candidates(copy_conditions, &mut candidates)?;
        } else if lang_items.discriminant_kind_trait() == Some(def_id) {
            // `DiscriminantKind` is automatically implemented for every type.
            candidates.vec.push(DiscriminantKindCandidate);
        } else if lang_items.sized_trait() == Some(def_id) {
            // Sized is never implementable by end-users, it is
            // always automatically computed.
            let sized_conditions = self.sized_conditions(obligation);
            self.assemble_builtin_bound_candidates(sized_conditions, &mut candidates)?;
        } else if lang_items.unsize_trait() == Some(def_id) {
            self.assemble_candidates_for_unsizing(obligation, &mut candidates);
        } else {
            if lang_items.clone_trait() == Some(def_id) {
                // Same builtin conditions as `Copy`, i.e., every type which has builtin support
                // for `Copy` also has builtin support for `Clone`, and tuples/arrays of `Clone`
                // types have builtin support for `Clone`.
                let clone_conditions = self.copy_clone_conditions(obligation);
                self.assemble_builtin_bound_candidates(clone_conditions, &mut candidates)?;
            }

            self.assemble_generator_candidates(obligation, &mut candidates)?;
            self.assemble_closure_candidates(obligation, &mut candidates)?;
            self.assemble_fn_pointer_candidates(obligation, &mut candidates)?;
            self.assemble_candidates_from_impls(obligation, &mut candidates)?;
            self.assemble_candidates_from_object_ty(obligation, &mut candidates);
        }

        self.assemble_candidates_from_projected_tys(obligation, &mut candidates);
        self.assemble_candidates_from_caller_bounds(stack, &mut candidates)?;
        // Auto implementations have lower priority, so we only
        // consider triggering a default if there is no other impl that can apply.
        if candidates.vec.is_empty() {
            self.assemble_candidates_from_auto_impls(obligation, &mut candidates)?;
        }
        debug!("candidate list size: {}", candidates.vec.len());
        Ok(candidates)
    }

    fn assemble_candidates_from_projected_tys(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        debug!("assemble_candidates_for_projected_tys({:?})", obligation);

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

        if result {
            candidates.vec.push(ProjectionCandidate);
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
        debug!("assemble_candidates_from_caller_bounds({:?})", stack.obligation);

        let all_bounds = stack
            .obligation
            .param_env
            .caller_bounds()
            .iter()
            .filter_map(|o| o.to_opt_poly_trait_ref());

        // Micro-optimization: filter out predicates relating to different traits.
        let matching_bounds =
            all_bounds.filter(|p| p.def_id() == stack.obligation.predicate.def_id());

        // Keep only those bounds which may apply, and propagate overflow if it occurs.
        let mut param_candidates = vec![];
        for bound in matching_bounds {
            let wc = self.evaluate_where_clause(stack, bound)?;
            if wc.may_apply() {
                param_candidates.push(ParamCandidate(bound));
            }
        }

        candidates.vec.extend(param_candidates);

        Ok(())
    }

    fn assemble_generator_candidates(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        if self.tcx().lang_items().gen_trait() != Some(obligation.predicate.def_id()) {
            return Ok(());
        }

        // Okay to skip binder because the substs on generator types never
        // touch bound regions, they just capture the in-scope
        // type/region parameters.
        let self_ty = obligation.self_ty().skip_binder();
        match self_ty.kind() {
            ty::Generator(..) => {
                debug!(
                    "assemble_generator_candidates: self_ty={:?} obligation={:?}",
                    self_ty, obligation
                );

                candidates.vec.push(GeneratorCandidate);
            }
            ty::Infer(ty::TyVar(_)) => {
                debug!("assemble_generator_candidates: ambiguous self-type");
                candidates.ambiguous = true;
            }
            _ => {}
        }

        Ok(())
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
    ) -> Result<(), SelectionError<'tcx>> {
        let kind = match self.tcx().fn_trait_kind_from_lang_item(obligation.predicate.def_id()) {
            Some(k) => k,
            None => {
                return Ok(());
            }
        };

        // Okay to skip binder because the substs on closure types never
        // touch bound regions, they just capture the in-scope
        // type/region parameters
        match *obligation.self_ty().skip_binder().kind() {
            ty::Closure(_, closure_substs) => {
                debug!("assemble_unboxed_candidates: kind={:?} obligation={:?}", kind, obligation);
                match self.infcx.closure_kind(closure_substs) {
                    Some(closure_kind) => {
                        debug!("assemble_unboxed_candidates: closure_kind = {:?}", closure_kind);
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

        Ok(())
    }

    /// Implements one of the `Fn()` family for a fn pointer.
    fn assemble_fn_pointer_candidates(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        // We provide impl of all fn traits for fn pointers.
        if self.tcx().fn_trait_kind_from_lang_item(obligation.predicate.def_id()).is_none() {
            return Ok(());
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
                    candidates.vec.push(FnPointerCandidate);
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
                        candidates.vec.push(FnPointerCandidate);
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Searches for impls that might apply to `obligation`.
    fn assemble_candidates_from_impls(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        debug!("assemble_candidates_from_impls(obligation={:?})", obligation);

        // Essentially any user-written impl will match with an error type,
        // so creating `ImplCandidates` isn't useful. However, we might
        // end up finding a candidate elsewhere (e.g. a `BuiltinCandidate` for `Sized)
        // This helps us avoid overflow: see issue #72839
        // Since compilation is already guaranteed to fail, this is just
        // to try to show the 'nicest' possible errors to the user.
        if obligation.references_error() {
            return Ok(());
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

        Ok(())
    }

    fn assemble_candidates_from_auto_impls(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        // Okay to skip binder here because the tests we do below do not involve bound regions.
        let self_ty = obligation.self_ty().skip_binder();
        debug!("assemble_candidates_from_auto_impls(self_ty={:?})", self_ty);

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
                    // `compile-fail/typeck-default-trait-impl-send-param.rs`
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

        Ok(())
    }

    /// Searches for impls that might apply to `obligation`.
    fn assemble_candidates_from_object_ty(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) {
        debug!(
            "assemble_candidates_from_object_ty(self_ty={:?})",
            obligation.self_ty().skip_binder()
        );

        self.infcx.probe(|_snapshot| {
            // The code below doesn't care about regions, and the
            // self-ty here doesn't escape this probe, so just erase
            // any LBR.
            let self_ty = self.tcx().erase_late_bound_regions(&obligation.self_ty());
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

            debug!("assemble_candidates_from_object_ty: poly_trait_ref={:?}", poly_trait_ref);

            // Count only those upcast versions that match the trait-ref
            // we are looking for. Specifically, do not only check for the
            // correct trait, but also the correct type parameters.
            // For example, we may be trying to upcast `Foo` to `Bar<i32>`,
            // but `Foo` is declared as `trait Foo: Bar<u32>`.
            let upcast_trait_refs = util::supertraits(self.tcx(), poly_trait_ref)
                .filter(|upcast_trait_ref| {
                    self.infcx
                        .probe(|_| self.match_poly_trait_ref(obligation, *upcast_trait_ref).is_ok())
                })
                .count();

            if upcast_trait_refs > 1 {
                // Can be upcast in many ways; need more type information.
                candidates.ambiguous = true;
            } else if upcast_trait_refs == 1 {
                candidates.vec.push(ObjectCandidate);
            }
        })
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

        debug!("assemble_candidates_for_unsizing(source={:?}, target={:?})", source, target);

        let may_apply = match (source.kind(), target.kind()) {
            // Trait+Kx+'a -> Trait+Ky+'b (upcasts).
            (&ty::Dynamic(ref data_a, ..), &ty::Dynamic(ref data_b, ..)) => {
                // Upcasts permit two things:
                //
                // 1. Dropping auto traits, e.g., `Foo + Send` to `Foo`
                // 2. Tightening the region bound, e.g., `Foo + 'a` to `Foo + 'b` if `'a: 'b`
                //
                // Note that neither of these changes requires any
                // change at runtime. Eventually this will be
                // generalized.
                //
                // We always upcast when we can because of reason
                // #2 (region bounds).
                data_a.principal_def_id() == data_b.principal_def_id()
                    && data_b
                        .auto_traits()
                        // All of a's auto traits need to be in b's auto traits.
                        .all(|b| data_a.auto_traits().any(|a| a == b))
            }

            // `T` -> `Trait`
            (_, &ty::Dynamic(..)) => true,

            // Ambiguous handling is below `T` -> `Trait`, because inference
            // variables can still implement `Unsize<Trait>` and nested
            // obligations will have the final say (likely deferred).
            (&ty::Infer(ty::TyVar(_)), _) | (_, &ty::Infer(ty::TyVar(_))) => {
                debug!("assemble_candidates_for_unsizing: ambiguous");
                candidates.ambiguous = true;
                false
            }

            // `[T; n]` -> `[T]`
            (&ty::Array(..), &ty::Slice(_)) => true,

            // `Struct<T>` -> `Struct<U>`
            (&ty::Adt(def_id_a, _), &ty::Adt(def_id_b, _)) if def_id_a.is_struct() => {
                def_id_a == def_id_b
            }

            // `(.., T)` -> `(.., U)`
            (&ty::Tuple(tys_a), &ty::Tuple(tys_b)) => tys_a.len() == tys_b.len(),

            _ => false,
        };

        if may_apply {
            candidates.vec.push(BuiltinUnsizeCandidate);
        }
    }

    fn assemble_candidates_for_trait_alias(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        // Okay to skip binder here because the tests we do below do not involve bound regions.
        let self_ty = obligation.self_ty().skip_binder();
        debug!("assemble_candidates_for_trait_alias(self_ty={:?})", self_ty);

        let def_id = obligation.predicate.def_id();

        if self.tcx().is_trait_alias(def_id) {
            candidates.vec.push(TraitAliasCandidate(def_id));
        }

        Ok(())
    }

    /// Assembles the trait which are built-in to the language itself:
    /// `Copy`, `Clone` and `Sized`.
    fn assemble_builtin_bound_candidates(
        &mut self,
        conditions: BuiltinImplConditions<'tcx>,
        candidates: &mut SelectionCandidateSet<'tcx>,
    ) -> Result<(), SelectionError<'tcx>> {
        match conditions {
            BuiltinImplConditions::Where(nested) => {
                debug!("builtin_bound: nested={:?}", nested);
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

        Ok(())
    }
}
