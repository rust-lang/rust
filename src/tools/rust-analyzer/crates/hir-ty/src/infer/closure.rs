//! Inference of closure parameter types based on the closure's expected type.

pub(crate) mod analysis;

use std::{iter, mem, ops::ControlFlow};

use hir_def::{
    TraitId,
    hir::{ClosureKind, ExprId, PatId},
    type_ref::TypeRefId,
};
use rustc_type_ir::{
    ClosureArgs, ClosureArgsParts, CoroutineArgs, CoroutineArgsParts, CoroutineClosureArgs,
    CoroutineClosureArgsParts, Interner, TypeSuperVisitable, TypeVisitable, TypeVisitableExt,
    TypeVisitor,
    inherent::{BoundExistentialPredicates, GenericArgs as _, IntoKind, SliceLike, Ty as _},
};
use tracing::debug;

use crate::{
    FnAbi,
    db::{InternedClosure, InternedCoroutine},
    infer::{BreakableKind, Diverges, coerce::CoerceMany},
    next_solver::{
        AliasTy, Binder, BoundRegionKind, BoundVarKind, BoundVarKinds, ClauseKind, DbInterner,
        ErrorGuaranteed, FnSig, GenericArgs, PolyFnSig, PolyProjectionPredicate, Predicate,
        PredicateKind, SolverDefId, Ty, TyKind,
        abi::Safety,
        infer::{
            BoundRegionConversionTime, InferOk, InferResult,
            traits::{ObligationCause, PredicateObligations},
        },
    },
    traits::FnTrait,
};

use super::{Expectation, InferenceContext};

#[derive(Debug)]
struct ClosureSignatures<'db> {
    /// The signature users of the closure see.
    bound_sig: PolyFnSig<'db>,
    /// The signature within the function body.
    /// This mostly differs in the sense that lifetimes are now early bound and any
    /// opaque types from the signature expectation are overridden in case there are
    /// explicit hidden types written by the user in the closure signature.
    liberated_sig: FnSig<'db>,
}

impl<'db> InferenceContext<'_, 'db> {
    pub(super) fn infer_closure(
        &mut self,
        body: ExprId,
        args: &[PatId],
        ret_type: Option<TypeRefId>,
        arg_types: &[Option<TypeRefId>],
        closure_kind: ClosureKind,
        tgt_expr: ExprId,
        expected: &Expectation<'db>,
    ) -> Ty<'db> {
        assert_eq!(args.len(), arg_types.len());

        let interner = self.interner();
        let (expected_sig, expected_kind) = match expected.to_option(&mut self.table) {
            Some(expected_ty) => self.deduce_closure_signature(expected_ty, closure_kind),
            None => (None, None),
        };

        let ClosureSignatures { bound_sig, liberated_sig } =
            self.sig_of_closure(arg_types, ret_type, expected_sig);
        let body_ret_ty = bound_sig.output().skip_binder();
        let sig_ty = Ty::new_fn_ptr(interner, bound_sig);

        let parent_args = GenericArgs::identity_for_item(interner, self.generic_def.into());
        // FIXME: Make this an infer var and infer it later.
        let tupled_upvars_ty = self.types.unit;
        let (id, ty, resume_yield_tys) = match closure_kind {
            ClosureKind::Coroutine(_) => {
                let yield_ty = self.table.next_ty_var();
                let resume_ty = liberated_sig.inputs().get(0).unwrap_or(self.types.unit);

                // FIXME: Infer the upvars later.
                let parts = CoroutineArgsParts {
                    parent_args,
                    kind_ty: self.types.unit,
                    resume_ty,
                    yield_ty,
                    return_ty: body_ret_ty,
                    tupled_upvars_ty,
                };

                let coroutine_id =
                    self.db.intern_coroutine(InternedCoroutine(self.owner, tgt_expr)).into();
                let coroutine_ty = Ty::new_coroutine(
                    interner,
                    coroutine_id,
                    CoroutineArgs::new(interner, parts).args,
                );

                (None, coroutine_ty, Some((resume_ty, yield_ty)))
            }
            ClosureKind::Closure => {
                let closure_id = self.db.intern_closure(InternedClosure(self.owner, tgt_expr));
                match expected_kind {
                    Some(kind) => {
                        self.result.closure_info.insert(
                            closure_id,
                            (
                                Vec::new(),
                                match kind {
                                    rustc_type_ir::ClosureKind::Fn => FnTrait::Fn,
                                    rustc_type_ir::ClosureKind::FnMut => FnTrait::FnMut,
                                    rustc_type_ir::ClosureKind::FnOnce => FnTrait::FnOnce,
                                },
                            ),
                        );
                    }
                    None => {}
                };
                // FIXME: Infer the kind later if needed.
                let parts = ClosureArgsParts {
                    parent_args,
                    closure_kind_ty: Ty::from_closure_kind(
                        interner,
                        expected_kind.unwrap_or(rustc_type_ir::ClosureKind::Fn),
                    ),
                    closure_sig_as_fn_ptr_ty: sig_ty,
                    tupled_upvars_ty,
                };
                let closure_ty = Ty::new_closure(
                    interner,
                    closure_id.into(),
                    ClosureArgs::new(interner, parts).args,
                );
                self.deferred_closures.entry(closure_id).or_default();
                self.add_current_closure_dependency(closure_id);
                (Some(closure_id), closure_ty, None)
            }
            ClosureKind::Async => {
                // async closures always return the type ascribed after the `->` (if present),
                // and yield `()`.
                let bound_return_ty = bound_sig.skip_binder().output();
                let bound_yield_ty = self.types.unit;
                // rustc uses a special lang item type for the resume ty. I don't believe this can cause us problems.
                let resume_ty = self.types.unit;

                // FIXME: Infer the kind later if needed.
                let closure_kind_ty = Ty::from_closure_kind(
                    interner,
                    expected_kind.unwrap_or(rustc_type_ir::ClosureKind::Fn),
                );

                // FIXME: Infer captures later.
                // `for<'env> fn() -> ()`, for no captures.
                let coroutine_captures_by_ref_ty = Ty::new_fn_ptr(
                    interner,
                    Binder::bind_with_vars(
                        interner.mk_fn_sig([], self.types.unit, false, Safety::Safe, FnAbi::Rust),
                        BoundVarKinds::new_from_iter(
                            interner,
                            [BoundVarKind::Region(BoundRegionKind::ClosureEnv)],
                        ),
                    ),
                );
                let closure_args = CoroutineClosureArgs::new(
                    interner,
                    CoroutineClosureArgsParts {
                        parent_args,
                        closure_kind_ty,
                        signature_parts_ty: Ty::new_fn_ptr(
                            interner,
                            bound_sig.map_bound(|sig| {
                                interner.mk_fn_sig(
                                    [
                                        resume_ty,
                                        Ty::new_tup_from_iter(interner, sig.inputs().iter()),
                                    ],
                                    Ty::new_tup(interner, &[bound_yield_ty, bound_return_ty]),
                                    sig.c_variadic,
                                    sig.safety,
                                    sig.abi,
                                )
                            }),
                        ),
                        tupled_upvars_ty,
                        coroutine_captures_by_ref_ty,
                    },
                );

                let coroutine_id =
                    self.db.intern_coroutine(InternedCoroutine(self.owner, tgt_expr)).into();
                (None, Ty::new_coroutine_closure(interner, coroutine_id, closure_args.args), None)
            }
        };

        // Now go through the argument patterns
        for (arg_pat, arg_ty) in args.iter().zip(bound_sig.skip_binder().inputs()) {
            self.infer_top_pat(*arg_pat, arg_ty, None);
        }

        // FIXME: lift these out into a struct
        let prev_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
        let prev_closure = mem::replace(&mut self.current_closure, id);
        let prev_ret_ty = mem::replace(&mut self.return_ty, body_ret_ty);
        let prev_ret_coercion = self.return_coercion.replace(CoerceMany::new(body_ret_ty));
        let prev_resume_yield_tys = mem::replace(&mut self.resume_yield_tys, resume_yield_tys);

        self.with_breakable_ctx(BreakableKind::Border, None, None, |this| {
            this.infer_return(body);
        });

        self.diverges = prev_diverges;
        self.return_ty = prev_ret_ty;
        self.return_coercion = prev_ret_coercion;
        self.current_closure = prev_closure;
        self.resume_yield_tys = prev_resume_yield_tys;

        ty
    }

    fn fn_trait_kind_from_def_id(&self, trait_id: TraitId) -> Option<rustc_type_ir::ClosureKind> {
        match trait_id {
            _ if self.lang_items.Fn == Some(trait_id) => Some(rustc_type_ir::ClosureKind::Fn),
            _ if self.lang_items.FnMut == Some(trait_id) => Some(rustc_type_ir::ClosureKind::FnMut),
            _ if self.lang_items.FnOnce == Some(trait_id) => {
                Some(rustc_type_ir::ClosureKind::FnOnce)
            }
            _ => None,
        }
    }

    fn async_fn_trait_kind_from_def_id(
        &self,
        trait_id: TraitId,
    ) -> Option<rustc_type_ir::ClosureKind> {
        match trait_id {
            _ if self.lang_items.AsyncFn == Some(trait_id) => Some(rustc_type_ir::ClosureKind::Fn),
            _ if self.lang_items.AsyncFnMut == Some(trait_id) => {
                Some(rustc_type_ir::ClosureKind::FnMut)
            }
            _ if self.lang_items.AsyncFnOnce == Some(trait_id) => {
                Some(rustc_type_ir::ClosureKind::FnOnce)
            }
            _ => None,
        }
    }

    /// Given the expected type, figures out what it can about this closure we
    /// are about to type check:
    fn deduce_closure_signature(
        &mut self,
        expected_ty: Ty<'db>,
        closure_kind: ClosureKind,
    ) -> (Option<PolyFnSig<'db>>, Option<rustc_type_ir::ClosureKind>) {
        match expected_ty.kind() {
            TyKind::Alias(rustc_type_ir::Opaque, AliasTy { def_id, args, .. }) => self
                .deduce_closure_signature_from_predicates(
                    expected_ty,
                    closure_kind,
                    def_id
                        .expect_opaque_ty()
                        .predicates(self.db)
                        .iter_instantiated_copied(self.interner(), args.as_slice())
                        .map(|clause| clause.as_predicate()),
                ),
            TyKind::Dynamic(object_type, ..) => {
                let sig = object_type.projection_bounds().into_iter().find_map(|pb| {
                    let pb = pb.with_self_ty(self.interner(), Ty::new_unit(self.interner()));
                    self.deduce_sig_from_projection(closure_kind, pb)
                });
                let kind = object_type
                    .principal_def_id()
                    .and_then(|did| self.fn_trait_kind_from_def_id(did.0));
                (sig, kind)
            }
            TyKind::Infer(rustc_type_ir::TyVar(vid)) => self
                .deduce_closure_signature_from_predicates(
                    Ty::new_var(self.interner(), self.table.infer_ctxt.root_var(vid)),
                    closure_kind,
                    self.table.obligations_for_self_ty(vid).into_iter().map(|obl| obl.predicate),
                ),
            TyKind::FnPtr(sig_tys, hdr) => match closure_kind {
                ClosureKind::Closure => {
                    let expected_sig = sig_tys.with(hdr);
                    (Some(expected_sig), Some(rustc_type_ir::ClosureKind::Fn))
                }
                ClosureKind::Coroutine(_) | ClosureKind::Async => (None, None),
            },
            _ => (None, None),
        }
    }

    fn deduce_closure_signature_from_predicates(
        &mut self,
        expected_ty: Ty<'db>,
        closure_kind: ClosureKind,
        predicates: impl DoubleEndedIterator<Item = Predicate<'db>>,
    ) -> (Option<PolyFnSig<'db>>, Option<rustc_type_ir::ClosureKind>) {
        let mut expected_sig = None;
        let mut expected_kind = None;

        for pred in rustc_type_ir::elaborate::elaborate(
            self.interner(),
            // Reverse the obligations here, since `elaborate_*` uses a stack,
            // and we want to keep inference generally in the same order of
            // the registered obligations.
            predicates.rev(),
        )
        // We only care about self bounds
        .filter_only_self()
        {
            debug!(?pred);
            let bound_predicate = pred.kind();

            // Given a Projection predicate, we can potentially infer
            // the complete signature.
            if expected_sig.is_none()
                && let PredicateKind::Clause(ClauseKind::Projection(proj_predicate)) =
                    bound_predicate.skip_binder()
            {
                let inferred_sig = self.deduce_sig_from_projection(
                    closure_kind,
                    bound_predicate.rebind(proj_predicate),
                );

                // Make sure that we didn't infer a signature that mentions itself.
                // This can happen when we elaborate certain supertrait bounds that
                // mention projections containing the `Self` type. See #105401.
                struct MentionsTy<'db> {
                    expected_ty: Ty<'db>,
                }
                impl<'db> TypeVisitor<DbInterner<'db>> for MentionsTy<'db> {
                    type Result = ControlFlow<()>;

                    fn visit_ty(&mut self, t: Ty<'db>) -> Self::Result {
                        if t == self.expected_ty {
                            ControlFlow::Break(())
                        } else {
                            t.super_visit_with(self)
                        }
                    }
                }

                // Don't infer a closure signature from a goal that names the closure type as this will
                // (almost always) lead to occurs check errors later in type checking.
                if let Some(inferred_sig) = inferred_sig {
                    // In the new solver it is difficult to explicitly normalize the inferred signature as we
                    // would have to manually handle universes and rewriting bound vars and placeholders back
                    // and forth.
                    //
                    // Instead we take advantage of the fact that we relating an inference variable with an alias
                    // will only instantiate the variable if the alias is rigid(*not quite). Concretely we:
                    // - Create some new variable `?sig`
                    // - Equate `?sig` with the unnormalized signature, e.g. `fn(<Foo<?x> as Trait>::Assoc)`
                    // - Depending on whether `<Foo<?x> as Trait>::Assoc` is rigid, ambiguous or normalizeable,
                    //   we will either wind up with `?sig=<Foo<?x> as Trait>::Assoc/?y/ConcreteTy` respectively.
                    //
                    // *: In cases where there are ambiguous aliases in the signature that make use of bound vars
                    //    they will wind up present in `?sig` even though they are non-rigid.
                    //
                    //    This is a bit weird and means we may wind up discarding the goal due to it naming `expected_ty`
                    //    even though the normalized form may not name `expected_ty`. However, this matches the existing
                    //    behaviour of the old solver and would be technically a breaking change to fix.
                    let generalized_fnptr_sig = self.table.next_ty_var();
                    let inferred_fnptr_sig = Ty::new_fn_ptr(self.interner(), inferred_sig);
                    // FIXME: Report diagnostics.
                    _ = self
                        .table
                        .infer_ctxt
                        .at(&ObligationCause::new(), self.table.param_env)
                        .eq(inferred_fnptr_sig, generalized_fnptr_sig)
                        .map(|infer_ok| self.table.register_infer_ok(infer_ok));

                    let resolved_sig =
                        self.table.infer_ctxt.resolve_vars_if_possible(generalized_fnptr_sig);

                    if resolved_sig.visit_with(&mut MentionsTy { expected_ty }).is_continue() {
                        expected_sig = Some(resolved_sig.fn_sig(self.interner()));
                    }
                } else if inferred_sig.visit_with(&mut MentionsTy { expected_ty }).is_continue() {
                    expected_sig = inferred_sig;
                }
            }

            // Even if we can't infer the full signature, we may be able to
            // infer the kind. This can occur when we elaborate a predicate
            // like `F : Fn<A>`. Note that due to subtyping we could encounter
            // many viable options, so pick the most restrictive.
            let trait_def_id = match bound_predicate.skip_binder() {
                PredicateKind::Clause(ClauseKind::Projection(data)) => {
                    Some(data.projection_term.trait_def_id(self.interner()).0)
                }
                PredicateKind::Clause(ClauseKind::Trait(data)) => Some(data.def_id().0),
                _ => None,
            };

            if let Some(trait_def_id) = trait_def_id {
                let found_kind = match closure_kind {
                    ClosureKind::Closure => self.fn_trait_kind_from_def_id(trait_def_id),
                    ClosureKind::Async => self
                        .async_fn_trait_kind_from_def_id(trait_def_id)
                        .or_else(|| self.fn_trait_kind_from_def_id(trait_def_id)),
                    _ => None,
                };

                if let Some(found_kind) = found_kind {
                    // always use the closure kind that is more permissive.
                    match (expected_kind, found_kind) {
                        (None, _) => expected_kind = Some(found_kind),
                        (
                            Some(rustc_type_ir::ClosureKind::FnMut),
                            rustc_type_ir::ClosureKind::Fn,
                        ) => expected_kind = Some(rustc_type_ir::ClosureKind::Fn),
                        (
                            Some(rustc_type_ir::ClosureKind::FnOnce),
                            rustc_type_ir::ClosureKind::Fn | rustc_type_ir::ClosureKind::FnMut,
                        ) => expected_kind = Some(found_kind),
                        _ => {}
                    }
                }
            }
        }

        (expected_sig, expected_kind)
    }

    /// Given a projection like "<F as Fn(X)>::Result == Y", we can deduce
    /// everything we need to know about a closure or coroutine.
    ///
    /// The `cause_span` should be the span that caused us to
    /// have this expected signature, or `None` if we can't readily
    /// know that.
    fn deduce_sig_from_projection(
        &mut self,
        closure_kind: ClosureKind,
        projection: PolyProjectionPredicate<'db>,
    ) -> Option<PolyFnSig<'db>> {
        let SolverDefId::TypeAliasId(def_id) = projection.item_def_id() else { unreachable!() };

        // For now, we only do signature deduction based off of the `Fn` and `AsyncFn` traits,
        // for closures and async closures, respectively.
        match closure_kind {
            ClosureKind::Closure if Some(def_id) == self.lang_items.FnOnceOutput => {
                self.extract_sig_from_projection(projection)
            }
            ClosureKind::Async if Some(def_id) == self.lang_items.AsyncFnOnceOutput => {
                self.extract_sig_from_projection(projection)
            }
            // It's possible we've passed the closure to a (somewhat out-of-fashion)
            // `F: FnOnce() -> Fut, Fut: Future<Output = T>` style bound. Let's still
            // guide inference here, since it's beneficial for the user.
            ClosureKind::Async if Some(def_id) == self.lang_items.FnOnceOutput => {
                self.extract_sig_from_projection_and_future_bound(projection)
            }
            _ => None,
        }
    }

    /// Given an `FnOnce::Output` or `AsyncFn::Output` projection, extract the args
    /// and return type to infer a [`ty::PolyFnSig`] for the closure.
    fn extract_sig_from_projection(
        &self,
        projection: PolyProjectionPredicate<'db>,
    ) -> Option<PolyFnSig<'db>> {
        let projection = self.table.infer_ctxt.resolve_vars_if_possible(projection);

        let arg_param_ty = projection.skip_binder().projection_term.args.type_at(1);
        debug!(?arg_param_ty);

        let TyKind::Tuple(input_tys) = arg_param_ty.kind() else {
            return None;
        };

        // Since this is a return parameter type it is safe to unwrap.
        let ret_param_ty = projection.skip_binder().term.expect_type();
        debug!(?ret_param_ty);

        let sig = projection.rebind(self.interner().mk_fn_sig(
            input_tys,
            ret_param_ty,
            false,
            Safety::Safe,
            FnAbi::Rust,
        ));

        Some(sig)
    }

    /// When an async closure is passed to a function that has a "two-part" `Fn`
    /// and `Future` trait bound, like:
    ///
    /// ```rust
    /// use std::future::Future;
    ///
    /// fn not_exactly_an_async_closure<F, Fut>(_f: F)
    /// where
    ///     F: FnOnce(String, u32) -> Fut,
    ///     Fut: Future<Output = i32>,
    /// {}
    /// ```
    ///
    /// The we want to be able to extract the signature to guide inference in the async
    /// closure. We will have two projection predicates registered in this case. First,
    /// we identify the `FnOnce<Args, Output = ?Fut>` bound, and if the output type is
    /// an inference variable `?Fut`, we check if that is bounded by a `Future<Output = Ty>`
    /// projection.
    ///
    /// This function is actually best-effort with the return type; if we don't find a
    /// `Future` projection, we still will return arguments that we extracted from the `FnOnce`
    /// projection, and the output will be an unconstrained type variable instead.
    fn extract_sig_from_projection_and_future_bound(
        &mut self,
        projection: PolyProjectionPredicate<'db>,
    ) -> Option<PolyFnSig<'db>> {
        let projection = self.table.infer_ctxt.resolve_vars_if_possible(projection);

        let arg_param_ty = projection.skip_binder().projection_term.args.type_at(1);
        debug!(?arg_param_ty);

        let TyKind::Tuple(input_tys) = arg_param_ty.kind() else {
            return None;
        };

        // If the return type is a type variable, look for bounds on it.
        // We could theoretically support other kinds of return types here,
        // but none of them would be useful, since async closures return
        // concrete anonymous future types, and their futures are not coerced
        // into any other type within the body of the async closure.
        let TyKind::Infer(rustc_type_ir::TyVar(return_vid)) =
            projection.skip_binder().term.expect_type().kind()
        else {
            return None;
        };

        // FIXME: We may want to elaborate here, though I assume this will be exceedingly rare.
        let mut return_ty = None;
        for bound in self.table.obligations_for_self_ty(return_vid) {
            if let PredicateKind::Clause(ClauseKind::Projection(ret_projection)) =
                bound.predicate.kind().skip_binder()
                && let ret_projection = bound.predicate.kind().rebind(ret_projection)
                && let Some(ret_projection) = ret_projection.no_bound_vars()
                && let SolverDefId::TypeAliasId(assoc_type) = ret_projection.def_id()
                && Some(assoc_type) == self.lang_items.FutureOutput
            {
                return_ty = Some(ret_projection.term.expect_type());
                break;
            }
        }

        // SUBTLE: If we didn't find a `Future<Output = ...>` bound for the return
        // vid, we still want to attempt to provide inference guidance for the async
        // closure's arguments. Instantiate a new vid to plug into the output type.
        //
        // You may be wondering, what if it's higher-ranked? Well, given that we
        // found a type variable for the `FnOnce::Output` projection above, we know
        // that the output can't mention any of the vars.
        //
        // Also note that we use a fresh var here for the signature since the signature
        // records the output of the *future*, and `return_vid` above is the type
        // variable of the future, not its output.
        //
        // FIXME: We probably should store this signature inference output in a way
        // that does not misuse a `FnSig` type, but that can be done separately.
        let return_ty = return_ty.unwrap_or_else(|| self.table.next_ty_var());

        let sig = projection.rebind(self.interner().mk_fn_sig(
            input_tys,
            return_ty,
            false,
            Safety::Safe,
            FnAbi::Rust,
        ));

        Some(sig)
    }

    fn sig_of_closure(
        &mut self,
        decl_inputs: &[Option<TypeRefId>],
        decl_output: Option<TypeRefId>,
        expected_sig: Option<PolyFnSig<'db>>,
    ) -> ClosureSignatures<'db> {
        if let Some(e) = expected_sig {
            self.sig_of_closure_with_expectation(decl_inputs, decl_output, e)
        } else {
            self.sig_of_closure_no_expectation(decl_inputs, decl_output)
        }
    }

    /// If there is no expected signature, then we will convert the
    /// types that the user gave into a signature.
    fn sig_of_closure_no_expectation(
        &mut self,
        decl_inputs: &[Option<TypeRefId>],
        decl_output: Option<TypeRefId>,
    ) -> ClosureSignatures<'db> {
        let bound_sig = self.supplied_sig_of_closure(decl_inputs, decl_output);

        self.closure_sigs(bound_sig)
    }

    /// Invoked to compute the signature of a closure expression. This
    /// combines any user-provided type annotations (e.g., `|x: u32|
    /// -> u32 { .. }`) with the expected signature.
    ///
    /// The approach is as follows:
    ///
    /// - Let `S` be the (higher-ranked) signature that we derive from the user's annotations.
    /// - Let `E` be the (higher-ranked) signature that we derive from the expectations, if any.
    ///   - If we have no expectation `E`, then the signature of the closure is `S`.
    ///   - Otherwise, the signature of the closure is E. Moreover:
    ///     - Skolemize the late-bound regions in `E`, yielding `E'`.
    ///     - Instantiate all the late-bound regions bound in the closure within `S`
    ///       with fresh (existential) variables, yielding `S'`
    ///     - Require that `E' = S'`
    ///       - We could use some kind of subtyping relationship here,
    ///         I imagine, but equality is easier and works fine for
    ///         our purposes.
    ///
    /// The key intuition here is that the user's types must be valid
    /// from "the inside" of the closure, but the expectation
    /// ultimately drives the overall signature.
    ///
    /// # Examples
    ///
    /// ```ignore (illustrative)
    /// fn with_closure<F>(_: F)
    ///   where F: Fn(&u32) -> &u32 { .. }
    ///
    /// with_closure(|x: &u32| { ... })
    /// ```
    ///
    /// Here:
    /// - E would be `fn(&u32) -> &u32`.
    /// - S would be `fn(&u32) -> ?T`
    /// - E' is `&'!0 u32 -> &'!0 u32`
    /// - S' is `&'?0 u32 -> ?T`
    ///
    /// S' can be unified with E' with `['?0 = '!0, ?T = &'!10 u32]`.
    ///
    /// # Arguments
    ///
    /// - `expr_def_id`: the `LocalDefId` of the closure expression
    /// - `decl`: the HIR declaration of the closure
    /// - `body`: the body of the closure
    /// - `expected_sig`: the expected signature (if any). Note that
    ///   this is missing a binder: that is, there may be late-bound
    ///   regions with depth 1, which are bound then by the closure.
    fn sig_of_closure_with_expectation(
        &mut self,
        decl_inputs: &[Option<TypeRefId>],
        decl_output: Option<TypeRefId>,
        expected_sig: PolyFnSig<'db>,
    ) -> ClosureSignatures<'db> {
        // Watch out for some surprises and just ignore the
        // expectation if things don't see to match up with what we
        // expect.
        if expected_sig.c_variadic() {
            return self.sig_of_closure_no_expectation(decl_inputs, decl_output);
        } else if expected_sig.skip_binder().inputs_and_output.len() != decl_inputs.len() + 1 {
            return self
                .sig_of_closure_with_mismatched_number_of_arguments(decl_inputs, decl_output);
        }

        // Create a `PolyFnSig`. Note the oddity that late bound
        // regions appearing free in `expected_sig` are now bound up
        // in this binder we are creating.
        assert!(!expected_sig.skip_binder().has_vars_bound_above(rustc_type_ir::INNERMOST));
        let bound_sig = expected_sig.map_bound(|sig| {
            self.interner().mk_fn_sig(
                sig.inputs(),
                sig.output(),
                sig.c_variadic,
                Safety::Safe,
                FnAbi::RustCall,
            )
        });

        // `deduce_expectations_from_expected_type` introduces
        // late-bound lifetimes defined elsewhere, which we now
        // anonymize away, so as not to confuse the user.
        let bound_sig = self.interner().anonymize_bound_vars(bound_sig);

        let closure_sigs = self.closure_sigs(bound_sig);

        // Up till this point, we have ignored the annotations that the user
        // gave. This function will check that they unify successfully.
        // Along the way, it also writes out entries for types that the user
        // wrote into our typeck results, which are then later used by the privacy
        // check.
        match self.merge_supplied_sig_with_expectation(decl_inputs, decl_output, closure_sigs) {
            Ok(infer_ok) => self.table.register_infer_ok(infer_ok),
            Err(_) => self.sig_of_closure_no_expectation(decl_inputs, decl_output),
        }
    }

    fn sig_of_closure_with_mismatched_number_of_arguments(
        &mut self,
        decl_inputs: &[Option<TypeRefId>],
        decl_output: Option<TypeRefId>,
    ) -> ClosureSignatures<'db> {
        let error_sig = self.error_sig_of_closure(decl_inputs, decl_output);

        self.closure_sigs(error_sig)
    }

    /// Enforce the user's types against the expectation. See
    /// `sig_of_closure_with_expectation` for details on the overall
    /// strategy.
    fn merge_supplied_sig_with_expectation(
        &mut self,
        decl_inputs: &[Option<TypeRefId>],
        decl_output: Option<TypeRefId>,
        mut expected_sigs: ClosureSignatures<'db>,
    ) -> InferResult<'db, ClosureSignatures<'db>> {
        // Get the signature S that the user gave.
        //
        // (See comment on `sig_of_closure_with_expectation` for the
        // meaning of these letters.)
        let supplied_sig = self.supplied_sig_of_closure(decl_inputs, decl_output);

        debug!(?supplied_sig);

        // FIXME(#45727): As discussed in [this comment][c1], naively
        // forcing equality here actually results in suboptimal error
        // messages in some cases. For now, if there would have been
        // an obvious error, we fallback to declaring the type of the
        // closure to be the one the user gave, which allows other
        // error message code to trigger.
        //
        // However, I think [there is potential to do even better
        // here][c2], since in *this* code we have the precise span of
        // the type parameter in question in hand when we report the
        // error.
        //
        // [c1]: https://github.com/rust-lang/rust/pull/45072#issuecomment-341089706
        // [c2]: https://github.com/rust-lang/rust/pull/45072#issuecomment-341096796
        self.table.commit_if_ok(|table| {
            let mut all_obligations = PredicateObligations::new();
            let supplied_sig = table.infer_ctxt.instantiate_binder_with_fresh_vars(
                BoundRegionConversionTime::FnCall,
                supplied_sig,
            );

            // The liberated version of this signature should be a subtype
            // of the liberated form of the expectation.
            for (supplied_ty, expected_ty) in
                iter::zip(supplied_sig.inputs(), expected_sigs.liberated_sig.inputs())
            {
                // Check that E' = S'.
                let cause = ObligationCause::new();
                let InferOk { value: (), obligations } =
                    table.infer_ctxt.at(&cause, table.param_env).eq(expected_ty, supplied_ty)?;
                all_obligations.extend(obligations);
            }

            let supplied_output_ty = supplied_sig.output();
            let cause = ObligationCause::new();
            let InferOk { value: (), obligations } =
                table
                    .infer_ctxt
                    .at(&cause, table.param_env)
                    .eq(expected_sigs.liberated_sig.output(), supplied_output_ty)?;
            all_obligations.extend(obligations);

            let inputs = supplied_sig
                .inputs()
                .into_iter()
                .map(|ty| table.infer_ctxt.resolve_vars_if_possible(ty));

            expected_sigs.liberated_sig = table.interner().mk_fn_sig(
                inputs,
                supplied_output_ty,
                expected_sigs.liberated_sig.c_variadic,
                Safety::Safe,
                FnAbi::RustCall,
            );

            Ok(InferOk { value: expected_sigs, obligations: all_obligations })
        })
    }

    /// If there is no expected signature, then we will convert the
    /// types that the user gave into a signature.
    ///
    /// Also, record this closure signature for later.
    fn supplied_sig_of_closure(
        &mut self,
        decl_inputs: &[Option<TypeRefId>],
        decl_output: Option<TypeRefId>,
    ) -> PolyFnSig<'db> {
        let interner = self.interner();

        let supplied_return = match decl_output {
            Some(output) => {
                let output = self.make_body_ty(output);
                self.process_user_written_ty(output)
            }
            None => self.table.next_ty_var(),
        };
        // First, convert the types that the user supplied (if any).
        let supplied_arguments = decl_inputs.iter().map(|&input| match input {
            Some(input) => {
                let input = self.make_body_ty(input);
                self.process_user_written_ty(input)
            }
            None => self.table.next_ty_var(),
        });

        Binder::dummy(interner.mk_fn_sig(
            supplied_arguments,
            supplied_return,
            false,
            Safety::Safe,
            FnAbi::RustCall,
        ))
    }

    /// Converts the types that the user supplied, in case that doing
    /// so should yield an error, but returns back a signature where
    /// all parameters are of type `ty::Error`.
    fn error_sig_of_closure(
        &mut self,
        decl_inputs: &[Option<TypeRefId>],
        decl_output: Option<TypeRefId>,
    ) -> PolyFnSig<'db> {
        let interner = self.interner();
        let err_ty = Ty::new_error(interner, ErrorGuaranteed);

        if let Some(output) = decl_output {
            self.make_body_ty(output);
        }
        let supplied_arguments = decl_inputs.iter().map(|&input| match input {
            Some(input) => {
                self.make_body_ty(input);
                err_ty
            }
            None => err_ty,
        });

        let result = Binder::dummy(interner.mk_fn_sig(
            supplied_arguments,
            err_ty,
            false,
            Safety::Safe,
            FnAbi::RustCall,
        ));

        debug!("supplied_sig_of_closure: result={:?}", result);

        result
    }

    fn closure_sigs(&self, bound_sig: PolyFnSig<'db>) -> ClosureSignatures<'db> {
        let liberated_sig = bound_sig.skip_binder();
        // FIXME: When we lower HRTB we'll need to actually liberate regions here.
        ClosureSignatures { bound_sig, liberated_sig }
    }
}
