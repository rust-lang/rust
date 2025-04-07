//! Code for type-checking closure expressions.

use std::iter;
use std::ops::ControlFlow;

use rustc_abi::ExternAbi;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::lang_items::LangItem;
use rustc_hir_analysis::hir_ty_lowering::HirTyLowerer;
use rustc_infer::infer::{BoundRegionConversionTime, DefineOpaqueTypes, InferOk, InferResult};
use rustc_infer::traits::{ObligationCauseCode, PredicateObligations};
use rustc_macros::{TypeFoldable, TypeVisitable};
use rustc_middle::span_bug;
use rustc_middle::ty::{
    self, ClosureKind, GenericArgs, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable,
    TypeVisitableExt, TypeVisitor,
};
use rustc_span::def_id::LocalDefId;
use rustc_span::{DUMMY_SP, Span};
use rustc_trait_selection::error_reporting::traits::ArgKind;
use rustc_trait_selection::traits;
use tracing::{debug, instrument, trace};

use super::{CoroutineTypes, Expectation, FnCtxt, check_fn};

/// What signature do we *expect* the closure to have from context?
#[derive(Debug, Clone, TypeFoldable, TypeVisitable)]
struct ExpectedSig<'tcx> {
    /// Span that gave us this expectation, if we know that.
    cause_span: Option<Span>,
    sig: ty::PolyFnSig<'tcx>,
}

#[derive(Debug)]
struct ClosureSignatures<'tcx> {
    /// The signature users of the closure see.
    bound_sig: ty::PolyFnSig<'tcx>,
    /// The signature within the function body.
    /// This mostly differs in the sense that lifetimes are now early bound and any
    /// opaque types from the signature expectation are overridden in case there are
    /// explicit hidden types written by the user in the closure signature.
    liberated_sig: ty::FnSig<'tcx>,
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    #[instrument(skip(self, closure), level = "debug")]
    pub(crate) fn check_expr_closure(
        &self,
        closure: &hir::Closure<'tcx>,
        expr_span: Span,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let body = tcx.hir_body(closure.body);
        let expr_def_id = closure.def_id;

        // It's always helpful for inference if we know the kind of
        // closure sooner rather than later, so first examine the expected
        // type, and see if can glean a closure kind from there.
        let (expected_sig, expected_kind) = match expected.to_option(self) {
            Some(ty) => self.deduce_closure_signature(
                self.try_structurally_resolve_type(expr_span, ty),
                closure.kind,
            ),
            None => (None, None),
        };

        let ClosureSignatures { bound_sig, mut liberated_sig } =
            self.sig_of_closure(expr_def_id, closure.fn_decl, closure.kind, expected_sig);

        debug!(?bound_sig, ?liberated_sig);

        let parent_args =
            GenericArgs::identity_for_item(tcx, tcx.typeck_root_def_id(expr_def_id.to_def_id()));

        let tupled_upvars_ty = self.next_ty_var(expr_span);

        // FIXME: We could probably actually just unify this further --
        // instead of having a `FnSig` and a `Option<CoroutineTypes>`,
        // we can have a `ClosureSignature { Coroutine { .. }, Closure { .. } }`,
        // similar to how `ty::GenSig` is a distinct data structure.
        let (closure_ty, coroutine_types) = match closure.kind {
            hir::ClosureKind::Closure => {
                // Tuple up the arguments and insert the resulting function type into
                // the `closures` table.
                let sig = bound_sig.map_bound(|sig| {
                    tcx.mk_fn_sig(
                        [Ty::new_tup(tcx, sig.inputs())],
                        sig.output(),
                        sig.c_variadic,
                        sig.safety,
                        sig.abi,
                    )
                });

                debug!(?sig, ?expected_kind);

                let closure_kind_ty = match expected_kind {
                    Some(kind) => Ty::from_closure_kind(tcx, kind),

                    // Create a type variable (for now) to represent the closure kind.
                    // It will be unified during the upvar inference phase (`upvar.rs`)
                    None => self.next_ty_var(expr_span),
                };

                let closure_args = ty::ClosureArgs::new(
                    tcx,
                    ty::ClosureArgsParts {
                        parent_args,
                        closure_kind_ty,
                        closure_sig_as_fn_ptr_ty: Ty::new_fn_ptr(tcx, sig),
                        tupled_upvars_ty,
                    },
                );

                (Ty::new_closure(tcx, expr_def_id.to_def_id(), closure_args.args), None)
            }
            hir::ClosureKind::Coroutine(kind) => {
                let yield_ty = match kind {
                    hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen, _)
                    | hir::CoroutineKind::Coroutine(_) => {
                        let yield_ty = self.next_ty_var(expr_span);
                        self.require_type_is_sized(
                            yield_ty,
                            expr_span,
                            ObligationCauseCode::SizedYieldType,
                        );
                        yield_ty
                    }
                    // HACK(-Ztrait-solver=next): In the *old* trait solver, we must eagerly
                    // guide inference on the yield type so that we can handle `AsyncIterator`
                    // in this block in projection correctly. In the new trait solver, it is
                    // not a problem.
                    hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::AsyncGen, _) => {
                        let yield_ty = self.next_ty_var(expr_span);
                        self.require_type_is_sized(
                            yield_ty,
                            expr_span,
                            ObligationCauseCode::SizedYieldType,
                        );

                        Ty::new_adt(
                            tcx,
                            tcx.adt_def(
                                tcx.require_lang_item(hir::LangItem::Poll, Some(expr_span)),
                            ),
                            tcx.mk_args(&[Ty::new_adt(
                                tcx,
                                tcx.adt_def(
                                    tcx.require_lang_item(hir::LangItem::Option, Some(expr_span)),
                                ),
                                tcx.mk_args(&[yield_ty.into()]),
                            )
                            .into()]),
                        )
                    }
                    hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async, _) => {
                        tcx.types.unit
                    }
                };

                // Resume type defaults to `()` if the coroutine has no argument.
                let resume_ty = liberated_sig.inputs().get(0).copied().unwrap_or(tcx.types.unit);

                let interior = self.next_ty_var(expr_span);
                self.deferred_coroutine_interiors.borrow_mut().push((expr_def_id, interior));

                // Coroutines that come from coroutine closures have not yet determined
                // their kind ty, so make a fresh infer var which will be constrained
                // later during upvar analysis. Regular coroutines always have the kind
                // ty of `().`
                let kind_ty = match kind {
                    hir::CoroutineKind::Desugared(_, hir::CoroutineSource::Closure) => {
                        self.next_ty_var(expr_span)
                    }
                    _ => tcx.types.unit,
                };

                let coroutine_args = ty::CoroutineArgs::new(
                    tcx,
                    ty::CoroutineArgsParts {
                        parent_args,
                        kind_ty,
                        resume_ty,
                        yield_ty,
                        return_ty: liberated_sig.output(),
                        witness: interior,
                        tupled_upvars_ty,
                    },
                );

                (
                    Ty::new_coroutine(tcx, expr_def_id.to_def_id(), coroutine_args.args),
                    Some(CoroutineTypes { resume_ty, yield_ty }),
                )
            }
            hir::ClosureKind::CoroutineClosure(kind) => {
                // async closures always return the type ascribed after the `->` (if present),
                // and yield `()`.
                let (bound_return_ty, bound_yield_ty) = match kind {
                    hir::CoroutineDesugaring::Async => {
                        (bound_sig.skip_binder().output(), tcx.types.unit)
                    }
                    hir::CoroutineDesugaring::Gen | hir::CoroutineDesugaring::AsyncGen => {
                        todo!("`gen` and `async gen` closures not supported yet")
                    }
                };
                // Compute all of the variables that will be used to populate the coroutine.
                let resume_ty = self.next_ty_var(expr_span);
                let interior = self.next_ty_var(expr_span);

                let closure_kind_ty = match expected_kind {
                    Some(kind) => Ty::from_closure_kind(tcx, kind),

                    // Create a type variable (for now) to represent the closure kind.
                    // It will be unified during the upvar inference phase (`upvar.rs`)
                    None => self.next_ty_var(expr_span),
                };

                let coroutine_captures_by_ref_ty = self.next_ty_var(expr_span);
                let closure_args = ty::CoroutineClosureArgs::new(
                    tcx,
                    ty::CoroutineClosureArgsParts {
                        parent_args,
                        closure_kind_ty,
                        signature_parts_ty: Ty::new_fn_ptr(
                            tcx,
                            bound_sig.map_bound(|sig| {
                                tcx.mk_fn_sig(
                                    [
                                        resume_ty,
                                        Ty::new_tup_from_iter(tcx, sig.inputs().iter().copied()),
                                    ],
                                    Ty::new_tup(tcx, &[bound_yield_ty, bound_return_ty]),
                                    sig.c_variadic,
                                    sig.safety,
                                    sig.abi,
                                )
                            }),
                        ),
                        tupled_upvars_ty,
                        coroutine_captures_by_ref_ty,
                        coroutine_witness_ty: interior,
                    },
                );

                let coroutine_kind_ty = match expected_kind {
                    Some(kind) => Ty::from_coroutine_closure_kind(tcx, kind),

                    // Create a type variable (for now) to represent the closure kind.
                    // It will be unified during the upvar inference phase (`upvar.rs`)
                    None => self.next_ty_var(expr_span),
                };

                let coroutine_upvars_ty = self.next_ty_var(expr_span);

                // We need to turn the liberated signature that we got from HIR, which
                // looks something like `|Args...| -> T`, into a signature that is suitable
                // for type checking the inner body of the closure, which always returns a
                // coroutine. To do so, we use the `CoroutineClosureSignature` to compute
                // the coroutine type, filling in the tupled_upvars_ty and kind_ty with infer
                // vars which will get constrained during upvar analysis.
                let coroutine_output_ty = tcx.liberate_late_bound_regions(
                    expr_def_id.to_def_id(),
                    closure_args.coroutine_closure_sig().map_bound(|sig| {
                        sig.to_coroutine(
                            tcx,
                            parent_args,
                            coroutine_kind_ty,
                            tcx.coroutine_for_closure(expr_def_id),
                            coroutine_upvars_ty,
                        )
                    }),
                );
                liberated_sig = tcx.mk_fn_sig(
                    liberated_sig.inputs().iter().copied(),
                    coroutine_output_ty,
                    liberated_sig.c_variadic,
                    liberated_sig.safety,
                    liberated_sig.abi,
                );

                (Ty::new_coroutine_closure(tcx, expr_def_id.to_def_id(), closure_args.args), None)
            }
        };

        check_fn(
            &mut FnCtxt::new(self, self.param_env, closure.def_id),
            liberated_sig,
            coroutine_types,
            closure.fn_decl,
            expr_def_id,
            body,
            // Closure "rust-call" ABI doesn't support unsized params
            false,
        );

        closure_ty
    }

    /// Given the expected type, figures out what it can about this closure we
    /// are about to type check:
    #[instrument(skip(self), level = "debug", ret)]
    fn deduce_closure_signature(
        &self,
        expected_ty: Ty<'tcx>,
        closure_kind: hir::ClosureKind,
    ) -> (Option<ExpectedSig<'tcx>>, Option<ty::ClosureKind>) {
        match *expected_ty.kind() {
            ty::Alias(ty::Opaque, ty::AliasTy { def_id, args, .. }) => self
                .deduce_closure_signature_from_predicates(
                    expected_ty,
                    closure_kind,
                    self.tcx
                        .explicit_item_self_bounds(def_id)
                        .iter_instantiated_copied(self.tcx, args)
                        .map(|(c, s)| (c.as_predicate(), s)),
                ),
            ty::Dynamic(object_type, ..) => {
                let sig = object_type.projection_bounds().find_map(|pb| {
                    let pb = pb.with_self_ty(self.tcx, self.tcx.types.trait_object_dummy_self);
                    self.deduce_sig_from_projection(None, closure_kind, pb)
                });
                let kind = object_type
                    .principal_def_id()
                    .and_then(|did| self.tcx.fn_trait_kind_from_def_id(did));
                (sig, kind)
            }
            ty::Infer(ty::TyVar(vid)) => self.deduce_closure_signature_from_predicates(
                Ty::new_var(self.tcx, self.root_var(vid)),
                closure_kind,
                self.obligations_for_self_ty(vid)
                    .into_iter()
                    .map(|obl| (obl.predicate, obl.cause.span)),
            ),
            ty::FnPtr(sig_tys, hdr) => match closure_kind {
                hir::ClosureKind::Closure => {
                    let expected_sig = ExpectedSig { cause_span: None, sig: sig_tys.with(hdr) };
                    (Some(expected_sig), Some(ty::ClosureKind::Fn))
                }
                hir::ClosureKind::Coroutine(_) | hir::ClosureKind::CoroutineClosure(_) => {
                    (None, None)
                }
            },
            _ => (None, None),
        }
    }

    fn deduce_closure_signature_from_predicates(
        &self,
        expected_ty: Ty<'tcx>,
        closure_kind: hir::ClosureKind,
        predicates: impl DoubleEndedIterator<Item = (ty::Predicate<'tcx>, Span)>,
    ) -> (Option<ExpectedSig<'tcx>>, Option<ty::ClosureKind>) {
        let mut expected_sig = None;
        let mut expected_kind = None;

        for (pred, span) in traits::elaborate(
            self.tcx,
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
                && let ty::PredicateKind::Clause(ty::ClauseKind::Projection(proj_predicate)) =
                    bound_predicate.skip_binder()
            {
                let inferred_sig = self.normalize(
                    span,
                    self.deduce_sig_from_projection(
                        Some(span),
                        closure_kind,
                        bound_predicate.rebind(proj_predicate),
                    ),
                );

                // Make sure that we didn't infer a signature that mentions itself.
                // This can happen when we elaborate certain supertrait bounds that
                // mention projections containing the `Self` type. See #105401.
                struct MentionsTy<'tcx> {
                    expected_ty: Ty<'tcx>,
                }
                impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for MentionsTy<'tcx> {
                    type Result = ControlFlow<()>;

                    fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
                        if t == self.expected_ty {
                            ControlFlow::Break(())
                        } else {
                            t.super_visit_with(self)
                        }
                    }
                }

                // Don't infer a closure signature from a goal that names the closure type as this will
                // (almost always) lead to occurs check errors later in type checking.
                if self.next_trait_solver()
                    && let Some(inferred_sig) = inferred_sig
                {
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
                    let generalized_fnptr_sig = self.next_ty_var(span);
                    let inferred_fnptr_sig = Ty::new_fn_ptr(self.tcx, inferred_sig.sig);
                    self.demand_eqtype(span, inferred_fnptr_sig, generalized_fnptr_sig);

                    let resolved_sig = self.resolve_vars_if_possible(generalized_fnptr_sig);

                    if resolved_sig.visit_with(&mut MentionsTy { expected_ty }).is_continue() {
                        expected_sig = Some(ExpectedSig {
                            cause_span: inferred_sig.cause_span,
                            sig: resolved_sig.fn_sig(self.tcx),
                        });
                    }
                } else {
                    if inferred_sig.visit_with(&mut MentionsTy { expected_ty }).is_continue() {
                        expected_sig = inferred_sig;
                    }
                }
            }

            // Even if we can't infer the full signature, we may be able to
            // infer the kind. This can occur when we elaborate a predicate
            // like `F : Fn<A>`. Note that due to subtyping we could encounter
            // many viable options, so pick the most restrictive.
            let trait_def_id = match bound_predicate.skip_binder() {
                ty::PredicateKind::Clause(ty::ClauseKind::Projection(data)) => {
                    Some(data.projection_term.trait_def_id(self.tcx))
                }
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(data)) => Some(data.def_id()),
                _ => None,
            };

            if let Some(trait_def_id) = trait_def_id {
                let found_kind = match closure_kind {
                    hir::ClosureKind::Closure => self.tcx.fn_trait_kind_from_def_id(trait_def_id),
                    hir::ClosureKind::CoroutineClosure(hir::CoroutineDesugaring::Async) => self
                        .tcx
                        .async_fn_trait_kind_from_def_id(trait_def_id)
                        .or_else(|| self.tcx.fn_trait_kind_from_def_id(trait_def_id)),
                    _ => None,
                };

                if let Some(found_kind) = found_kind {
                    // always use the closure kind that is more permissive.
                    match (expected_kind, found_kind) {
                        (None, _) => expected_kind = Some(found_kind),
                        (Some(ClosureKind::FnMut), ClosureKind::Fn) => {
                            expected_kind = Some(ClosureKind::Fn)
                        }
                        (Some(ClosureKind::FnOnce), ClosureKind::Fn | ClosureKind::FnMut) => {
                            expected_kind = Some(found_kind)
                        }
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
    #[instrument(level = "debug", skip(self, cause_span), ret)]
    fn deduce_sig_from_projection(
        &self,
        cause_span: Option<Span>,
        closure_kind: hir::ClosureKind,
        projection: ty::PolyProjectionPredicate<'tcx>,
    ) -> Option<ExpectedSig<'tcx>> {
        let def_id = projection.item_def_id();

        // For now, we only do signature deduction based off of the `Fn` and `AsyncFn` traits,
        // for closures and async closures, respectively.
        match closure_kind {
            hir::ClosureKind::Closure if self.tcx.is_lang_item(def_id, LangItem::FnOnceOutput) => {
                self.extract_sig_from_projection(cause_span, projection)
            }
            hir::ClosureKind::CoroutineClosure(hir::CoroutineDesugaring::Async)
                if self.tcx.is_lang_item(def_id, LangItem::AsyncFnOnceOutput) =>
            {
                self.extract_sig_from_projection(cause_span, projection)
            }
            // It's possible we've passed the closure to a (somewhat out-of-fashion)
            // `F: FnOnce() -> Fut, Fut: Future<Output = T>` style bound. Let's still
            // guide inference here, since it's beneficial for the user.
            hir::ClosureKind::CoroutineClosure(hir::CoroutineDesugaring::Async)
                if self.tcx.is_lang_item(def_id, LangItem::FnOnceOutput) =>
            {
                self.extract_sig_from_projection_and_future_bound(cause_span, projection)
            }
            _ => None,
        }
    }

    /// Given an `FnOnce::Output` or `AsyncFn::Output` projection, extract the args
    /// and return type to infer a [`ty::PolyFnSig`] for the closure.
    fn extract_sig_from_projection(
        &self,
        cause_span: Option<Span>,
        projection: ty::PolyProjectionPredicate<'tcx>,
    ) -> Option<ExpectedSig<'tcx>> {
        let projection = self.resolve_vars_if_possible(projection);

        let arg_param_ty = projection.skip_binder().projection_term.args.type_at(1);
        debug!(?arg_param_ty);

        let ty::Tuple(input_tys) = *arg_param_ty.kind() else {
            return None;
        };

        // Since this is a return parameter type it is safe to unwrap.
        let ret_param_ty = projection.skip_binder().term.expect_type();
        debug!(?ret_param_ty);

        let sig = projection.rebind(self.tcx.mk_fn_sig(
            input_tys,
            ret_param_ty,
            false,
            hir::Safety::Safe,
            ExternAbi::Rust,
        ));

        Some(ExpectedSig { cause_span, sig })
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
        &self,
        cause_span: Option<Span>,
        projection: ty::PolyProjectionPredicate<'tcx>,
    ) -> Option<ExpectedSig<'tcx>> {
        let projection = self.resolve_vars_if_possible(projection);

        let arg_param_ty = projection.skip_binder().projection_term.args.type_at(1);
        debug!(?arg_param_ty);

        let ty::Tuple(input_tys) = *arg_param_ty.kind() else {
            return None;
        };

        // If the return type is a type variable, look for bounds on it.
        // We could theoretically support other kinds of return types here,
        // but none of them would be useful, since async closures return
        // concrete anonymous future types, and their futures are not coerced
        // into any other type within the body of the async closure.
        let ty::Infer(ty::TyVar(return_vid)) = *projection.skip_binder().term.expect_type().kind()
        else {
            return None;
        };

        // FIXME: We may want to elaborate here, though I assume this will be exceedingly rare.
        let mut return_ty = None;
        for bound in self.obligations_for_self_ty(return_vid) {
            if let Some(ret_projection) = bound.predicate.as_projection_clause()
                && let Some(ret_projection) = ret_projection.no_bound_vars()
                && self.tcx.is_lang_item(ret_projection.def_id(), LangItem::FutureOutput)
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
        let return_ty =
            return_ty.unwrap_or_else(|| self.next_ty_var(cause_span.unwrap_or(DUMMY_SP)));

        let sig = projection.rebind(self.tcx.mk_fn_sig(
            input_tys,
            return_ty,
            false,
            hir::Safety::Safe,
            ExternAbi::Rust,
        ));

        Some(ExpectedSig { cause_span, sig })
    }

    fn sig_of_closure(
        &self,
        expr_def_id: LocalDefId,
        decl: &hir::FnDecl<'tcx>,
        closure_kind: hir::ClosureKind,
        expected_sig: Option<ExpectedSig<'tcx>>,
    ) -> ClosureSignatures<'tcx> {
        if let Some(e) = expected_sig {
            self.sig_of_closure_with_expectation(expr_def_id, decl, closure_kind, e)
        } else {
            self.sig_of_closure_no_expectation(expr_def_id, decl, closure_kind)
        }
    }

    /// If there is no expected signature, then we will convert the
    /// types that the user gave into a signature.
    #[instrument(skip(self, expr_def_id, decl), level = "debug")]
    fn sig_of_closure_no_expectation(
        &self,
        expr_def_id: LocalDefId,
        decl: &hir::FnDecl<'tcx>,
        closure_kind: hir::ClosureKind,
    ) -> ClosureSignatures<'tcx> {
        let bound_sig = self.supplied_sig_of_closure(expr_def_id, decl, closure_kind);

        self.closure_sigs(expr_def_id, bound_sig)
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
    #[instrument(skip(self, expr_def_id, decl), level = "debug")]
    fn sig_of_closure_with_expectation(
        &self,
        expr_def_id: LocalDefId,
        decl: &hir::FnDecl<'tcx>,
        closure_kind: hir::ClosureKind,
        expected_sig: ExpectedSig<'tcx>,
    ) -> ClosureSignatures<'tcx> {
        // Watch out for some surprises and just ignore the
        // expectation if things don't see to match up with what we
        // expect.
        if expected_sig.sig.c_variadic() != decl.c_variadic {
            return self.sig_of_closure_no_expectation(expr_def_id, decl, closure_kind);
        } else if expected_sig.sig.skip_binder().inputs_and_output.len() != decl.inputs.len() + 1 {
            return self.sig_of_closure_with_mismatched_number_of_arguments(
                expr_def_id,
                decl,
                expected_sig,
            );
        }

        // Create a `PolyFnSig`. Note the oddity that late bound
        // regions appearing free in `expected_sig` are now bound up
        // in this binder we are creating.
        assert!(!expected_sig.sig.skip_binder().has_vars_bound_above(ty::INNERMOST));
        let bound_sig = expected_sig.sig.map_bound(|sig| {
            self.tcx.mk_fn_sig(
                sig.inputs().iter().cloned(),
                sig.output(),
                sig.c_variadic,
                hir::Safety::Safe,
                ExternAbi::RustCall,
            )
        });

        // `deduce_expectations_from_expected_type` introduces
        // late-bound lifetimes defined elsewhere, which we now
        // anonymize away, so as not to confuse the user.
        let bound_sig = self.tcx.anonymize_bound_vars(bound_sig);

        let closure_sigs = self.closure_sigs(expr_def_id, bound_sig);

        // Up till this point, we have ignored the annotations that the user
        // gave. This function will check that they unify successfully.
        // Along the way, it also writes out entries for types that the user
        // wrote into our typeck results, which are then later used by the privacy
        // check.
        match self.merge_supplied_sig_with_expectation(
            expr_def_id,
            decl,
            closure_kind,
            closure_sigs,
        ) {
            Ok(infer_ok) => self.register_infer_ok_obligations(infer_ok),
            Err(_) => self.sig_of_closure_no_expectation(expr_def_id, decl, closure_kind),
        }
    }

    fn sig_of_closure_with_mismatched_number_of_arguments(
        &self,
        expr_def_id: LocalDefId,
        decl: &hir::FnDecl<'tcx>,
        expected_sig: ExpectedSig<'tcx>,
    ) -> ClosureSignatures<'tcx> {
        let expr_map_node = self.tcx.hir_node_by_def_id(expr_def_id);
        let expected_args: Vec<_> = expected_sig
            .sig
            .skip_binder()
            .inputs()
            .iter()
            .map(|ty| ArgKind::from_expected_ty(*ty, None))
            .collect();
        let (closure_span, closure_arg_span, found_args) =
            match self.err_ctxt().get_fn_like_arguments(expr_map_node) {
                Some((sp, arg_sp, args)) => (Some(sp), arg_sp, args),
                None => (None, None, Vec::new()),
            };
        let expected_span =
            expected_sig.cause_span.unwrap_or_else(|| self.tcx.def_span(expr_def_id));
        let guar = self
            .err_ctxt()
            .report_arg_count_mismatch(
                expected_span,
                closure_span,
                expected_args,
                found_args,
                true,
                closure_arg_span,
            )
            .emit();

        let error_sig = self.error_sig_of_closure(decl, guar);

        self.closure_sigs(expr_def_id, error_sig)
    }

    /// Enforce the user's types against the expectation. See
    /// `sig_of_closure_with_expectation` for details on the overall
    /// strategy.
    #[instrument(level = "debug", skip(self, expr_def_id, decl, expected_sigs))]
    fn merge_supplied_sig_with_expectation(
        &self,
        expr_def_id: LocalDefId,
        decl: &hir::FnDecl<'tcx>,
        closure_kind: hir::ClosureKind,
        mut expected_sigs: ClosureSignatures<'tcx>,
    ) -> InferResult<'tcx, ClosureSignatures<'tcx>> {
        // Get the signature S that the user gave.
        //
        // (See comment on `sig_of_closure_with_expectation` for the
        // meaning of these letters.)
        let supplied_sig = self.supplied_sig_of_closure(expr_def_id, decl, closure_kind);

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
        self.commit_if_ok(|_| {
            let mut all_obligations = PredicateObligations::new();
            let supplied_sig = self.instantiate_binder_with_fresh_vars(
                self.tcx.def_span(expr_def_id),
                BoundRegionConversionTime::FnCall,
                supplied_sig,
            );

            // The liberated version of this signature should be a subtype
            // of the liberated form of the expectation.
            for ((hir_ty, &supplied_ty), expected_ty) in iter::zip(
                iter::zip(decl.inputs, supplied_sig.inputs()),
                expected_sigs.liberated_sig.inputs(), // `liberated_sig` is E'.
            ) {
                // Check that E' = S'.
                let cause = self.misc(hir_ty.span);
                let InferOk { value: (), obligations } = self.at(&cause, self.param_env).eq(
                    DefineOpaqueTypes::Yes,
                    *expected_ty,
                    supplied_ty,
                )?;
                all_obligations.extend(obligations);
            }

            let supplied_output_ty = supplied_sig.output();
            let cause = &self.misc(decl.output.span());
            let InferOk { value: (), obligations } = self.at(cause, self.param_env).eq(
                DefineOpaqueTypes::Yes,
                expected_sigs.liberated_sig.output(),
                supplied_output_ty,
            )?;
            all_obligations.extend(obligations);

            let inputs =
                supplied_sig.inputs().into_iter().map(|&ty| self.resolve_vars_if_possible(ty));

            expected_sigs.liberated_sig = self.tcx.mk_fn_sig(
                inputs,
                supplied_output_ty,
                expected_sigs.liberated_sig.c_variadic,
                hir::Safety::Safe,
                ExternAbi::RustCall,
            );

            Ok(InferOk { value: expected_sigs, obligations: all_obligations })
        })
    }

    /// If there is no expected signature, then we will convert the
    /// types that the user gave into a signature.
    ///
    /// Also, record this closure signature for later.
    #[instrument(skip(self, decl), level = "debug", ret)]
    fn supplied_sig_of_closure(
        &self,
        expr_def_id: LocalDefId,
        decl: &hir::FnDecl<'tcx>,
        closure_kind: hir::ClosureKind,
    ) -> ty::PolyFnSig<'tcx> {
        let lowerer = self.lowerer();

        trace!("decl = {:#?}", decl);
        debug!(?closure_kind);

        let hir_id = self.tcx.local_def_id_to_hir_id(expr_def_id);
        let bound_vars = self.tcx.late_bound_vars(hir_id);

        // First, convert the types that the user supplied (if any).
        let supplied_arguments = decl.inputs.iter().map(|a| lowerer.lower_ty(a));
        let supplied_return = match decl.output {
            hir::FnRetTy::Return(ref output) => lowerer.lower_ty(output),
            hir::FnRetTy::DefaultReturn(_) => match closure_kind {
                // In the case of the async block that we create for a function body,
                // we expect the return type of the block to match that of the enclosing
                // function.
                hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                    hir::CoroutineDesugaring::Async,
                    hir::CoroutineSource::Fn,
                )) => {
                    debug!("closure is async fn body");
                    self.deduce_future_output_from_obligations(expr_def_id).unwrap_or_else(|| {
                        // AFAIK, deducing the future output
                        // always succeeds *except* in error cases
                        // like #65159. I'd like to return Error
                        // here, but I can't because I can't
                        // easily (and locally) prove that we
                        // *have* reported an
                        // error. --nikomatsakis
                        lowerer.ty_infer(None, decl.output.span())
                    })
                }
                // All `gen {}` and `async gen {}` must return unit.
                hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                    hir::CoroutineDesugaring::Gen | hir::CoroutineDesugaring::AsyncGen,
                    _,
                )) => self.tcx.types.unit,

                // For async blocks, we just fall back to `_` here.
                // For closures/coroutines, we know nothing about the return
                // type unless it was supplied.
                hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                    hir::CoroutineDesugaring::Async,
                    _,
                ))
                | hir::ClosureKind::Coroutine(hir::CoroutineKind::Coroutine(_))
                | hir::ClosureKind::Closure
                | hir::ClosureKind::CoroutineClosure(_) => {
                    lowerer.ty_infer(None, decl.output.span())
                }
            },
        };

        let result = ty::Binder::bind_with_vars(
            self.tcx.mk_fn_sig(
                supplied_arguments,
                supplied_return,
                decl.c_variadic,
                hir::Safety::Safe,
                ExternAbi::RustCall,
            ),
            bound_vars,
        );

        let c_result = self.infcx.canonicalize_response(result);
        self.typeck_results.borrow_mut().user_provided_sigs.insert(expr_def_id, c_result);

        // Normalize only after registering in `user_provided_sigs`.
        self.normalize(self.tcx.def_span(expr_def_id), result)
    }

    /// Invoked when we are translating the coroutine that results
    /// from desugaring an `async fn`. Returns the "sugared" return
    /// type of the `async fn` -- that is, the return type that the
    /// user specified. The "desugared" return type is an `impl
    /// Future<Output = T>`, so we do this by searching through the
    /// obligations to extract the `T`.
    #[instrument(skip(self), level = "debug", ret)]
    fn deduce_future_output_from_obligations(&self, body_def_id: LocalDefId) -> Option<Ty<'tcx>> {
        let ret_coercion = self.ret_coercion.as_ref().unwrap_or_else(|| {
            span_bug!(self.tcx.def_span(body_def_id), "async fn coroutine outside of a fn")
        });

        let closure_span = self.tcx.def_span(body_def_id);
        let ret_ty = ret_coercion.borrow().expected_ty();
        let ret_ty = self.try_structurally_resolve_type(closure_span, ret_ty);

        let get_future_output = |predicate: ty::Predicate<'tcx>, span| {
            // Search for a pending obligation like
            //
            // `<R as Future>::Output = T`
            //
            // where R is the return type we are expecting. This type `T`
            // will be our output.
            let bound_predicate = predicate.kind();
            if let ty::PredicateKind::Clause(ty::ClauseKind::Projection(proj_predicate)) =
                bound_predicate.skip_binder()
            {
                self.deduce_future_output_from_projection(
                    span,
                    bound_predicate.rebind(proj_predicate),
                )
            } else {
                None
            }
        };

        let output_ty = match *ret_ty.kind() {
            ty::Infer(ty::TyVar(ret_vid)) => {
                self.obligations_for_self_ty(ret_vid).into_iter().find_map(|obligation| {
                    get_future_output(obligation.predicate, obligation.cause.span)
                })?
            }
            ty::Alias(ty::Projection, _) => {
                return Some(Ty::new_error_with_message(
                    self.tcx,
                    closure_span,
                    "this projection should have been projected to an opaque type",
                ));
            }
            ty::Alias(ty::Opaque, ty::AliasTy { def_id, args, .. }) => self
                .tcx
                .explicit_item_self_bounds(def_id)
                .iter_instantiated_copied(self.tcx, args)
                .find_map(|(p, s)| get_future_output(p.as_predicate(), s))?,
            ty::Error(_) => return Some(ret_ty),
            _ => {
                span_bug!(closure_span, "invalid async fn coroutine return type: {ret_ty:?}")
            }
        };

        let output_ty = self.normalize(closure_span, output_ty);

        // async fn that have opaque types in their return type need to redo the conversion to inference variables
        // as they fetch the still opaque version from the signature.
        let InferOk { value: output_ty, obligations } = self
            .replace_opaque_types_with_inference_vars(
                output_ty,
                body_def_id,
                closure_span,
                self.param_env,
            );
        self.register_predicates(obligations);

        Some(output_ty)
    }

    /// Given a projection like
    ///
    /// `<X as Future>::Output = T`
    ///
    /// where `X` is some type that has no late-bound regions, returns
    /// `Some(T)`. If the projection is for some other trait, returns
    /// `None`.
    fn deduce_future_output_from_projection(
        &self,
        cause_span: Span,
        predicate: ty::PolyProjectionPredicate<'tcx>,
    ) -> Option<Ty<'tcx>> {
        debug!("deduce_future_output_from_projection(predicate={:?})", predicate);

        // We do not expect any bound regions in our predicate, so
        // skip past the bound vars.
        let Some(predicate) = predicate.no_bound_vars() else {
            debug!("deduce_future_output_from_projection: has late-bound regions");
            return None;
        };

        // Check that this is a projection from the `Future` trait.
        let trait_def_id = predicate.projection_term.trait_def_id(self.tcx);
        let future_trait = self.tcx.require_lang_item(LangItem::Future, Some(cause_span));
        if trait_def_id != future_trait {
            debug!("deduce_future_output_from_projection: not a future");
            return None;
        }

        // The `Future` trait has only one associated item, `Output`,
        // so check that this is what we see.
        let output_assoc_item = self.tcx.associated_item_def_ids(future_trait)[0];
        if output_assoc_item != predicate.projection_term.def_id {
            span_bug!(
                cause_span,
                "projecting associated item `{:?}` from future, which is not Output `{:?}`",
                predicate.projection_term.def_id,
                output_assoc_item,
            );
        }

        // Extract the type from the projection. Note that there can
        // be no bound variables in this type because the "self type"
        // does not have any regions in it.
        let output_ty = self.resolve_vars_if_possible(predicate.term);
        debug!("deduce_future_output_from_projection: output_ty={:?}", output_ty);
        // This is a projection on a Fn trait so will always be a type.
        Some(output_ty.expect_type())
    }

    /// Converts the types that the user supplied, in case that doing
    /// so should yield an error, but returns back a signature where
    /// all parameters are of type `ty::Error`.
    fn error_sig_of_closure(
        &self,
        decl: &hir::FnDecl<'tcx>,
        guar: ErrorGuaranteed,
    ) -> ty::PolyFnSig<'tcx> {
        let lowerer = self.lowerer();
        let err_ty = Ty::new_error(self.tcx, guar);

        let supplied_arguments = decl.inputs.iter().map(|a| {
            // Convert the types that the user supplied (if any), but ignore them.
            lowerer.lower_ty(a);
            err_ty
        });

        if let hir::FnRetTy::Return(ref output) = decl.output {
            lowerer.lower_ty(output);
        }

        let result = ty::Binder::dummy(self.tcx.mk_fn_sig(
            supplied_arguments,
            err_ty,
            decl.c_variadic,
            hir::Safety::Safe,
            ExternAbi::RustCall,
        ));

        debug!("supplied_sig_of_closure: result={:?}", result);

        result
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn closure_sigs(
        &self,
        expr_def_id: LocalDefId,
        bound_sig: ty::PolyFnSig<'tcx>,
    ) -> ClosureSignatures<'tcx> {
        let liberated_sig =
            self.tcx().liberate_late_bound_regions(expr_def_id.to_def_id(), bound_sig);
        let liberated_sig = self.normalize(self.tcx.def_span(expr_def_id), liberated_sig);
        ClosureSignatures { bound_sig, liberated_sig }
    }
}
