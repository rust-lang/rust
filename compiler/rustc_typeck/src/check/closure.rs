//! Code for type-checking closure expressions.

use super::{check_fn, Expectation, FnCtxt, GeneratorTypes};

use crate::astconv::AstConv;
use crate::rustc_middle::ty::subst::Subst;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::LateBoundRegionConversionTime;
use rustc_infer::infer::{InferOk, InferResult};
use rustc_middle::ty::subst::InternalSubsts;
use rustc_middle::ty::visit::TypeVisitable;
use rustc_middle::ty::{self, Ty};
use rustc_span::source_map::Span;
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::traits::error_reporting::ArgKind;
use rustc_trait_selection::traits::error_reporting::InferCtxtExt as _;
use std::cmp;
use std::iter;

/// What signature do we *expect* the closure to have from context?
#[derive(Debug)]
struct ExpectedSig<'tcx> {
    /// Span that gave us this expectation, if we know that.
    cause_span: Option<Span>,
    sig: ty::PolyFnSig<'tcx>,
}

struct ClosureSignatures<'tcx> {
    bound_sig: ty::PolyFnSig<'tcx>,
    liberated_sig: ty::FnSig<'tcx>,
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    #[instrument(skip(self, expr, _capture, decl, body_id), level = "debug")]
    pub fn check_expr_closure(
        &self,
        expr: &hir::Expr<'_>,
        _capture: hir::CaptureBy,
        decl: &'tcx hir::FnDecl<'tcx>,
        body_id: hir::BodyId,
        gen: Option<hir::Movability>,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        trace!("decl = {:#?}", decl);
        trace!("expr = {:#?}", expr);

        // It's always helpful for inference if we know the kind of
        // closure sooner rather than later, so first examine the expected
        // type, and see if can glean a closure kind from there.
        let (expected_sig, expected_kind) = match expected.to_option(self) {
            Some(ty) => self.deduce_expectations_from_expected_type(ty),
            None => (None, None),
        };
        let body = self.tcx.hir().body(body_id);
        self.check_closure(expr, expected_kind, decl, body, gen, expected_sig)
    }

    #[instrument(skip(self, expr, body, decl), level = "debug", ret)]
    fn check_closure(
        &self,
        expr: &hir::Expr<'_>,
        opt_kind: Option<ty::ClosureKind>,
        decl: &'tcx hir::FnDecl<'tcx>,
        body: &'tcx hir::Body<'tcx>,
        gen: Option<hir::Movability>,
        expected_sig: Option<ExpectedSig<'tcx>>,
    ) -> Ty<'tcx> {
        trace!("decl = {:#?}", decl);
        let expr_def_id = self.tcx.hir().local_def_id(expr.hir_id);
        debug!(?expr_def_id);

        let ClosureSignatures { bound_sig, liberated_sig } =
            self.sig_of_closure(expr.hir_id, expr_def_id.to_def_id(), decl, body, expected_sig);

        debug!(?bound_sig, ?liberated_sig);

        let return_type_pre_known = !liberated_sig.output().is_ty_infer();

        let generator_types = check_fn(
            self,
            self.param_env.without_const(),
            liberated_sig,
            decl,
            expr.hir_id,
            body,
            gen,
            return_type_pre_known,
        )
        .1;

        let parent_substs = InternalSubsts::identity_for_item(
            self.tcx,
            self.tcx.typeck_root_def_id(expr_def_id.to_def_id()),
        );

        let tupled_upvars_ty = self.next_ty_var(TypeVariableOrigin {
            kind: TypeVariableOriginKind::ClosureSynthetic,
            span: self.tcx.hir().span(expr.hir_id),
        });

        if let Some(GeneratorTypes { resume_ty, yield_ty, interior, movability }) = generator_types
        {
            let generator_substs = ty::GeneratorSubsts::new(
                self.tcx,
                ty::GeneratorSubstsParts {
                    parent_substs,
                    resume_ty,
                    yield_ty,
                    return_ty: liberated_sig.output(),
                    witness: interior,
                    tupled_upvars_ty,
                },
            );

            return self.tcx.mk_generator(
                expr_def_id.to_def_id(),
                generator_substs.substs,
                movability,
            );
        }

        // Tuple up the arguments and insert the resulting function type into
        // the `closures` table.
        let sig = bound_sig.map_bound(|sig| {
            self.tcx.mk_fn_sig(
                iter::once(self.tcx.intern_tup(sig.inputs())),
                sig.output(),
                sig.c_variadic,
                sig.unsafety,
                sig.abi,
            )
        });

        debug!(?sig, ?opt_kind);

        let closure_kind_ty = match opt_kind {
            Some(kind) => kind.to_ty(self.tcx),

            // Create a type variable (for now) to represent the closure kind.
            // It will be unified during the upvar inference phase (`upvar.rs`)
            None => self.next_ty_var(TypeVariableOrigin {
                // FIXME(eddyb) distinguish closure kind inference variables from the rest.
                kind: TypeVariableOriginKind::ClosureSynthetic,
                span: expr.span,
            }),
        };

        let closure_substs = ty::ClosureSubsts::new(
            self.tcx,
            ty::ClosureSubstsParts {
                parent_substs,
                closure_kind_ty,
                closure_sig_as_fn_ptr_ty: self.tcx.mk_fn_ptr(sig),
                tupled_upvars_ty,
            },
        );

        self.tcx.mk_closure(expr_def_id.to_def_id(), closure_substs.substs)
    }

    /// Given the expected type, figures out what it can about this closure we
    /// are about to type check:
    #[instrument(skip(self), level = "debug")]
    fn deduce_expectations_from_expected_type(
        &self,
        expected_ty: Ty<'tcx>,
    ) -> (Option<ExpectedSig<'tcx>>, Option<ty::ClosureKind>) {
        match *expected_ty.kind() {
            ty::Opaque(def_id, substs) => {
                let bounds = self.tcx.bound_explicit_item_bounds(def_id);
                let sig = bounds
                    .transpose_iter()
                    .map(|e| e.map_bound(|e| *e).transpose_tuple2())
                    .find_map(|(pred, span)| match pred.0.kind().skip_binder() {
                        ty::PredicateKind::Projection(proj_predicate) => self
                            .deduce_sig_from_projection(
                                Some(span.0),
                                pred.0
                                    .kind()
                                    .rebind(pred.rebind(proj_predicate).subst(self.tcx, substs)),
                            ),
                        _ => None,
                    });

                let kind = bounds
                    .transpose_iter()
                    .map(|e| e.map_bound(|e| *e).transpose_tuple2())
                    .filter_map(|(pred, _)| match pred.0.kind().skip_binder() {
                        ty::PredicateKind::Trait(tp) => {
                            self.tcx.fn_trait_kind_from_lang_item(tp.def_id())
                        }
                        _ => None,
                    })
                    .fold(None, |best, cur| Some(best.map_or(cur, |best| cmp::min(best, cur))));
                trace!(?sig, ?kind);
                (sig, kind)
            }
            ty::Dynamic(ref object_type, ..) => {
                let sig = object_type.projection_bounds().find_map(|pb| {
                    let pb = pb.with_self_ty(self.tcx, self.tcx.types.trait_object_dummy_self);
                    self.deduce_sig_from_projection(None, pb)
                });
                let kind = object_type
                    .principal_def_id()
                    .and_then(|did| self.tcx.fn_trait_kind_from_lang_item(did));
                (sig, kind)
            }
            ty::Infer(ty::TyVar(vid)) => self.deduce_expectations_from_obligations(vid),
            ty::FnPtr(sig) => {
                let expected_sig = ExpectedSig { cause_span: None, sig };
                (Some(expected_sig), Some(ty::ClosureKind::Fn))
            }
            _ => (None, None),
        }
    }

    fn deduce_expectations_from_obligations(
        &self,
        expected_vid: ty::TyVid,
    ) -> (Option<ExpectedSig<'tcx>>, Option<ty::ClosureKind>) {
        let expected_sig =
            self.obligations_for_self_ty(expected_vid).find_map(|(_, obligation)| {
                debug!(?obligation.predicate);

                let bound_predicate = obligation.predicate.kind();
                if let ty::PredicateKind::Projection(proj_predicate) =
                    obligation.predicate.kind().skip_binder()
                {
                    // Given a Projection predicate, we can potentially infer
                    // the complete signature.
                    self.deduce_sig_from_projection(
                        Some(obligation.cause.span),
                        bound_predicate.rebind(proj_predicate),
                    )
                } else {
                    None
                }
            });

        // Even if we can't infer the full signature, we may be able to
        // infer the kind. This can occur when we elaborate a predicate
        // like `F : Fn<A>`. Note that due to subtyping we could encounter
        // many viable options, so pick the most restrictive.
        let expected_kind = self
            .obligations_for_self_ty(expected_vid)
            .filter_map(|(tr, _)| self.tcx.fn_trait_kind_from_lang_item(tr.def_id()))
            .fold(None, |best, cur| Some(best.map_or(cur, |best| cmp::min(best, cur))));

        (expected_sig, expected_kind)
    }

    /// Given a projection like "<F as Fn(X)>::Result == Y", we can deduce
    /// everything we need to know about a closure or generator.
    ///
    /// The `cause_span` should be the span that caused us to
    /// have this expected signature, or `None` if we can't readily
    /// know that.
    #[instrument(level = "debug", skip(self, cause_span), ret)]
    fn deduce_sig_from_projection(
        &self,
        cause_span: Option<Span>,
        projection: ty::PolyProjectionPredicate<'tcx>,
    ) -> Option<ExpectedSig<'tcx>> {
        let tcx = self.tcx;

        let trait_def_id = projection.trait_def_id(tcx);

        let is_fn = tcx.fn_trait_kind_from_lang_item(trait_def_id).is_some();
        let gen_trait = tcx.require_lang_item(LangItem::Generator, cause_span);
        let is_gen = gen_trait == trait_def_id;
        if !is_fn && !is_gen {
            debug!("not fn or generator");
            return None;
        }

        if is_gen {
            // Check that we deduce the signature from the `<_ as std::ops::Generator>::Return`
            // associated item and not yield.
            let return_assoc_item = self.tcx.associated_item_def_ids(gen_trait)[1];
            if return_assoc_item != projection.projection_def_id() {
                debug!("not return assoc item of generator");
                return None;
            }
        }

        let input_tys = if is_fn {
            let arg_param_ty = projection.skip_binder().projection_ty.substs.type_at(1);
            let arg_param_ty = self.resolve_vars_if_possible(arg_param_ty);
            debug!(?arg_param_ty);

            match arg_param_ty.kind() {
                &ty::Tuple(tys) => tys,
                _ => return None,
            }
        } else {
            // Generators with a `()` resume type may be defined with 0 or 1 explicit arguments,
            // else they must have exactly 1 argument. For now though, just give up in this case.
            return None;
        };

        // Since this is a return parameter type it is safe to unwrap.
        let ret_param_ty = projection.skip_binder().term.ty().unwrap();
        let ret_param_ty = self.resolve_vars_if_possible(ret_param_ty);
        debug!(?ret_param_ty);

        let sig = projection.rebind(self.tcx.mk_fn_sig(
            input_tys.iter(),
            ret_param_ty,
            false,
            hir::Unsafety::Normal,
            Abi::Rust,
        ));

        Some(ExpectedSig { cause_span, sig })
    }

    fn sig_of_closure(
        &self,
        hir_id: hir::HirId,
        expr_def_id: DefId,
        decl: &hir::FnDecl<'_>,
        body: &hir::Body<'_>,
        expected_sig: Option<ExpectedSig<'tcx>>,
    ) -> ClosureSignatures<'tcx> {
        if let Some(e) = expected_sig {
            self.sig_of_closure_with_expectation(hir_id, expr_def_id, decl, body, e)
        } else {
            self.sig_of_closure_no_expectation(hir_id, expr_def_id, decl, body)
        }
    }

    /// If there is no expected signature, then we will convert the
    /// types that the user gave into a signature.
    #[instrument(skip(self, hir_id, expr_def_id, decl, body), level = "debug")]
    fn sig_of_closure_no_expectation(
        &self,
        hir_id: hir::HirId,
        expr_def_id: DefId,
        decl: &hir::FnDecl<'_>,
        body: &hir::Body<'_>,
    ) -> ClosureSignatures<'tcx> {
        let bound_sig = self.supplied_sig_of_closure(hir_id, expr_def_id, decl, body);

        self.closure_sigs(expr_def_id, body, bound_sig)
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
    /// - S would be `fn(&u32) ->
    /// - E' is `&'!0 u32 -> &'!0 u32`
    /// - S' is `&'?0 u32 -> ?T`
    ///
    /// S' can be unified with E' with `['?0 = '!0, ?T = &'!10 u32]`.
    ///
    /// # Arguments
    ///
    /// - `expr_def_id`: the `DefId` of the closure expression
    /// - `decl`: the HIR declaration of the closure
    /// - `body`: the body of the closure
    /// - `expected_sig`: the expected signature (if any). Note that
    ///   this is missing a binder: that is, there may be late-bound
    ///   regions with depth 1, which are bound then by the closure.
    #[instrument(skip(self, hir_id, expr_def_id, decl, body), level = "debug")]
    fn sig_of_closure_with_expectation(
        &self,
        hir_id: hir::HirId,
        expr_def_id: DefId,
        decl: &hir::FnDecl<'_>,
        body: &hir::Body<'_>,
        expected_sig: ExpectedSig<'tcx>,
    ) -> ClosureSignatures<'tcx> {
        // Watch out for some surprises and just ignore the
        // expectation if things don't see to match up with what we
        // expect.
        if expected_sig.sig.c_variadic() != decl.c_variadic {
            return self.sig_of_closure_no_expectation(hir_id, expr_def_id, decl, body);
        } else if expected_sig.sig.skip_binder().inputs_and_output.len() != decl.inputs.len() + 1 {
            return self.sig_of_closure_with_mismatched_number_of_arguments(
                expr_def_id,
                decl,
                body,
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
                hir::Unsafety::Normal,
                Abi::RustCall,
            )
        });

        // `deduce_expectations_from_expected_type` introduces
        // late-bound lifetimes defined elsewhere, which we now
        // anonymize away, so as not to confuse the user.
        let bound_sig = self.tcx.anonymize_late_bound_regions(bound_sig);

        let closure_sigs = self.closure_sigs(expr_def_id, body, bound_sig);

        // Up till this point, we have ignored the annotations that the user
        // gave. This function will check that they unify successfully.
        // Along the way, it also writes out entries for types that the user
        // wrote into our typeck results, which are then later used by the privacy
        // check.
        match self.check_supplied_sig_against_expectation(
            hir_id,
            expr_def_id,
            decl,
            body,
            &closure_sigs,
        ) {
            Ok(infer_ok) => self.register_infer_ok_obligations(infer_ok),
            Err(_) => return self.sig_of_closure_no_expectation(hir_id, expr_def_id, decl, body),
        }

        closure_sigs
    }

    fn sig_of_closure_with_mismatched_number_of_arguments(
        &self,
        expr_def_id: DefId,
        decl: &hir::FnDecl<'_>,
        body: &hir::Body<'_>,
        expected_sig: ExpectedSig<'tcx>,
    ) -> ClosureSignatures<'tcx> {
        let hir = self.tcx.hir();
        let expr_map_node = hir.get_if_local(expr_def_id).unwrap();
        let expected_args: Vec<_> = expected_sig
            .sig
            .skip_binder()
            .inputs()
            .iter()
            .map(|ty| ArgKind::from_expected_ty(*ty, None))
            .collect();
        let (closure_span, found_args) = match self.get_fn_like_arguments(expr_map_node) {
            Some((sp, args)) => (Some(sp), args),
            None => (None, Vec::new()),
        };
        let expected_span =
            expected_sig.cause_span.unwrap_or_else(|| hir.span_if_local(expr_def_id).unwrap());
        self.report_arg_count_mismatch(
            expected_span,
            closure_span,
            expected_args,
            found_args,
            true,
        )
        .emit();

        let error_sig = self.error_sig_of_closure(decl);

        self.closure_sigs(expr_def_id, body, error_sig)
    }

    /// Enforce the user's types against the expectation. See
    /// `sig_of_closure_with_expectation` for details on the overall
    /// strategy.
    fn check_supplied_sig_against_expectation(
        &self,
        hir_id: hir::HirId,
        expr_def_id: DefId,
        decl: &hir::FnDecl<'_>,
        body: &hir::Body<'_>,
        expected_sigs: &ClosureSignatures<'tcx>,
    ) -> InferResult<'tcx, ()> {
        // Get the signature S that the user gave.
        //
        // (See comment on `sig_of_closure_with_expectation` for the
        // meaning of these letters.)
        let supplied_sig = self.supplied_sig_of_closure(hir_id, expr_def_id, decl, body);

        debug!("check_supplied_sig_against_expectation: supplied_sig={:?}", supplied_sig);

        // FIXME(#45727): As discussed in [this comment][c1], naively
        // forcing equality here actually results in suboptimal error
        // messages in some cases.  For now, if there would have been
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
            let mut all_obligations = vec![];

            // The liberated version of this signature should be a subtype
            // of the liberated form of the expectation.
            for ((hir_ty, &supplied_ty), expected_ty) in iter::zip(
                iter::zip(
                    decl.inputs,
                    supplied_sig.inputs().skip_binder(), // binder moved to (*) below
                ),
                expected_sigs.liberated_sig.inputs(), // `liberated_sig` is E'.
            ) {
                // Instantiate (this part of..) S to S', i.e., with fresh variables.
                let supplied_ty = self.replace_bound_vars_with_fresh_vars(
                    hir_ty.span,
                    LateBoundRegionConversionTime::FnCall,
                    supplied_sig.inputs().rebind(supplied_ty),
                ); // recreated from (*) above

                // Check that E' = S'.
                let cause = self.misc(hir_ty.span);
                let InferOk { value: (), obligations } =
                    self.at(&cause, self.param_env).eq(*expected_ty, supplied_ty)?;
                all_obligations.extend(obligations);
            }

            let supplied_output_ty = self.replace_bound_vars_with_fresh_vars(
                decl.output.span(),
                LateBoundRegionConversionTime::FnCall,
                supplied_sig.output(),
            );
            let cause = &self.misc(decl.output.span());
            let InferOk { value: (), obligations } = self
                .at(cause, self.param_env)
                .eq(expected_sigs.liberated_sig.output(), supplied_output_ty)?;
            all_obligations.extend(obligations);

            Ok(InferOk { value: (), obligations: all_obligations })
        })
    }

    /// If there is no expected signature, then we will convert the
    /// types that the user gave into a signature.
    ///
    /// Also, record this closure signature for later.
    #[instrument(skip(self, decl, body), level = "debug", ret)]
    fn supplied_sig_of_closure(
        &self,
        hir_id: hir::HirId,
        expr_def_id: DefId,
        decl: &hir::FnDecl<'_>,
        body: &hir::Body<'_>,
    ) -> ty::PolyFnSig<'tcx> {
        let astconv: &dyn AstConv<'_> = self;

        trace!("decl = {:#?}", decl);
        debug!(?body.generator_kind);

        let bound_vars = self.tcx.late_bound_vars(hir_id);

        // First, convert the types that the user supplied (if any).
        let supplied_arguments = decl.inputs.iter().map(|a| astconv.ast_ty_to_ty(a));
        let supplied_return = match decl.output {
            hir::FnRetTy::Return(ref output) => astconv.ast_ty_to_ty(&output),
            hir::FnRetTy::DefaultReturn(_) => match body.generator_kind {
                // In the case of the async block that we create for a function body,
                // we expect the return type of the block to match that of the enclosing
                // function.
                Some(hir::GeneratorKind::Async(hir::AsyncGeneratorKind::Fn)) => {
                    debug!("closure is async fn body");
                    self.deduce_future_output_from_obligations(expr_def_id, body.id().hir_id)
                        .unwrap_or_else(|| {
                            // AFAIK, deducing the future output
                            // always succeeds *except* in error cases
                            // like #65159. I'd like to return Error
                            // here, but I can't because I can't
                            // easily (and locally) prove that we
                            // *have* reported an
                            // error. --nikomatsakis
                            astconv.ty_infer(None, decl.output.span())
                        })
                }

                _ => astconv.ty_infer(None, decl.output.span()),
            },
        };

        let result = ty::Binder::bind_with_vars(
            self.tcx.mk_fn_sig(
                supplied_arguments,
                supplied_return,
                decl.c_variadic,
                hir::Unsafety::Normal,
                Abi::RustCall,
            ),
            bound_vars,
        );

        let c_result = self.inh.infcx.canonicalize_response(result);
        self.typeck_results.borrow_mut().user_provided_sigs.insert(expr_def_id, c_result);

        result
    }

    /// Invoked when we are translating the generator that results
    /// from desugaring an `async fn`. Returns the "sugared" return
    /// type of the `async fn` -- that is, the return type that the
    /// user specified. The "desugared" return type is an `impl
    /// Future<Output = T>`, so we do this by searching through the
    /// obligations to extract the `T`.
    #[instrument(skip(self), level = "debug", ret)]
    fn deduce_future_output_from_obligations(
        &self,
        expr_def_id: DefId,
        body_id: hir::HirId,
    ) -> Option<Ty<'tcx>> {
        let ret_coercion = self.ret_coercion.as_ref().unwrap_or_else(|| {
            span_bug!(self.tcx.def_span(expr_def_id), "async fn generator outside of a fn")
        });

        let ret_ty = ret_coercion.borrow().expected_ty();
        let ret_ty = self.inh.infcx.shallow_resolve(ret_ty);

        let get_future_output = |predicate: ty::Predicate<'tcx>, span| {
            // Search for a pending obligation like
            //
            // `<R as Future>::Output = T`
            //
            // where R is the return type we are expecting. This type `T`
            // will be our output.
            let bound_predicate = predicate.kind();
            if let ty::PredicateKind::Projection(proj_predicate) = bound_predicate.skip_binder() {
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
                self.obligations_for_self_ty(ret_vid).find_map(|(_, obligation)| {
                    get_future_output(obligation.predicate, obligation.cause.span)
                })?
            }
            ty::Opaque(def_id, substs) => self
                .tcx
                .bound_explicit_item_bounds(def_id)
                .transpose_iter()
                .map(|e| e.map_bound(|e| *e).transpose_tuple2())
                .find_map(|(p, s)| get_future_output(p.subst(self.tcx, substs), s.0))?,
            ty::Error(_) => return None,
            _ => span_bug!(
                self.tcx.def_span(expr_def_id),
                "async fn generator return type not an inference variable"
            ),
        };

        // async fn that have opaque types in their return type need to redo the conversion to inference variables
        // as they fetch the still opaque version from the signature.
        let InferOk { value: output_ty, obligations } = self
            .replace_opaque_types_with_inference_vars(
                output_ty,
                body_id,
                self.tcx.def_span(expr_def_id),
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
        let trait_def_id = predicate.projection_ty.trait_def_id(self.tcx);
        let future_trait = self.tcx.require_lang_item(LangItem::Future, Some(cause_span));
        if trait_def_id != future_trait {
            debug!("deduce_future_output_from_projection: not a future");
            return None;
        }

        // The `Future` trait has only one associated item, `Output`,
        // so check that this is what we see.
        let output_assoc_item = self.tcx.associated_item_def_ids(future_trait)[0];
        if output_assoc_item != predicate.projection_ty.item_def_id {
            span_bug!(
                cause_span,
                "projecting associated item `{:?}` from future, which is not Output `{:?}`",
                predicate.projection_ty.item_def_id,
                output_assoc_item,
            );
        }

        // Extract the type from the projection. Note that there can
        // be no bound variables in this type because the "self type"
        // does not have any regions in it.
        let output_ty = self.resolve_vars_if_possible(predicate.term);
        debug!("deduce_future_output_from_projection: output_ty={:?}", output_ty);
        // This is a projection on a Fn trait so will always be a type.
        Some(output_ty.ty().unwrap())
    }

    /// Converts the types that the user supplied, in case that doing
    /// so should yield an error, but returns back a signature where
    /// all parameters are of type `TyErr`.
    fn error_sig_of_closure(&self, decl: &hir::FnDecl<'_>) -> ty::PolyFnSig<'tcx> {
        let astconv: &dyn AstConv<'_> = self;

        let supplied_arguments = decl.inputs.iter().map(|a| {
            // Convert the types that the user supplied (if any), but ignore them.
            astconv.ast_ty_to_ty(a);
            self.tcx.ty_error()
        });

        if let hir::FnRetTy::Return(ref output) = decl.output {
            astconv.ast_ty_to_ty(&output);
        }

        let result = ty::Binder::dummy(self.tcx.mk_fn_sig(
            supplied_arguments,
            self.tcx.ty_error(),
            decl.c_variadic,
            hir::Unsafety::Normal,
            Abi::RustCall,
        ));

        debug!("supplied_sig_of_closure: result={:?}", result);

        result
    }

    fn closure_sigs(
        &self,
        expr_def_id: DefId,
        body: &hir::Body<'_>,
        bound_sig: ty::PolyFnSig<'tcx>,
    ) -> ClosureSignatures<'tcx> {
        let liberated_sig = self.tcx().liberate_late_bound_regions(expr_def_id, bound_sig);
        let liberated_sig = self.inh.normalize_associated_types_in(
            body.value.span,
            body.value.hir_id,
            self.param_env,
            liberated_sig,
        );
        ClosureSignatures { bound_sig, liberated_sig }
    }
}
