use super::method::MethodCallee;
use super::{Expectation, FnCtxt, TupleArgumentsFlag};
use crate::type_error_struct;

use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def::{Namespace, Res};
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_infer::{
    infer,
    traits::{self, Obligation},
};
use rustc_infer::{
    infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind},
    traits::ObligationCause,
};
use rustc_middle::ty::adjustment::{
    Adjust, Adjustment, AllowTwoPhase, AutoBorrow, AutoBorrowMutability,
};
use rustc_middle::ty::subst::{Subst, SubstsRef};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;
use rustc_target::spec::abi;
use rustc_trait_selection::autoderef::Autoderef;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use std::iter;

/// Checks that it is legal to call methods of the trait corresponding
/// to `trait_id` (this only cares about the trait, not the specific
/// method that is called).
pub fn check_legal_trait_for_method_call(
    tcx: TyCtxt<'_>,
    span: Span,
    receiver: Option<Span>,
    expr_span: Span,
    trait_id: DefId,
) {
    if tcx.lang_items().drop_trait() == Some(trait_id) {
        let mut err = struct_span_err!(tcx.sess, span, E0040, "explicit use of destructor method");
        err.span_label(span, "explicit destructor calls not allowed");

        let (sp, suggestion) = receiver
            .and_then(|s| tcx.sess.source_map().span_to_snippet(s).ok())
            .filter(|snippet| !snippet.is_empty())
            .map(|snippet| (expr_span, format!("drop({})", snippet)))
            .unwrap_or_else(|| (span, "drop".to_string()));

        err.span_suggestion(
            sp,
            "consider using `drop` function",
            suggestion,
            Applicability::MaybeIncorrect,
        );

        err.emit();
    }
}

enum CallStep<'tcx> {
    Builtin(Ty<'tcx>),
    DeferredClosure(ty::FnSig<'tcx>),
    /// E.g., enum variant constructors.
    Overloaded(MethodCallee<'tcx>),
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn check_call(
        &self,
        call_expr: &'tcx hir::Expr<'tcx>,
        callee_expr: &'tcx hir::Expr<'tcx>,
        arg_exprs: &'tcx [hir::Expr<'tcx>],
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let original_callee_ty = match &callee_expr.kind {
            hir::ExprKind::Path(hir::QPath::Resolved(..) | hir::QPath::TypeRelative(..)) => self
                .check_expr_with_expectation_and_args(
                    callee_expr,
                    Expectation::NoExpectation,
                    arg_exprs,
                ),
            _ => self.check_expr(callee_expr),
        };

        let expr_ty = self.structurally_resolved_type(call_expr.span, original_callee_ty);

        let mut autoderef = self.autoderef(callee_expr.span, expr_ty);
        let mut result = None;
        while result.is_none() && autoderef.next().is_some() {
            result = self.try_overloaded_call_step(call_expr, callee_expr, arg_exprs, &autoderef);
        }
        self.register_predicates(autoderef.into_obligations());

        let output = match result {
            None => {
                // this will report an error since original_callee_ty is not a fn
                self.confirm_builtin_call(
                    call_expr,
                    callee_expr,
                    original_callee_ty,
                    arg_exprs,
                    expected,
                )
            }

            Some(CallStep::Builtin(callee_ty)) => {
                self.confirm_builtin_call(call_expr, callee_expr, callee_ty, arg_exprs, expected)
            }

            Some(CallStep::DeferredClosure(fn_sig)) => {
                self.confirm_deferred_closure_call(call_expr, arg_exprs, expected, fn_sig)
            }

            Some(CallStep::Overloaded(method_callee)) => {
                self.confirm_overloaded_call(call_expr, arg_exprs, expected, method_callee)
            }
        };

        // we must check that return type of called functions is WF:
        self.register_wf_obligation(output.into(), call_expr.span, traits::MiscObligation);

        output
    }

    fn try_overloaded_call_step(
        &self,
        call_expr: &'tcx hir::Expr<'tcx>,
        callee_expr: &'tcx hir::Expr<'tcx>,
        arg_exprs: &'tcx [hir::Expr<'tcx>],
        autoderef: &Autoderef<'a, 'tcx>,
    ) -> Option<CallStep<'tcx>> {
        let adjusted_ty =
            self.structurally_resolved_type(autoderef.span(), autoderef.final_ty(false));
        debug!(
            "try_overloaded_call_step(call_expr={:?}, adjusted_ty={:?})",
            call_expr, adjusted_ty
        );

        // If the callee is a bare function or a closure, then we're all set.
        match *adjusted_ty.kind() {
            ty::FnDef(..) | ty::FnPtr(_) => {
                let adjustments = self.adjust_steps(autoderef);
                self.apply_adjustments(callee_expr, adjustments);
                return Some(CallStep::Builtin(adjusted_ty));
            }

            ty::Closure(def_id, substs) => {
                assert_eq!(def_id.krate, LOCAL_CRATE);

                // Check whether this is a call to a closure where we
                // haven't yet decided on whether the closure is fn vs
                // fnmut vs fnonce. If so, we have to defer further processing.
                if self.closure_kind(substs).is_none() {
                    let closure_sig = substs.as_closure().sig();
                    let closure_sig = self
                        .replace_bound_vars_with_fresh_vars(
                            call_expr.span,
                            infer::FnCall,
                            closure_sig,
                        )
                        .0;
                    let adjustments = self.adjust_steps(autoderef);
                    self.record_deferred_call_resolution(
                        def_id,
                        DeferredCallResolution {
                            call_expr,
                            callee_expr,
                            adjusted_ty,
                            adjustments,
                            fn_sig: closure_sig,
                            closure_substs: substs,
                        },
                    );
                    return Some(CallStep::DeferredClosure(closure_sig));
                }
            }

            // Hack: we know that there are traits implementing Fn for &F
            // where F:Fn and so forth. In the particular case of types
            // like `x: &mut FnMut()`, if there is a call `x()`, we would
            // normally translate to `FnMut::call_mut(&mut x, ())`, but
            // that winds up requiring `mut x: &mut FnMut()`. A little
            // over the top. The simplest fix by far is to just ignore
            // this case and deref again, so we wind up with
            // `FnMut::call_mut(&mut *x, ())`.
            ty::Ref(..) if autoderef.step_count() == 0 => {
                return None;
            }

            _ => {}
        }

        // Now, we look for the implementation of a Fn trait on the object's type.
        // We first do it with the explicit instruction to look for an impl of
        // `Fn<Tuple>`, with the tuple `Tuple` having an arity corresponding
        // to the number of call parameters.
        // If that fails (or_else branch), we try again without specifying the
        // shape of the tuple (hence the None). This allows to detect an Fn trait
        // is implemented, and use this information for diagnostic.
        self.try_overloaded_call_traits(call_expr, adjusted_ty, Some(arg_exprs))
            .or_else(|| self.try_overloaded_call_traits(call_expr, adjusted_ty, None))
            .map(|(autoref, method)| {
                let mut adjustments = self.adjust_steps(autoderef);
                adjustments.extend(autoref);
                self.apply_adjustments(callee_expr, adjustments);
                CallStep::Overloaded(method)
            })
    }

    fn try_overloaded_call_traits(
        &self,
        call_expr: &hir::Expr<'_>,
        adjusted_ty: Ty<'tcx>,
        opt_arg_exprs: Option<&'tcx [hir::Expr<'tcx>]>,
    ) -> Option<(Option<Adjustment<'tcx>>, MethodCallee<'tcx>)> {
        // Try the options that are least restrictive on the caller first.
        for (opt_trait_def_id, method_name, borrow) in [
            (self.tcx.lang_items().fn_trait(), Ident::with_dummy_span(sym::call), true),
            (self.tcx.lang_items().fn_mut_trait(), Ident::with_dummy_span(sym::call_mut), true),
            (self.tcx.lang_items().fn_once_trait(), Ident::with_dummy_span(sym::call_once), false),
        ] {
            let trait_def_id = match opt_trait_def_id {
                Some(def_id) => def_id,
                None => continue,
            };

            let opt_input_types = opt_arg_exprs.map(|arg_exprs| {
                [self.tcx.mk_tup(arg_exprs.iter().map(|e| {
                    self.next_ty_var(TypeVariableOrigin {
                        kind: TypeVariableOriginKind::TypeInference,
                        span: e.span,
                    })
                }))]
            });
            let opt_input_types = opt_input_types.as_ref().map(AsRef::as_ref);

            if let Some(ok) = self.lookup_method_in_trait(
                call_expr.span,
                method_name,
                trait_def_id,
                adjusted_ty,
                opt_input_types,
            ) {
                let method = self.register_infer_ok_obligations(ok);
                let mut autoref = None;
                if borrow {
                    // Check for &self vs &mut self in the method signature. Since this is either
                    // the Fn or FnMut trait, it should be one of those.
                    let (region, mutbl) = if let ty::Ref(r, _, mutbl) =
                        method.sig.inputs()[0].kind()
                    {
                        (r, mutbl)
                    } else {
                        // The `fn`/`fn_mut` lang item is ill-formed, which should have
                        // caused an error elsewhere.
                        self.tcx
                            .sess
                            .delay_span_bug(call_expr.span, "input to call/call_mut is not a ref?");
                        return None;
                    };

                    let mutbl = match mutbl {
                        hir::Mutability::Not => AutoBorrowMutability::Not,
                        hir::Mutability::Mut => AutoBorrowMutability::Mut {
                            // For initial two-phase borrow
                            // deployment, conservatively omit
                            // overloaded function call ops.
                            allow_two_phase_borrow: AllowTwoPhase::No,
                        },
                    };
                    autoref = Some(Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::Ref(*region, mutbl)),
                        target: method.sig.inputs()[0],
                    });
                }
                return Some((autoref, method));
            }
        }

        None
    }

    /// Give appropriate suggestion when encountering `||{/* not callable */}()`, where the
    /// likely intention is to call the closure, suggest `(||{})()`. (#55851)
    fn identify_bad_closure_def_and_call(
        &self,
        err: &mut DiagnosticBuilder<'a>,
        hir_id: hir::HirId,
        callee_node: &hir::ExprKind<'_>,
        callee_span: Span,
    ) {
        let hir_id = self.tcx.hir().get_parent_node(hir_id);
        let parent_node = self.tcx.hir().get(hir_id);
        if let (
            hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Closure(_, _, _, sp, ..), .. }),
            hir::ExprKind::Block(..),
        ) = (parent_node, callee_node)
        {
            let start = sp.shrink_to_lo();
            let end = callee_span.shrink_to_hi();
            err.multipart_suggestion(
                "if you meant to create this closure and immediately call it, surround the \
                closure with parentheses",
                vec![(start, "(".to_string()), (end, ")".to_string())],
                Applicability::MaybeIncorrect,
            );
        }
    }

    /// Give appropriate suggestion when encountering `[("a", 0) ("b", 1)]`, where the
    /// likely intention is to create an array containing tuples.
    fn maybe_suggest_bad_array_definition(
        &self,
        err: &mut DiagnosticBuilder<'a>,
        call_expr: &'tcx hir::Expr<'tcx>,
        callee_expr: &'tcx hir::Expr<'tcx>,
    ) -> bool {
        let hir_id = self.tcx.hir().get_parent_node(call_expr.hir_id);
        let parent_node = self.tcx.hir().get(hir_id);
        if let (
            hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Array(_), .. }),
            hir::ExprKind::Tup(exp),
            hir::ExprKind::Call(_, args),
        ) = (parent_node, &callee_expr.kind, &call_expr.kind)
        {
            if args.len() == exp.len() {
                let start = callee_expr.span.shrink_to_hi();
                err.span_suggestion(
                    start,
                    "consider separating array elements with a comma",
                    ",".to_string(),
                    Applicability::MaybeIncorrect,
                );
                return true;
            }
        }
        false
    }

    fn confirm_builtin_call(
        &self,
        call_expr: &'tcx hir::Expr<'tcx>,
        callee_expr: &'tcx hir::Expr<'tcx>,
        callee_ty: Ty<'tcx>,
        arg_exprs: &'tcx [hir::Expr<'tcx>],
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let (fn_sig, def_id) = match *callee_ty.kind() {
            ty::FnDef(def_id, subst) => {
                let fn_sig = self.tcx.fn_sig(def_id).subst(self.tcx, subst);

                // Unit testing: function items annotated with
                // `#[rustc_evaluate_where_clauses]` trigger special output
                // to let us test the trait evaluation system.
                if self.tcx.has_attr(def_id, sym::rustc_evaluate_where_clauses) {
                    let predicates = self.tcx.predicates_of(def_id);
                    let predicates = predicates.instantiate(self.tcx, subst);
                    for (predicate, predicate_span) in
                        predicates.predicates.iter().zip(&predicates.spans)
                    {
                        let obligation = Obligation::new(
                            ObligationCause::dummy_with_span(callee_expr.span),
                            self.param_env,
                            *predicate,
                        );
                        let result = self.infcx.evaluate_obligation(&obligation);
                        self.tcx
                            .sess
                            .struct_span_err(
                                callee_expr.span,
                                &format!("evaluate({:?}) = {:?}", predicate, result),
                            )
                            .span_label(*predicate_span, "predicate")
                            .emit();
                    }
                }
                (fn_sig, Some(def_id))
            }
            ty::FnPtr(sig) => (sig, None),
            ref t => {
                let mut unit_variant = None;
                let mut removal_span = call_expr.span;
                if let ty::Adt(adt_def, ..) = t {
                    if adt_def.is_enum() {
                        if let hir::ExprKind::Call(expr, _) = call_expr.kind {
                            removal_span =
                                expr.span.shrink_to_hi().to(call_expr.span.shrink_to_hi());
                            unit_variant =
                                self.tcx.sess.source_map().span_to_snippet(expr.span).ok();
                        }
                    }
                }

                let callee_ty = self.resolve_vars_if_possible(callee_ty);
                let mut err = type_error_struct!(
                    self.tcx.sess,
                    callee_expr.span,
                    callee_ty,
                    E0618,
                    "expected function, found {}",
                    match unit_variant {
                        Some(ref path) => format!("enum variant `{}`", path),
                        None => format!("`{}`", callee_ty),
                    }
                );

                self.identify_bad_closure_def_and_call(
                    &mut err,
                    call_expr.hir_id,
                    &callee_expr.kind,
                    callee_expr.span,
                );

                if let Some(ref path) = unit_variant {
                    err.span_suggestion_verbose(
                        removal_span,
                        &format!(
                            "`{}` is a unit variant, you need to write it without the parentheses",
                            path
                        ),
                        String::new(),
                        Applicability::MachineApplicable,
                    );
                }

                let mut inner_callee_path = None;
                let def = match callee_expr.kind {
                    hir::ExprKind::Path(ref qpath) => {
                        self.typeck_results.borrow().qpath_res(qpath, callee_expr.hir_id)
                    }
                    hir::ExprKind::Call(ref inner_callee, _) => {
                        // If the call spans more than one line and the callee kind is
                        // itself another `ExprCall`, that's a clue that we might just be
                        // missing a semicolon (Issue #51055)
                        let call_is_multiline =
                            self.tcx.sess.source_map().is_multiline(call_expr.span);
                        if call_is_multiline {
                            err.span_suggestion(
                                callee_expr.span.shrink_to_hi(),
                                "consider using a semicolon here",
                                ";".to_owned(),
                                Applicability::MaybeIncorrect,
                            );
                        }
                        if let hir::ExprKind::Path(ref inner_qpath) = inner_callee.kind {
                            inner_callee_path = Some(inner_qpath);
                            self.typeck_results.borrow().qpath_res(inner_qpath, inner_callee.hir_id)
                        } else {
                            Res::Err
                        }
                    }
                    _ => Res::Err,
                };

                if !self.maybe_suggest_bad_array_definition(&mut err, call_expr, callee_expr) {
                    err.span_label(call_expr.span, "call expression requires function");
                }

                if let Some(span) = self.tcx.hir().res_span(def) {
                    let callee_ty = callee_ty.to_string();
                    let label = match (unit_variant, inner_callee_path) {
                        (Some(path), _) => Some(format!("`{}` defined here", path)),
                        (_, Some(hir::QPath::Resolved(_, path))) => self
                            .tcx
                            .sess
                            .source_map()
                            .span_to_snippet(path.span)
                            .ok()
                            .map(|p| format!("`{}` defined here returns `{}`", p, callee_ty)),
                        _ => {
                            match def {
                                // Emit a different diagnostic for local variables, as they are not
                                // type definitions themselves, but rather variables *of* that type.
                                Res::Local(hir_id) => Some(format!(
                                    "`{}` has type `{}`",
                                    self.tcx.hir().name(hir_id),
                                    callee_ty
                                )),
                                Res::Def(kind, def_id) if kind.ns() == Some(Namespace::ValueNS) => {
                                    Some(format!(
                                        "`{}` defined here",
                                        self.tcx.def_path_str(def_id),
                                    ))
                                }
                                _ => Some(format!("`{}` defined here", callee_ty)),
                            }
                        }
                    };
                    if let Some(label) = label {
                        err.span_label(span, label);
                    }
                }
                err.emit();

                // This is the "default" function signature, used in case of error.
                // In that case, we check each argument against "error" in order to
                // set up all the node type bindings.
                (
                    ty::Binder::dummy(self.tcx.mk_fn_sig(
                        self.err_args(arg_exprs.len()).into_iter(),
                        self.tcx.ty_error(),
                        false,
                        hir::Unsafety::Normal,
                        abi::Abi::Rust,
                    )),
                    None,
                )
            }
        };

        // Replace any late-bound regions that appear in the function
        // signature with region variables. We also have to
        // renormalize the associated types at this point, since they
        // previously appeared within a `Binder<>` and hence would not
        // have been normalized before.
        let fn_sig =
            self.replace_bound_vars_with_fresh_vars(call_expr.span, infer::FnCall, fn_sig).0;
        let fn_sig = self.normalize_associated_types_in(call_expr.span, fn_sig);

        // Call the generic checker.
        let expected_arg_tys = self.expected_inputs_for_expected_output(
            call_expr.span,
            expected,
            fn_sig.output(),
            fn_sig.inputs(),
        );
        self.check_argument_types(
            call_expr.span,
            call_expr,
            fn_sig.inputs(),
            expected_arg_tys,
            arg_exprs,
            fn_sig.c_variadic,
            TupleArgumentsFlag::DontTupleArguments,
            def_id,
        );

        fn_sig.output()
    }

    fn confirm_deferred_closure_call(
        &self,
        call_expr: &'tcx hir::Expr<'tcx>,
        arg_exprs: &'tcx [hir::Expr<'tcx>],
        expected: Expectation<'tcx>,
        fn_sig: ty::FnSig<'tcx>,
    ) -> Ty<'tcx> {
        // `fn_sig` is the *signature* of the cosure being called. We
        // don't know the full details yet (`Fn` vs `FnMut` etc), but we
        // do know the types expected for each argument and the return
        // type.

        let expected_arg_tys = self.expected_inputs_for_expected_output(
            call_expr.span,
            expected,
            fn_sig.output(),
            fn_sig.inputs(),
        );

        self.check_argument_types(
            call_expr.span,
            call_expr,
            fn_sig.inputs(),
            expected_arg_tys,
            arg_exprs,
            fn_sig.c_variadic,
            TupleArgumentsFlag::TupleArguments,
            None,
        );

        fn_sig.output()
    }

    fn confirm_overloaded_call(
        &self,
        call_expr: &'tcx hir::Expr<'tcx>,
        arg_exprs: &'tcx [hir::Expr<'tcx>],
        expected: Expectation<'tcx>,
        method_callee: MethodCallee<'tcx>,
    ) -> Ty<'tcx> {
        let output_type = self.check_method_argument_types(
            call_expr.span,
            call_expr,
            Ok(method_callee),
            arg_exprs,
            TupleArgumentsFlag::TupleArguments,
            expected,
        );

        self.write_method_call(call_expr.hir_id, method_callee);
        output_type
    }
}

#[derive(Debug)]
pub struct DeferredCallResolution<'tcx> {
    call_expr: &'tcx hir::Expr<'tcx>,
    callee_expr: &'tcx hir::Expr<'tcx>,
    adjusted_ty: Ty<'tcx>,
    adjustments: Vec<Adjustment<'tcx>>,
    fn_sig: ty::FnSig<'tcx>,
    closure_substs: SubstsRef<'tcx>,
}

impl<'a, 'tcx> DeferredCallResolution<'tcx> {
    pub fn resolve(self, fcx: &FnCtxt<'a, 'tcx>) {
        debug!("DeferredCallResolution::resolve() {:?}", self);

        // we should not be invoked until the closure kind has been
        // determined by upvar inference
        assert!(fcx.closure_kind(self.closure_substs).is_some());

        // We may now know enough to figure out fn vs fnmut etc.
        match fcx.try_overloaded_call_traits(self.call_expr, self.adjusted_ty, None) {
            Some((autoref, method_callee)) => {
                // One problem is that when we get here, we are going
                // to have a newly instantiated function signature
                // from the call trait. This has to be reconciled with
                // the older function signature we had before. In
                // principle we *should* be able to fn_sigs(), but we
                // can't because of the annoying need for a TypeTrace.
                // (This always bites me, should find a way to
                // refactor it.)
                let method_sig = method_callee.sig;

                debug!("attempt_resolution: method_callee={:?}", method_callee);

                for (method_arg_ty, self_arg_ty) in
                    iter::zip(method_sig.inputs().iter().skip(1), self.fn_sig.inputs())
                {
                    fcx.demand_eqtype(self.call_expr.span, *self_arg_ty, *method_arg_ty);
                }

                fcx.demand_eqtype(self.call_expr.span, method_sig.output(), self.fn_sig.output());

                let mut adjustments = self.adjustments;
                adjustments.extend(autoref);
                fcx.apply_adjustments(self.callee_expr, adjustments);

                fcx.write_method_call(self.call_expr.hir_id, method_callee);
            }
            None => {
                // This can happen if `#![no_core]` is used and the `fn/fn_mut/fn_once`
                // lang items are not defined (issue #86238).
                let mut err = fcx.inh.tcx.sess.struct_span_err(
                    self.call_expr.span,
                    "failed to find an overloaded call trait for closure call",
                );
                err.help(
                    "make sure the `fn`/`fn_mut`/`fn_once` lang items are defined \
                     and have associated `call`/`call_mut`/`call_once` functions",
                );
                err.emit();
            }
        }
    }
}
