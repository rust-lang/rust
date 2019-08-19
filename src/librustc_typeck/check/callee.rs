use super::autoderef::Autoderef;
use super::method::MethodCallee;
use super::{Expectation, FnCtxt, Needs, TupleArgumentsFlag};

use errors::{Applicability, DiagnosticBuilder};
use hir::def::Res;
use hir::def_id::{DefId, LOCAL_CRATE};
use rustc::ty::adjustment::{Adjust, Adjustment, AllowTwoPhase, AutoBorrow, AutoBorrowMutability};
use rustc::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc::{infer, traits};
use rustc::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_target::spec::abi;
use syntax::ast::Ident;
use syntax_pos::Span;

use rustc::hir;

/// Checks that it is legal to call methods of the trait corresponding
/// to `trait_id` (this only cares about the trait, not the specific
/// method that is called).
pub fn check_legal_trait_for_method_call(tcx: TyCtxt<'_>, span: Span, trait_id: DefId) {
    if tcx.lang_items().drop_trait() == Some(trait_id) {
        struct_span_err!(tcx.sess, span, E0040, "explicit use of destructor method")
            .span_label(span, "explicit destructor calls not allowed")
            .emit();
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
        call_expr: &'tcx hir::Expr,
        callee_expr: &'tcx hir::Expr,
        arg_exprs: &'tcx [hir::Expr],
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let original_callee_ty = self.check_expr(callee_expr);
        let expr_ty = self.structurally_resolved_type(call_expr.span, original_callee_ty);

        let mut autoderef = self.autoderef(callee_expr.span, expr_ty);
        let mut result = None;
        while result.is_none() && autoderef.next().is_some() {
            result = self.try_overloaded_call_step(call_expr, callee_expr, arg_exprs, &autoderef);
        }
        autoderef.finalize(self);

        let output = match result {
            None => {
                // this will report an error since original_callee_ty is not a fn
                self.confirm_builtin_call(call_expr, original_callee_ty, arg_exprs, expected)
            }

            Some(CallStep::Builtin(callee_ty)) => {
                self.confirm_builtin_call(call_expr, callee_ty, arg_exprs, expected)
            }

            Some(CallStep::DeferredClosure(fn_sig)) => {
                self.confirm_deferred_closure_call(call_expr, arg_exprs, expected, fn_sig)
            }

            Some(CallStep::Overloaded(method_callee)) => {
                self.confirm_overloaded_call(call_expr, arg_exprs, expected, method_callee)
            }
        };

        // we must check that return type of called functions is WF:
        self.register_wf_obligation(output, call_expr.span, traits::MiscObligation);

        output
    }

    fn try_overloaded_call_step(
        &self,
        call_expr: &'tcx hir::Expr,
        callee_expr: &'tcx hir::Expr,
        arg_exprs: &'tcx [hir::Expr],
        autoderef: &Autoderef<'a, 'tcx>,
    ) -> Option<CallStep<'tcx>> {
        let adjusted_ty = autoderef.unambiguous_final_ty(self);
        debug!(
            "try_overloaded_call_step(call_expr={:?}, adjusted_ty={:?})",
            call_expr, adjusted_ty
        );

        // If the callee is a bare function or a closure, then we're all set.
        match adjusted_ty.sty {
            ty::FnDef(..) | ty::FnPtr(_) => {
                let adjustments = autoderef.adjust_steps(self, Needs::None);
                self.apply_adjustments(callee_expr, adjustments);
                return Some(CallStep::Builtin(adjusted_ty));
            }

            ty::Closure(def_id, substs) => {
                assert_eq!(def_id.krate, LOCAL_CRATE);

                // Check whether this is a call to a closure where we
                // haven't yet decided on whether the closure is fn vs
                // fnmut vs fnonce. If so, we have to defer further processing.
                if self.closure_kind(def_id, substs).is_none() {
                    let closure_ty = self.closure_sig(def_id, substs);
                    let fn_sig = self
                        .replace_bound_vars_with_fresh_vars(
                            call_expr.span,
                            infer::FnCall,
                            &closure_ty,
                        )
                        .0;
                    let adjustments = autoderef.adjust_steps(self, Needs::None);
                    self.record_deferred_call_resolution(
                        def_id,
                        DeferredCallResolution {
                            call_expr,
                            callee_expr,
                            adjusted_ty,
                            adjustments,
                            fn_sig,
                            closure_def_id: def_id,
                            closure_substs: substs,
                        },
                    );
                    return Some(CallStep::DeferredClosure(fn_sig));
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
                let mut adjustments = autoderef.adjust_steps(self, Needs::None);
                adjustments.extend(autoref);
                self.apply_adjustments(callee_expr, adjustments);
                CallStep::Overloaded(method)
            })
    }

    fn try_overloaded_call_traits(
        &self,
        call_expr: &hir::Expr,
        adjusted_ty: Ty<'tcx>,
        opt_arg_exprs: Option<&'tcx [hir::Expr]>,
    ) -> Option<(Option<Adjustment<'tcx>>, MethodCallee<'tcx>)> {
        // Try the options that are least restrictive on the caller first.
        for &(opt_trait_def_id, method_name, borrow) in &[
            (
                self.tcx.lang_items().fn_trait(),
                Ident::from_str("call"),
                true,
            ),
            (
                self.tcx.lang_items().fn_mut_trait(),
                Ident::from_str("call_mut"),
                true,
            ),
            (
                self.tcx.lang_items().fn_once_trait(),
                Ident::from_str("call_once"),
                false,
            ),
        ] {
            let trait_def_id = match opt_trait_def_id {
                Some(def_id) => def_id,
                None => continue,
            };

            let opt_input_types = opt_arg_exprs.map(|arg_exprs| [self.tcx.mk_tup(
                arg_exprs
                .iter()
                .map(|e| {
                    self.next_ty_var(TypeVariableOrigin {
                        kind: TypeVariableOriginKind::TypeInference,
                        span: e.span,
                    })
                })
            )]);
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
                    if let ty::Ref(region, _, mutbl) = method.sig.inputs()[0].sty {
                        let mutbl = match mutbl {
                            hir::MutImmutable => AutoBorrowMutability::Immutable,
                            hir::MutMutable => AutoBorrowMutability::Mutable {
                                // For initial two-phase borrow
                                // deployment, conservatively omit
                                // overloaded function call ops.
                                allow_two_phase_borrow: AllowTwoPhase::No,
                            },
                        };
                        autoref = Some(Adjustment {
                            kind: Adjust::Borrow(AutoBorrow::Ref(region, mutbl)),
                            target: method.sig.inputs()[0],
                        });
                    }
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
        callee_node: &hir::ExprKind,
        callee_span: Span,
    ) {
        let hir_id = self.tcx.hir().get_parent_node(hir_id);
        let parent_node = self.tcx.hir().get(hir_id);
        if let (
            hir::Node::Expr(hir::Expr { node: hir::ExprKind::Closure(_, _, _, sp, ..), .. }),
            hir::ExprKind::Block(..),
        ) = (parent_node, callee_node) {
            let start = sp.shrink_to_lo();
            let end = self.tcx.sess.source_map().next_point(callee_span);
            err.multipart_suggestion(
                "if you meant to create this closure and immediately call it, surround the \
                closure with parenthesis",
                vec![(start, "(".to_string()), (end, ")".to_string())],
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn confirm_builtin_call(
        &self,
        call_expr: &hir::Expr,
        callee_ty: Ty<'tcx>,
        arg_exprs: &'tcx [hir::Expr],
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let (fn_sig, def_span) = match callee_ty.sty {
            ty::FnDef(def_id, _) => (
                callee_ty.fn_sig(self.tcx),
                self.tcx.hir().span_if_local(def_id),
            ),
            ty::FnPtr(sig) => (sig, None),
            ref t => {
                let mut unit_variant = None;
                if let &ty::Adt(adt_def, ..) = t {
                    if adt_def.is_enum() {
                        if let hir::ExprKind::Call(ref expr, _) = call_expr.node {
                            unit_variant = Some(self.tcx.hir().hir_to_pretty_string(expr.hir_id))
                        }
                    }
                }

                if let hir::ExprKind::Call(ref callee, _) = call_expr.node {
                    let mut err = type_error_struct!(
                        self.tcx.sess,
                        callee.span,
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
                        &callee.node,
                        callee.span,
                    );

                    if let Some(ref path) = unit_variant {
                        err.span_suggestion(
                            call_expr.span,
                            &format!(
                                "`{}` is a unit variant, you need to write it \
                                 without the parenthesis",
                                path
                            ),
                            path.to_string(),
                            Applicability::MachineApplicable,
                        );
                    }

                    let mut inner_callee_path = None;
                    let def = match callee.node {
                        hir::ExprKind::Path(ref qpath) => {
                            self.tables.borrow().qpath_res(qpath, callee.hir_id)
                        }
                        hir::ExprKind::Call(ref inner_callee, _) => {
                            // If the call spans more than one line and the callee kind is
                            // itself another `ExprCall`, that's a clue that we might just be
                            // missing a semicolon (Issue #51055)
                            let call_is_multiline =
                                self.tcx.sess.source_map().is_multiline(call_expr.span);
                            if call_is_multiline {
                                let span = self.tcx.sess.source_map().next_point(callee.span);
                                err.span_suggestion(
                                    span,
                                    "try adding a semicolon",
                                    ";".to_owned(),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                            if let hir::ExprKind::Path(ref inner_qpath) = inner_callee.node {
                                inner_callee_path = Some(inner_qpath);
                                self.tables
                                    .borrow()
                                    .qpath_res(inner_qpath, inner_callee.hir_id)
                            } else {
                                Res::Err
                            }
                        }
                        _ => Res::Err,
                    };

                    err.span_label(call_expr.span, "call expression requires function");

                    let def_span = match def {
                        Res::Err => None,
                        Res::Local(id) => {
                            Some(self.tcx.hir().span(id))
                        },
                        _ => def
                            .opt_def_id()
                            .and_then(|did| self.tcx.hir().span_if_local(did)),
                    };
                    if let Some(span) = def_span {
                        let label = match (unit_variant, inner_callee_path) {
                            (Some(path), _) => format!("`{}` defined here", path),
                            (_, Some(hir::QPath::Resolved(_, path))) => format!(
                                "`{}` defined here returns `{}`",
                                path,
                                callee_ty.to_string()
                            ),
                            _ => format!("`{}` defined here", callee_ty.to_string()),
                        };
                        err.span_label(span, label);
                    }
                    err.emit();
                } else {
                    bug!(
                        "call_expr.node should be an ExprKind::Call, got {:?}",
                        call_expr.node
                    );
                }

                // This is the "default" function signature, used in case of error.
                // In that case, we check each argument against "error" in order to
                // set up all the node type bindings.
                (
                    ty::Binder::bind(self.tcx.mk_fn_sig(
                        self.err_args(arg_exprs.len()).into_iter(),
                        self.tcx.types.err,
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
        let fn_sig = self
            .replace_bound_vars_with_fresh_vars(call_expr.span, infer::FnCall, &fn_sig)
            .0;
        let fn_sig = self.normalize_associated_types_in(call_expr.span, &fn_sig);

        let inputs = if fn_sig.c_variadic {
            if fn_sig.inputs().len() > 1 {
                &fn_sig.inputs()[..fn_sig.inputs().len() - 1]
            } else {
                span_bug!(call_expr.span,
                          "C-variadic functions are only valid with one or more fixed arguments");
            }
        } else {
            &fn_sig.inputs()[..]
        };
        // Call the generic checker.
        let expected_arg_tys = self.expected_inputs_for_expected_output(
            call_expr.span,
            expected,
            fn_sig.output(),
            inputs,
        );
        self.check_argument_types(
            call_expr.span,
            call_expr.span,
            inputs,
            &expected_arg_tys[..],
            arg_exprs,
            fn_sig.c_variadic,
            TupleArgumentsFlag::DontTupleArguments,
            def_span,
        );

        fn_sig.output()
    }

    fn confirm_deferred_closure_call(
        &self,
        call_expr: &hir::Expr,
        arg_exprs: &'tcx [hir::Expr],
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
            fn_sig.output().clone(),
            fn_sig.inputs(),
        );

        self.check_argument_types(
            call_expr.span,
            call_expr.span,
            fn_sig.inputs(),
            &expected_arg_tys,
            arg_exprs,
            fn_sig.c_variadic,
            TupleArgumentsFlag::TupleArguments,
            None,
        );

        fn_sig.output()
    }

    fn confirm_overloaded_call(
        &self,
        call_expr: &hir::Expr,
        arg_exprs: &'tcx [hir::Expr],
        expected: Expectation<'tcx>,
        method_callee: MethodCallee<'tcx>,
    ) -> Ty<'tcx> {
        let output_type = self.check_method_argument_types(
            call_expr.span,
            call_expr.span,
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
    call_expr: &'tcx hir::Expr,
    callee_expr: &'tcx hir::Expr,
    adjusted_ty: Ty<'tcx>,
    adjustments: Vec<Adjustment<'tcx>>,
    fn_sig: ty::FnSig<'tcx>,
    closure_def_id: DefId,
    closure_substs: ty::ClosureSubsts<'tcx>,
}

impl<'a, 'tcx> DeferredCallResolution<'tcx> {
    pub fn resolve(self, fcx: &FnCtxt<'a, 'tcx>) {
        debug!("DeferredCallResolution::resolve() {:?}", self);

        // we should not be invoked until the closure kind has been
        // determined by upvar inference
        assert!(fcx
            .closure_kind(self.closure_def_id, self.closure_substs)
            .is_some());

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
                    method_sig.inputs().iter().skip(1).zip(self.fn_sig.inputs())
                {
                    fcx.demand_eqtype(self.call_expr.span, &self_arg_ty, &method_arg_ty);
                }

                fcx.demand_eqtype(
                    self.call_expr.span,
                    method_sig.output(),
                    self.fn_sig.output(),
                );

                let mut adjustments = self.adjustments;
                adjustments.extend(autoref);
                fcx.apply_adjustments(self.callee_expr, adjustments);

                fcx.write_method_call(self.call_expr.hir_id, method_callee);
            }
            None => {
                span_bug!(
                    self.call_expr.span,
                    "failed to find an overloaded call trait for closure call"
                );
            }
        }
    }
}
