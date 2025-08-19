use rustc_errors::{Applicability, Diag, MultiSpan, listify};
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::intravisit::Visitor;
use rustc_infer::infer::DefineOpaqueTypes;
use rustc_middle::bug;
use rustc_middle::ty::adjustment::AllowTwoPhase;
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, AssocItem, BottomUpFolder, Ty, TypeFoldable, TypeVisitableExt};
use rustc_span::{DUMMY_SP, Ident, Span, sym};
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::ObligationCause;
use tracing::instrument;

use super::method::probe;
use crate::FnCtxt;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(crate) fn emit_type_mismatch_suggestions(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'tcx>,
        expr_ty: Ty<'tcx>,
        expected: Ty<'tcx>,
        expected_ty_expr: Option<&'tcx hir::Expr<'tcx>>,
        error: Option<TypeError<'tcx>>,
    ) {
        if expr_ty == expected {
            return;
        }
        self.annotate_alternative_method_deref(err, expr, error);
        self.explain_self_literal(err, expr, expected, expr_ty);

        // Use `||` to give these suggestions a precedence
        let suggested = self.suggest_missing_parentheses(err, expr)
            || self.suggest_missing_unwrap_expect(err, expr, expected, expr_ty)
            || self.suggest_remove_last_method_call(err, expr, expected)
            || self.suggest_associated_const(err, expr, expected)
            || self.suggest_semicolon_in_repeat_expr(err, expr, expr_ty)
            || self.suggest_deref_ref_or_into(err, expr, expected, expr_ty, expected_ty_expr)
            || self.suggest_option_to_bool(err, expr, expr_ty, expected)
            || self.suggest_compatible_variants(err, expr, expected, expr_ty)
            || self.suggest_non_zero_new_unwrap(err, expr, expected, expr_ty)
            || self.suggest_calling_boxed_future_when_appropriate(err, expr, expected, expr_ty)
            || self.suggest_no_capture_closure(err, expected, expr_ty)
            || self.suggest_boxing_when_appropriate(
                err,
                expr.peel_blocks().span,
                expr.hir_id,
                expected,
                expr_ty,
            )
            || self.suggest_block_to_brackets_peeling_refs(err, expr, expr_ty, expected)
            || self.suggest_copied_cloned_or_as_ref(err, expr, expr_ty, expected)
            || self.suggest_clone_for_ref(err, expr, expr_ty, expected)
            || self.suggest_into(err, expr, expr_ty, expected)
            || self.suggest_floating_point_literal(err, expr, expected)
            || self.suggest_null_ptr_for_literal_zero_given_to_ptr_arg(err, expr, expected)
            || self.suggest_coercing_result_via_try_operator(err, expr, expected, expr_ty)
            || self.suggest_returning_value_after_loop(err, expr, expected);

        if !suggested {
            self.note_source_of_type_mismatch_constraint(
                err,
                expr,
                TypeMismatchSource::Ty(expected),
            );
        }
    }

    pub(crate) fn emit_coerce_suggestions(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'tcx>,
        expr_ty: Ty<'tcx>,
        expected: Ty<'tcx>,
        expected_ty_expr: Option<&'tcx hir::Expr<'tcx>>,
        error: Option<TypeError<'tcx>>,
    ) {
        if expr_ty == expected {
            return;
        }

        self.annotate_expected_due_to_let_ty(err, expr, error);
        self.annotate_loop_expected_due_to_inference(err, expr, error);
        if self.annotate_mut_binding_to_immutable_binding(err, expr, expr_ty, expected, error) {
            return;
        }

        // FIXME(#73154): For now, we do leak check when coercing function
        // pointers in typeck, instead of only during borrowck. This can lead
        // to these `RegionsInsufficientlyPolymorphic` errors that aren't helpful.
        if matches!(error, Some(TypeError::RegionsInsufficientlyPolymorphic(..))) {
            return;
        }

        if self.is_destruct_assignment_desugaring(expr) {
            return;
        }
        self.emit_type_mismatch_suggestions(err, expr, expr_ty, expected, expected_ty_expr, error);
        self.note_type_is_not_clone(err, expected, expr_ty, expr);
        self.note_internal_mutation_in_method(err, expr, Some(expected), expr_ty);
        self.suggest_method_call_on_range_literal(err, expr, expr_ty, expected);
        self.suggest_return_binding_for_missing_tail_expr(err, expr, expr_ty, expected);
        self.note_wrong_return_ty_due_to_generic_arg(err, expr, expr_ty);
    }

    /// Really hacky heuristic to remap an `assert_eq!` error to the user
    /// expressions provided to the macro.
    fn adjust_expr_for_assert_eq_macro(
        &self,
        found_expr: &mut &'tcx hir::Expr<'tcx>,
        expected_expr: &mut Option<&'tcx hir::Expr<'tcx>>,
    ) {
        let Some(expected_expr) = expected_expr else {
            return;
        };

        if !found_expr.span.eq_ctxt(expected_expr.span) {
            return;
        }

        if !found_expr
            .span
            .ctxt()
            .outer_expn_data()
            .macro_def_id
            .is_some_and(|def_id| self.tcx.is_diagnostic_item(sym::assert_eq_macro, def_id))
        {
            return;
        }

        let hir::ExprKind::Unary(
            hir::UnOp::Deref,
            hir::Expr { kind: hir::ExprKind::Path(found_path), .. },
        ) = found_expr.kind
        else {
            return;
        };
        let hir::ExprKind::Unary(
            hir::UnOp::Deref,
            hir::Expr { kind: hir::ExprKind::Path(expected_path), .. },
        ) = expected_expr.kind
        else {
            return;
        };

        for (path, name, idx, var) in [
            (expected_path, "left_val", 0, expected_expr),
            (found_path, "right_val", 1, found_expr),
        ] {
            if let hir::QPath::Resolved(_, path) = path
                && let [segment] = path.segments
                && segment.ident.name.as_str() == name
                && let Res::Local(hir_id) = path.res
                && let Some((_, hir::Node::Expr(match_expr))) =
                    self.tcx.hir_parent_iter(hir_id).nth(2)
                && let hir::ExprKind::Match(scrutinee, _, _) = match_expr.kind
                && let hir::ExprKind::Tup(exprs) = scrutinee.kind
                && let hir::ExprKind::AddrOf(_, _, macro_arg) = exprs[idx].kind
            {
                *var = macro_arg;
            }
        }
    }

    /// Requires that the two types unify, and prints an error message if
    /// they don't.
    pub(crate) fn demand_suptype(&self, sp: Span, expected: Ty<'tcx>, actual: Ty<'tcx>) {
        if let Err(e) = self.demand_suptype_diag(sp, expected, actual) {
            e.emit();
        }
    }

    pub(crate) fn demand_suptype_diag(
        &'a self,
        sp: Span,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
    ) -> Result<(), Diag<'a>> {
        self.demand_suptype_with_origin(&self.misc(sp), expected, actual)
    }

    #[instrument(skip(self), level = "debug")]
    pub(crate) fn demand_suptype_with_origin(
        &'a self,
        cause: &ObligationCause<'tcx>,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
    ) -> Result<(), Diag<'a>> {
        self.at(cause, self.param_env)
            .sup(DefineOpaqueTypes::Yes, expected, actual)
            .map(|infer_ok| self.register_infer_ok_obligations(infer_ok))
            .map_err(|e| {
                self.err_ctxt().report_mismatched_types(cause, self.param_env, expected, actual, e)
            })
    }

    pub(crate) fn demand_eqtype(&self, sp: Span, expected: Ty<'tcx>, actual: Ty<'tcx>) {
        if let Err(err) = self.demand_eqtype_diag(sp, expected, actual) {
            err.emit();
        }
    }

    pub(crate) fn demand_eqtype_diag(
        &'a self,
        sp: Span,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
    ) -> Result<(), Diag<'a>> {
        self.demand_eqtype_with_origin(&self.misc(sp), expected, actual)
    }

    pub(crate) fn demand_eqtype_with_origin(
        &'a self,
        cause: &ObligationCause<'tcx>,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
    ) -> Result<(), Diag<'a>> {
        self.at(cause, self.param_env)
            .eq(DefineOpaqueTypes::Yes, expected, actual)
            .map(|infer_ok| self.register_infer_ok_obligations(infer_ok))
            .map_err(|e| {
                self.err_ctxt().report_mismatched_types(cause, self.param_env, expected, actual, e)
            })
    }

    pub(crate) fn demand_coerce(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        checked_ty: Ty<'tcx>,
        expected: Ty<'tcx>,
        expected_ty_expr: Option<&'tcx hir::Expr<'tcx>>,
        allow_two_phase: AllowTwoPhase,
    ) -> Ty<'tcx> {
        match self.demand_coerce_diag(expr, checked_ty, expected, expected_ty_expr, allow_two_phase)
        {
            Ok(ty) => ty,
            Err(err) => {
                err.emit();
                // Return the original type instead of an error type here, otherwise the type of `x` in
                // `let x: u32 = ();` will be a type error, causing all subsequent usages of `x` to not
                // report errors, even though `x` is definitely `u32`.
                expected
            }
        }
    }

    /// Checks that the type of `expr` can be coerced to `expected`.
    ///
    /// N.B., this code relies on `self.diverges` to be accurate. In particular, assignments to `!`
    /// will be permitted if the diverges flag is currently "always".
    #[instrument(level = "debug", skip(self, expr, expected_ty_expr, allow_two_phase))]
    pub(crate) fn demand_coerce_diag(
        &'a self,
        mut expr: &'tcx hir::Expr<'tcx>,
        checked_ty: Ty<'tcx>,
        expected: Ty<'tcx>,
        mut expected_ty_expr: Option<&'tcx hir::Expr<'tcx>>,
        allow_two_phase: AllowTwoPhase,
    ) -> Result<Ty<'tcx>, Diag<'a>> {
        let expected = if self.next_trait_solver() {
            expected
        } else {
            self.resolve_vars_with_obligations(expected)
        };

        let e = match self.coerce(expr, checked_ty, expected, allow_two_phase, None) {
            Ok(ty) => return Ok(ty),
            Err(e) => e,
        };

        self.adjust_expr_for_assert_eq_macro(&mut expr, &mut expected_ty_expr);

        self.set_tainted_by_errors(self.dcx().span_delayed_bug(
            expr.span,
            "`TypeError` when attempting coercion but no error emitted",
        ));
        let expr = expr.peel_drop_temps();
        let cause = self.misc(expr.span);
        let expr_ty = self.resolve_vars_if_possible(checked_ty);
        let mut err =
            self.err_ctxt().report_mismatched_types(&cause, self.param_env, expected, expr_ty, e);

        self.emit_coerce_suggestions(&mut err, expr, expr_ty, expected, expected_ty_expr, Some(e));

        Err(err)
    }

    /// Notes the point at which a variable is constrained to some type incompatible
    /// with some expectation given by `source`.
    pub(crate) fn note_source_of_type_mismatch_constraint(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        source: TypeMismatchSource<'tcx>,
    ) -> bool {
        let hir::ExprKind::Path(hir::QPath::Resolved(None, p)) = expr.kind else {
            return false;
        };
        let [hir::PathSegment { ident, args: None, .. }] = p.segments else {
            return false;
        };
        let hir::def::Res::Local(local_hir_id) = p.res else {
            return false;
        };
        let hir::Node::Pat(pat) = self.tcx.hir_node(local_hir_id) else {
            return false;
        };
        let (init_ty_hir_id, init) = match self.tcx.parent_hir_node(pat.hir_id) {
            hir::Node::LetStmt(hir::LetStmt { ty: Some(ty), init, .. }) => (ty.hir_id, *init),
            hir::Node::LetStmt(hir::LetStmt { init: Some(init), .. }) => (init.hir_id, Some(*init)),
            _ => return false,
        };
        let Some(init_ty) = self.node_ty_opt(init_ty_hir_id) else {
            return false;
        };

        // Locate all the usages of the relevant binding.
        struct FindExprs<'tcx> {
            hir_id: hir::HirId,
            uses: Vec<&'tcx hir::Expr<'tcx>>,
        }
        impl<'tcx> Visitor<'tcx> for FindExprs<'tcx> {
            fn visit_expr(&mut self, ex: &'tcx hir::Expr<'tcx>) {
                if let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = ex.kind
                    && let hir::def::Res::Local(hir_id) = path.res
                    && hir_id == self.hir_id
                {
                    self.uses.push(ex);
                }
                hir::intravisit::walk_expr(self, ex);
            }
        }

        let mut expr_finder = FindExprs { hir_id: local_hir_id, uses: init.into_iter().collect() };
        let body = self.tcx.hir_body_owned_by(self.body_id);
        expr_finder.visit_expr(body.value);

        // Replaces all of the variables in the given type with a fresh inference variable.
        let mut fudger = BottomUpFolder {
            tcx: self.tcx,
            ty_op: |ty| {
                if let ty::Infer(infer) = ty.kind() {
                    match infer {
                        ty::TyVar(_) => self.next_ty_var(DUMMY_SP),
                        ty::IntVar(_) => self.next_int_var(),
                        ty::FloatVar(_) => self.next_float_var(),
                        ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_) => {
                            bug!("unexpected fresh ty outside of the trait solver")
                        }
                    }
                } else {
                    ty
                }
            },
            lt_op: |_| self.tcx.lifetimes.re_erased,
            ct_op: |ct| {
                if let ty::ConstKind::Infer(_) = ct.kind() {
                    self.next_const_var(DUMMY_SP)
                } else {
                    ct
                }
            },
        };

        let expected_ty = match source {
            TypeMismatchSource::Ty(expected_ty) => expected_ty,
            // Try to deduce what the possible value of `expr` would be if the
            // incompatible arg were compatible. For example, given `Vec<i32>`
            // and `vec.push(1u32)`, we ideally want to deduce that the type of
            // `vec` *should* have been `Vec<u32>`. This will allow us to then
            // run the subsequent code with this expectation, finding out exactly
            // when this type diverged from our expectation.
            TypeMismatchSource::Arg { call_expr, incompatible_arg: idx } => {
                let hir::ExprKind::MethodCall(segment, _, args, _) = call_expr.kind else {
                    return false;
                };
                let Some(arg_ty) = self.node_ty_opt(args[idx].hir_id) else {
                    return false;
                };
                let possible_rcvr_ty = expr_finder.uses.iter().rev().find_map(|binding| {
                    let possible_rcvr_ty = self.node_ty_opt(binding.hir_id)?;
                    if possible_rcvr_ty.is_ty_var() {
                        return None;
                    }
                    // Fudge the receiver, so we can do new inference on it.
                    let possible_rcvr_ty = possible_rcvr_ty.fold_with(&mut fudger);
                    let method = self
                        .lookup_method_for_diagnostic(
                            possible_rcvr_ty,
                            segment,
                            DUMMY_SP,
                            call_expr,
                            binding,
                        )
                        .ok()?;
                    // Make sure we select the same method that we started with...
                    if Some(method.def_id)
                        != self.typeck_results.borrow().type_dependent_def_id(call_expr.hir_id)
                    {
                        return None;
                    }
                    // Unify the method signature with our incompatible arg, to
                    // do inference in the *opposite* direction and to find out
                    // what our ideal rcvr ty would look like.
                    let _ = self
                        .at(&ObligationCause::dummy(), self.param_env)
                        .eq(DefineOpaqueTypes::Yes, method.sig.inputs()[idx + 1], arg_ty)
                        .ok()?;
                    self.select_obligations_where_possible(|errs| {
                        // Yeet the errors, we're already reporting errors.
                        errs.clear();
                    });
                    Some(self.resolve_vars_if_possible(possible_rcvr_ty))
                });
                if let Some(rcvr_ty) = possible_rcvr_ty {
                    rcvr_ty
                } else {
                    return false;
                }
            }
        };

        // If our expected_ty does not equal init_ty, then it *began* as incompatible.
        // No need to note in this case...
        if !self.can_eq(self.param_env, expected_ty, init_ty.fold_with(&mut fudger)) {
            return false;
        }

        for window in expr_finder.uses.windows(2) {
            // Bindings always update their recorded type after the fact, so we
            // need to look at the *following* usage's type to see when the
            // binding became incompatible.
            let [binding, next_usage] = *window else {
                continue;
            };

            // Don't go past the binding (always gonna be a nonsense label if so)
            if binding.hir_id == expr.hir_id {
                break;
            }

            let Some(next_use_ty) = self.node_ty_opt(next_usage.hir_id) else {
                continue;
            };

            // If the type is not constrained in a way making it not possible to
            // equate with `expected_ty` by this point, skip.
            if self.can_eq(self.param_env, expected_ty, next_use_ty.fold_with(&mut fudger)) {
                continue;
            }

            if let hir::Node::Expr(parent_expr) = self.tcx.parent_hir_node(binding.hir_id)
                && let hir::ExprKind::MethodCall(segment, rcvr, args, _) = parent_expr.kind
                && rcvr.hir_id == binding.hir_id
            {
                // If our binding became incompatible while it was a receiver
                // to a method call, we may be able to make a better guess to
                // the source of a type mismatch.
                let Some(rcvr_ty) = self.node_ty_opt(rcvr.hir_id) else {
                    continue;
                };
                let rcvr_ty = rcvr_ty.fold_with(&mut fudger);
                let Ok(method) = self.lookup_method_for_diagnostic(
                    rcvr_ty,
                    segment,
                    DUMMY_SP,
                    parent_expr,
                    rcvr,
                ) else {
                    continue;
                };
                // Make sure we select the same method that we started with...
                if Some(method.def_id)
                    != self.typeck_results.borrow().type_dependent_def_id(parent_expr.hir_id)
                {
                    continue;
                }

                let ideal_rcvr_ty = rcvr_ty.fold_with(&mut fudger);
                let ideal_method = self
                    .lookup_method_for_diagnostic(
                        ideal_rcvr_ty,
                        segment,
                        DUMMY_SP,
                        parent_expr,
                        rcvr,
                    )
                    .ok()
                    .and_then(|method| {
                        let _ = self
                            .at(&ObligationCause::dummy(), self.param_env)
                            .eq(DefineOpaqueTypes::Yes, ideal_rcvr_ty, expected_ty)
                            .ok()?;
                        Some(method)
                    });

                // Find what argument caused our rcvr to become incompatible
                // with the expected ty.
                for (idx, (expected_arg_ty, arg_expr)) in
                    std::iter::zip(&method.sig.inputs()[1..], args).enumerate()
                {
                    let Some(arg_ty) = self.node_ty_opt(arg_expr.hir_id) else {
                        continue;
                    };
                    let arg_ty = arg_ty.fold_with(&mut fudger);
                    let _ =
                        self.coerce(arg_expr, arg_ty, *expected_arg_ty, AllowTwoPhase::No, None);
                    self.select_obligations_where_possible(|errs| {
                        // Yeet the errors, we're already reporting errors.
                        errs.clear();
                    });
                    // If our rcvr, after inference due to unifying the signature
                    // with the expected argument type, is still compatible with
                    // the rcvr, then it must've not been the source of blame.
                    if self.can_eq(self.param_env, rcvr_ty, expected_ty) {
                        continue;
                    }
                    err.span_label(arg_expr.span, format!("this argument has type `{arg_ty}`..."));
                    err.span_label(
                        binding.span,
                        format!("... which causes `{ident}` to have type `{next_use_ty}`"),
                    );
                    // Using our "ideal" method signature, suggest a fix to this
                    // blame arg, if possible. Don't do this if we're coming from
                    // arg mismatch code, because we'll possibly suggest a mutually
                    // incompatible fix at the original mismatch site.
                    // HACK(compiler-errors): We don't actually consider the implications
                    // of our inference guesses in `emit_type_mismatch_suggestions`, so
                    // only suggest things when we know our type error is precisely due to
                    // a type mismatch, and not via some projection or something. See #116155.
                    if matches!(source, TypeMismatchSource::Ty(_))
                        && let Some(ideal_method) = ideal_method
                        && Some(ideal_method.def_id)
                            == self
                                .typeck_results
                                .borrow()
                                .type_dependent_def_id(parent_expr.hir_id)
                        && let ideal_arg_ty =
                            self.resolve_vars_if_possible(ideal_method.sig.inputs()[idx + 1])
                        && !ideal_arg_ty.has_non_region_infer()
                    {
                        self.emit_type_mismatch_suggestions(
                            err,
                            arg_expr,
                            arg_ty,
                            ideal_arg_ty,
                            None,
                            None,
                        );
                    }
                    return true;
                }
            }
            err.span_label(
                binding.span,
                format!("here the type of `{ident}` is inferred to be `{next_use_ty}`"),
            );
            return true;
        }

        // We must've not found something that constrained the expr.
        false
    }

    // When encountering a type error on the value of a `break`, try to point at the reason for the
    // expected type.
    pub(crate) fn annotate_loop_expected_due_to_inference(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        error: Option<TypeError<'tcx>>,
    ) {
        let Some(TypeError::Sorts(ExpectedFound { expected, .. })) = error else {
            return;
        };
        let mut parent_id = self.tcx.parent_hir_id(expr.hir_id);
        let mut parent;
        'outer: loop {
            // Climb the HIR tree to see if the current `Expr` is part of a `break;` statement.
            let (hir::Node::Stmt(&hir::Stmt { kind: hir::StmtKind::Semi(p), .. })
            | hir::Node::Block(&hir::Block { expr: Some(p), .. })
            | hir::Node::Expr(p)) = self.tcx.hir_node(parent_id)
            else {
                break;
            };
            parent = p;
            parent_id = self.tcx.parent_hir_id(parent_id);
            let hir::ExprKind::Break(destination, _) = parent.kind else {
                continue;
            };
            let mut parent_id = parent_id;
            let mut direct = false;
            loop {
                // Climb the HIR tree to find the (desugared) `loop` this `break` corresponds to.
                let parent = match self.tcx.hir_node(parent_id) {
                    hir::Node::Expr(parent) => {
                        parent_id = self.tcx.parent_hir_id(parent.hir_id);
                        parent
                    }
                    hir::Node::Stmt(hir::Stmt {
                        hir_id,
                        kind: hir::StmtKind::Semi(parent) | hir::StmtKind::Expr(parent),
                        ..
                    }) => {
                        parent_id = self.tcx.parent_hir_id(*hir_id);
                        parent
                    }
                    hir::Node::Block(_) => {
                        parent_id = self.tcx.parent_hir_id(parent_id);
                        parent
                    }
                    _ => break,
                };
                if let hir::ExprKind::Loop(..) = parent.kind {
                    // When you have `'a: loop { break; }`, the `break` corresponds to the labeled
                    // loop, so we need to account for that.
                    direct = !direct;
                }
                if let hir::ExprKind::Loop(block, label, _, span) = parent.kind
                    && (destination.label == label || direct)
                {
                    if let Some((reason_span, message)) =
                        self.maybe_get_coercion_reason(parent_id, parent.span)
                    {
                        err.span_label(reason_span, message);
                        err.span_label(
                            span,
                            format!("this loop is expected to be of type `{expected}`"),
                        );
                        break 'outer;
                    } else {
                        // Locate all other `break` statements within the same `loop` that might
                        // have affected inference.
                        struct FindBreaks<'tcx> {
                            label: Option<rustc_ast::Label>,
                            uses: Vec<&'tcx hir::Expr<'tcx>>,
                            nest_depth: usize,
                        }
                        impl<'tcx> Visitor<'tcx> for FindBreaks<'tcx> {
                            fn visit_expr(&mut self, ex: &'tcx hir::Expr<'tcx>) {
                                let nest_depth = self.nest_depth;
                                if let hir::ExprKind::Loop(_, label, _, _) = ex.kind {
                                    if label == self.label {
                                        // Account for `'a: loop { 'a: loop {...} }`.
                                        return;
                                    }
                                    self.nest_depth += 1;
                                }
                                if let hir::ExprKind::Break(destination, _) = ex.kind
                                    && (self.label == destination.label
                                        // Account for `loop { 'a: loop { loop { break; } } }`.
                                        || destination.label.is_none() && self.nest_depth == 0)
                                {
                                    self.uses.push(ex);
                                }
                                hir::intravisit::walk_expr(self, ex);
                                self.nest_depth = nest_depth;
                            }
                        }
                        let mut expr_finder = FindBreaks { label, uses: vec![], nest_depth: 0 };
                        expr_finder.visit_block(block);
                        let mut exit = false;
                        for ex in expr_finder.uses {
                            let hir::ExprKind::Break(_, val) = ex.kind else {
                                continue;
                            };
                            let ty = match val {
                                Some(val) => {
                                    match self.typeck_results.borrow().expr_ty_adjusted_opt(val) {
                                        None => continue,
                                        Some(ty) => ty,
                                    }
                                }
                                None => self.tcx.types.unit,
                            };
                            if self.can_eq(self.param_env, ty, expected) {
                                err.span_label(ex.span, "expected because of this `break`");
                                exit = true;
                            }
                        }
                        if exit {
                            break 'outer;
                        }
                    }
                }
            }
        }
    }

    fn annotate_expected_due_to_let_ty(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        error: Option<TypeError<'tcx>>,
    ) {
        match (self.tcx.parent_hir_node(expr.hir_id), error) {
            (hir::Node::LetStmt(hir::LetStmt { ty: Some(ty), init: Some(init), .. }), _)
                if init.hir_id == expr.hir_id && !ty.span.source_equal(init.span) =>
            {
                // Point at `let` assignment type.
                err.span_label(ty.span, "expected due to this");
            }
            (
                hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Assign(lhs, rhs, _), .. }),
                Some(TypeError::Sorts(ExpectedFound { expected, .. })),
            ) if rhs.hir_id == expr.hir_id && !expected.is_closure() => {
                // We ignore closures explicitly because we already point at them elsewhere.
                // Point at the assigned-to binding.
                let mut primary_span = lhs.span;
                let mut secondary_span = lhs.span;
                let mut post_message = "";
                match lhs.kind {
                    hir::ExprKind::Path(hir::QPath::Resolved(
                        None,
                        hir::Path {
                            res:
                                hir::def::Res::Def(
                                    hir::def::DefKind::Static { .. } | hir::def::DefKind::Const,
                                    def_id,
                                ),
                            ..
                        },
                    )) => {
                        if let Some(hir::Node::Item(hir::Item {
                            kind:
                                hir::ItemKind::Static(_, ident, ty, _)
                                | hir::ItemKind::Const(ident, _, ty, _),
                            ..
                        })) = self.tcx.hir_get_if_local(*def_id)
                        {
                            primary_span = ty.span;
                            secondary_span = ident.span;
                            post_message = " type";
                        }
                    }
                    hir::ExprKind::Path(hir::QPath::Resolved(
                        None,
                        hir::Path { res: hir::def::Res::Local(hir_id), .. },
                    )) => {
                        if let hir::Node::Pat(pat) = self.tcx.hir_node(*hir_id) {
                            primary_span = pat.span;
                            secondary_span = pat.span;
                            match self.tcx.parent_hir_node(pat.hir_id) {
                                hir::Node::LetStmt(hir::LetStmt { ty: Some(ty), .. }) => {
                                    primary_span = ty.span;
                                    post_message = " type";
                                }
                                hir::Node::LetStmt(hir::LetStmt { init: Some(init), .. }) => {
                                    primary_span = init.span;
                                    post_message = " value";
                                }
                                hir::Node::Param(hir::Param { ty_span, .. }) => {
                                    primary_span = *ty_span;
                                    post_message = " parameter type";
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }

                if primary_span != secondary_span
                    && self
                        .tcx
                        .sess
                        .source_map()
                        .is_multiline(secondary_span.shrink_to_hi().until(primary_span))
                {
                    // We are pointing at the binding's type or initializer value, but it's pattern
                    // is in a different line, so we point at both.
                    err.span_label(secondary_span, "expected due to the type of this binding");
                    err.span_label(primary_span, format!("expected due to this{post_message}"));
                } else if post_message.is_empty() {
                    // We are pointing at either the assignment lhs or the binding def pattern.
                    err.span_label(primary_span, "expected due to the type of this binding");
                } else {
                    // We are pointing at the binding's type or initializer value.
                    err.span_label(primary_span, format!("expected due to this{post_message}"));
                }

                if !lhs.is_syntactic_place_expr() {
                    // We already emitted E0070 "invalid left-hand side of assignment", so we
                    // silence this.
                    err.downgrade_to_delayed_bug();
                }
            }
            (
                hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Binary(_, lhs, rhs), .. }),
                Some(TypeError::Sorts(ExpectedFound { expected, .. })),
            ) if rhs.hir_id == expr.hir_id
                && self.typeck_results.borrow().expr_ty_adjusted_opt(lhs) == Some(expected) =>
            {
                err.span_label(lhs.span, format!("expected because this is `{expected}`"));
            }
            _ => {}
        }
    }

    /// Detect the following case
    ///
    /// ```text
    /// fn change_object(mut b: &Ty) {
    ///     let a = Ty::new();
    ///     b = a;
    /// }
    /// ```
    ///
    /// where the user likely meant to modify the value behind there reference, use `b` as an out
    /// parameter, instead of mutating the local binding. When encountering this we suggest:
    ///
    /// ```text
    /// fn change_object(b: &'_ mut Ty) {
    ///     let a = Ty::new();
    ///     *b = a;
    /// }
    /// ```
    fn annotate_mut_binding_to_immutable_binding(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        expr_ty: Ty<'tcx>,
        expected: Ty<'tcx>,
        error: Option<TypeError<'tcx>>,
    ) -> bool {
        if let Some(TypeError::Sorts(ExpectedFound { .. })) = error
            && let ty::Ref(_, inner, hir::Mutability::Not) = expected.kind()

            // The difference between the expected and found values is one level of borrowing.
            && self.can_eq(self.param_env, *inner, expr_ty)

            // We have an `ident = expr;` assignment.
            && let hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Assign(lhs, rhs, _), .. }) =
                self.tcx.parent_hir_node(expr.hir_id)
            && rhs.hir_id == expr.hir_id

            // We are assigning to some binding.
            && let hir::ExprKind::Path(hir::QPath::Resolved(
                None,
                hir::Path { res: hir::def::Res::Local(hir_id), .. },
            )) = lhs.kind
            && let hir::Node::Pat(pat) = self.tcx.hir_node(*hir_id)

            // The pattern we have is an fn argument.
            && let hir::Node::Param(hir::Param { ty_span, .. }) =
                self.tcx.parent_hir_node(pat.hir_id)
            && let item = self.tcx.hir_get_parent_item(pat.hir_id)
            && let item = self.tcx.hir_owner_node(item)
            && let Some(fn_decl) = item.fn_decl()

            // We have a mutable binding in the argument.
            && let hir::PatKind::Binding(hir::BindingMode::MUT, _hir_id, ident, _) = pat.kind

            // Look for the type corresponding to the argument pattern we have in the argument list.
            && let Some(ty_ref) = fn_decl
                .inputs
                .iter()
                .filter_map(|ty| match ty.kind {
                    hir::TyKind::Ref(lt, mut_ty) if ty.span == *ty_span => Some((lt, mut_ty)),
                    _ => None,
                })
                .next()
        {
            let mut sugg = if ty_ref.1.mutbl.is_mut() {
                // Leave `&'name mut Ty` and `&mut Ty` as they are (#136028).
                vec![]
            } else {
                // `&'name Ty` -> `&'name mut Ty` or `&Ty` -> `&mut Ty`
                vec![(
                    ty_ref.1.ty.span.shrink_to_lo(),
                    format!("{}mut ", if ty_ref.0.ident.span.is_empty() { "" } else { " " },),
                )]
            };
            sugg.extend([
                (pat.span.until(ident.span), String::new()),
                (lhs.span.shrink_to_lo(), "*".to_string()),
            ]);
            // We suggest changing the argument from `mut ident: &Ty` to `ident: &'_ mut Ty` and the
            // assignment from `ident = val;` to `*ident = val;`.
            err.multipart_suggestion_verbose(
                "you might have meant to mutate the pointed at value being passed in, instead of \
                changing the reference in the local binding",
                sugg,
                Applicability::MaybeIncorrect,
            );
            return true;
        }
        false
    }

    fn annotate_alternative_method_deref(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        error: Option<TypeError<'tcx>>,
    ) {
        let Some(TypeError::Sorts(ExpectedFound { expected, .. })) = error else {
            return;
        };
        let hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Assign(lhs, rhs, _), .. }) =
            self.tcx.parent_hir_node(expr.hir_id)
        else {
            return;
        };
        if rhs.hir_id != expr.hir_id || expected.is_closure() {
            return;
        }
        let hir::ExprKind::Unary(hir::UnOp::Deref, deref) = lhs.kind else {
            return;
        };
        let hir::ExprKind::MethodCall(path, base, args, _) = deref.kind else {
            return;
        };
        let Some(self_ty) = self.typeck_results.borrow().expr_ty_adjusted_opt(base) else {
            return;
        };

        let Ok(pick) = self.lookup_probe_for_diagnostic(
            path.ident,
            self_ty,
            deref,
            probe::ProbeScope::TraitsInScope,
            None,
        ) else {
            return;
        };

        let Ok(in_scope_methods) = self.probe_for_name_many(
            probe::Mode::MethodCall,
            path.ident,
            Some(expected),
            probe::IsSuggestion(true),
            self_ty,
            deref.hir_id,
            probe::ProbeScope::TraitsInScope,
        ) else {
            return;
        };

        let other_methods_in_scope: Vec<_> =
            in_scope_methods.iter().filter(|c| c.item.def_id != pick.item.def_id).collect();

        let Ok(all_methods) = self.probe_for_name_many(
            probe::Mode::MethodCall,
            path.ident,
            Some(expected),
            probe::IsSuggestion(true),
            self_ty,
            deref.hir_id,
            probe::ProbeScope::AllTraits,
        ) else {
            return;
        };

        let suggestions: Vec<_> = all_methods
            .into_iter()
            .filter(|c| c.item.def_id != pick.item.def_id)
            .map(|c| {
                let m = c.item;
                let generic_args = ty::GenericArgs::for_item(self.tcx, m.def_id, |param, _| {
                    self.var_for_def(deref.span, param)
                });
                let mutability =
                    match self.tcx.fn_sig(m.def_id).skip_binder().input(0).skip_binder().kind() {
                        ty::Ref(_, _, hir::Mutability::Mut) => "&mut ",
                        ty::Ref(_, _, _) => "&",
                        _ => "",
                    };
                vec![
                    (
                        deref.span.until(base.span),
                        format!(
                            "{}({}",
                            with_no_trimmed_paths!(
                                self.tcx.def_path_str_with_args(m.def_id, generic_args,)
                            ),
                            mutability,
                        ),
                    ),
                    match &args {
                        [] => (base.span.shrink_to_hi().with_hi(deref.span.hi()), ")".to_string()),
                        [first, ..] => (base.span.between(first.span), ", ".to_string()),
                    },
                ]
            })
            .collect();
        if suggestions.is_empty() {
            return;
        }
        let mut path_span: MultiSpan = path.ident.span.into();
        path_span.push_span_label(
            path.ident.span,
            with_no_trimmed_paths!(format!(
                "refers to `{}`",
                self.tcx.def_path_str(pick.item.def_id),
            )),
        );
        let container_id = pick.item.container_id(self.tcx);
        let container = with_no_trimmed_paths!(self.tcx.def_path_str(container_id));
        for def_id in pick.import_ids {
            let hir_id = self.tcx.local_def_id_to_hir_id(def_id);
            path_span
                .push_span_label(self.tcx.hir_span(hir_id), format!("`{container}` imported here"));
        }
        let tail = with_no_trimmed_paths!(match &other_methods_in_scope[..] {
            [] => return,
            [candidate] => format!(
                "the method of the same name on {} `{}`",
                match candidate.kind {
                    probe::CandidateKind::InherentImplCandidate { .. } => "the inherent impl for",
                    _ => "trait",
                },
                self.tcx.def_path_str(candidate.item.container_id(self.tcx))
            ),
            _ if other_methods_in_scope.len() < 5 => {
                format!(
                    "the methods of the same name on {}",
                    listify(
                        &other_methods_in_scope[..other_methods_in_scope.len() - 1],
                        |c| format!("`{}`", self.tcx.def_path_str(c.item.container_id(self.tcx)))
                    )
                    .unwrap_or_default(),
                )
            }
            _ => format!(
                "the methods of the same name on {} other traits",
                other_methods_in_scope.len()
            ),
        });
        err.span_note(
            path_span,
            format!(
                "the `{}` call is resolved to the method in `{container}`, shadowing {tail}",
                path.ident,
            ),
        );
        if suggestions.len() > other_methods_in_scope.len() {
            err.note(format!(
                "additionally, there are {} other available methods that aren't in scope",
                suggestions.len() - other_methods_in_scope.len()
            ));
        }
        err.multipart_suggestions(
            format!(
                "you might have meant to call {}; you can use the fully-qualified path to call {} \
                 explicitly",
                if suggestions.len() == 1 {
                    "the other method"
                } else {
                    "one of the other methods"
                },
                if suggestions.len() == 1 { "it" } else { "one of them" },
            ),
            suggestions,
            Applicability::MaybeIncorrect,
        );
    }

    pub(crate) fn get_conversion_methods_for_diagnostic(
        &self,
        span: Span,
        expected: Ty<'tcx>,
        checked_ty: Ty<'tcx>,
        hir_id: hir::HirId,
    ) -> Vec<AssocItem> {
        let methods = self.probe_for_return_type_for_diagnostic(
            span,
            probe::Mode::MethodCall,
            expected,
            checked_ty,
            hir_id,
            |m| {
                self.has_only_self_parameter(m)
                    && self
                        .tcx
                        // This special internal attribute is used to permit
                        // "identity-like" conversion methods to be suggested here.
                        //
                        // FIXME (#46459 and #46460): ideally
                        // `std::convert::Into::into` and `std::borrow:ToOwned` would
                        // also be `#[rustc_conversion_suggestion]`, if not for
                        // method-probing false-positives and -negatives (respectively).
                        //
                        // FIXME? Other potential candidate methods: `as_ref` and
                        // `as_mut`?
                        .has_attr(m.def_id, sym::rustc_conversion_suggestion)
            },
        );

        methods
    }

    /// This function checks whether the method is not static and does not accept other parameters than `self`.
    fn has_only_self_parameter(&self, method: &AssocItem) -> bool {
        method.is_method()
            && self.tcx.fn_sig(method.def_id).skip_binder().inputs().skip_binder().len() == 1
    }

    /// If the given `HirId` corresponds to a block with a trailing expression, return that expression
    pub(crate) fn maybe_get_block_expr(
        &self,
        expr: &hir::Expr<'tcx>,
    ) -> Option<&'tcx hir::Expr<'tcx>> {
        match expr {
            hir::Expr { kind: hir::ExprKind::Block(block, ..), .. } => block.expr,
            _ => None,
        }
    }

    /// Returns whether the given expression is a destruct assignment desugaring.
    /// For example, `(a, b) = (1, &2);`
    /// Here we try to find the pattern binding of the expression,
    /// `default_binding_modes` is false only for destruct assignment desugaring.
    pub(crate) fn is_destruct_assignment_desugaring(&self, expr: &hir::Expr<'_>) -> bool {
        if let hir::ExprKind::Path(hir::QPath::Resolved(
            _,
            hir::Path { res: hir::def::Res::Local(bind_hir_id), .. },
        )) = expr.kind
            && let bind = self.tcx.hir_node(*bind_hir_id)
            && let parent = self.tcx.parent_hir_node(*bind_hir_id)
            && let hir::Node::Pat(hir::Pat {
                kind: hir::PatKind::Binding(_, _hir_id, _, _), ..
            }) = bind
            && let hir::Node::Pat(hir::Pat { default_binding_modes: false, .. }) = parent
        {
            true
        } else {
            false
        }
    }

    fn explain_self_literal(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) {
        match expr.peel_drop_temps().kind {
            hir::ExprKind::Struct(
                hir::QPath::Resolved(
                    None,
                    hir::Path { res: hir::def::Res::SelfTyAlias { alias_to, .. }, span, .. },
                ),
                ..,
            )
            | hir::ExprKind::Call(
                hir::Expr {
                    kind:
                        hir::ExprKind::Path(hir::QPath::Resolved(
                            None,
                            hir::Path {
                                res: hir::def::Res::SelfTyAlias { alias_to, .. },
                                span,
                                ..
                            },
                        )),
                    ..
                },
                ..,
            ) => {
                if let Some(hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Impl(hir::Impl { self_ty, .. }),
                    ..
                })) = self.tcx.hir_get_if_local(*alias_to)
                {
                    err.span_label(self_ty.span, "this is the type of the `Self` literal");
                }
                if let ty::Adt(e_def, e_args) = expected.kind()
                    && let ty::Adt(f_def, _f_args) = found.kind()
                    && e_def == f_def
                {
                    err.span_suggestion_verbose(
                        *span,
                        "use the type name directly",
                        self.tcx.value_path_str_with_args(e_def.did(), e_args),
                        Applicability::MaybeIncorrect,
                    );
                }
            }
            _ => {}
        }
    }

    fn note_wrong_return_ty_due_to_generic_arg(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        checked_ty: Ty<'tcx>,
    ) {
        let hir::Node::Expr(parent_expr) = self.tcx.parent_hir_node(expr.hir_id) else {
            return;
        };
        enum CallableKind {
            Function,
            Method,
            Constructor,
        }
        let mut maybe_emit_help = |def_id: hir::def_id::DefId,
                                   callable: Ident,
                                   args: &[hir::Expr<'_>],
                                   kind: CallableKind| {
            let arg_idx = args.iter().position(|a| a.hir_id == expr.hir_id).unwrap();
            let fn_ty = self.tcx.type_of(def_id).skip_binder();
            if !fn_ty.is_fn() {
                return;
            }
            let fn_sig = fn_ty.fn_sig(self.tcx).skip_binder();
            let Some(&arg) = fn_sig
                .inputs()
                .get(arg_idx + if matches!(kind, CallableKind::Method) { 1 } else { 0 })
            else {
                return;
            };
            if matches!(arg.kind(), ty::Param(_))
                && fn_sig.output().contains(arg)
                && self.node_ty(args[arg_idx].hir_id) == checked_ty
            {
                let mut multi_span: MultiSpan = parent_expr.span.into();
                multi_span.push_span_label(
                    args[arg_idx].span,
                    format!(
                        "this argument influences the {} of `{}`",
                        if matches!(kind, CallableKind::Constructor) {
                            "type"
                        } else {
                            "return type"
                        },
                        callable
                    ),
                );
                err.span_help(
                    multi_span,
                    format!(
                        "the {} `{}` due to the type of the argument passed",
                        match kind {
                            CallableKind::Function => "return type of this call is",
                            CallableKind::Method => "return type of this call is",
                            CallableKind::Constructor => "type constructed contains",
                        },
                        checked_ty
                    ),
                );
            }
        };
        match parent_expr.kind {
            hir::ExprKind::Call(fun, args) => {
                let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = fun.kind else {
                    return;
                };
                let hir::def::Res::Def(kind, def_id) = path.res else {
                    return;
                };
                let callable_kind = if matches!(kind, hir::def::DefKind::Ctor(_, _)) {
                    CallableKind::Constructor
                } else {
                    CallableKind::Function
                };
                maybe_emit_help(def_id, path.segments.last().unwrap().ident, args, callable_kind);
            }
            hir::ExprKind::MethodCall(method, _receiver, args, _span) => {
                let Some(def_id) =
                    self.typeck_results.borrow().type_dependent_def_id(parent_expr.hir_id)
                else {
                    return;
                };
                maybe_emit_help(def_id, method.ident, args, CallableKind::Method)
            }
            _ => return,
        }
    }
}

pub(crate) enum TypeMismatchSource<'tcx> {
    /// Expected the binding to have the given type, but it was found to have
    /// a different type. Find out when that type first became incompatible.
    Ty(Ty<'tcx>),
    /// When we fail during method argument checking, try to find out if a previous
    /// expression has constrained the method's receiver in a way that makes the
    /// argument's type incompatible.
    Arg { call_expr: &'tcx hir::Expr<'tcx>, incompatible_arg: usize },
}
