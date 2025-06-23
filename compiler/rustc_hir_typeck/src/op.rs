//! Code related to processing overloaded binary and unary operators.

use rustc_data_structures::packed::Pu128;
use rustc_errors::codes::*;
use rustc_errors::{Applicability, Diag, struct_span_code_err};
use rustc_infer::traits::ObligationCauseCode;
use rustc_middle::bug;
use rustc_middle::ty::adjustment::{
    Adjust, Adjustment, AllowTwoPhase, AutoBorrow, AutoBorrowMutability,
};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, IsSuggestable, Ty, TyCtxt, TypeVisitableExt};
use rustc_session::errors::ExprParenthesesNeeded;
use rustc_span::source_map::Spanned;
use rustc_span::{Span, Symbol, sym};
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::{FulfillmentError, Obligation, ObligationCtxt};
use tracing::debug;
use {rustc_ast as ast, rustc_hir as hir};

use super::FnCtxt;
use super::method::MethodCallee;
use crate::Expectation;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Checks a `a <op>= b`
    pub(crate) fn check_expr_assign_op(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        op: hir::AssignOp,
        lhs: &'tcx hir::Expr<'tcx>,
        rhs: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let (lhs_ty, rhs_ty, return_ty) =
            self.check_overloaded_binop(expr, lhs, rhs, Op::AssignOp(op), expected);

        let category = BinOpCategory::from(op.node);
        let ty = if !lhs_ty.is_ty_var()
            && !rhs_ty.is_ty_var()
            && is_builtin_binop(lhs_ty, rhs_ty, category)
        {
            self.enforce_builtin_binop_types(lhs.span, lhs_ty, rhs.span, rhs_ty, category);
            self.tcx.types.unit
        } else {
            return_ty
        };

        self.check_lhs_assignable(lhs, E0067, op.span, |err| {
            if let Some(lhs_deref_ty) = self.deref_once_mutably_for_diagnostic(lhs_ty) {
                if self
                    .lookup_op_method(
                        (lhs, lhs_deref_ty),
                        Some((rhs, rhs_ty)),
                        lang_item_for_binop(self.tcx, Op::AssignOp(op)),
                        op.span,
                        expected,
                    )
                    .is_ok()
                {
                    // If LHS += RHS is an error, but *LHS += RHS is successful, then we will have
                    // emitted a better suggestion during error handling in check_overloaded_binop.
                    if self
                        .lookup_op_method(
                            (lhs, lhs_ty),
                            Some((rhs, rhs_ty)),
                            lang_item_for_binop(self.tcx, Op::AssignOp(op)),
                            op.span,
                            expected,
                        )
                        .is_err()
                    {
                        err.downgrade_to_delayed_bug();
                    } else {
                        // Otherwise, it's valid to suggest dereferencing the LHS here.
                        err.span_suggestion_verbose(
                            lhs.span.shrink_to_lo(),
                            "consider dereferencing the left-hand side of this operation",
                            "*",
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            }
        });

        ty
    }

    /// Checks a potentially overloaded binary operator.
    pub(crate) fn check_expr_binop(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        op: hir::BinOp,
        lhs_expr: &'tcx hir::Expr<'tcx>,
        rhs_expr: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;

        debug!(
            "check_binop(expr.hir_id={}, expr={:?}, op={:?}, lhs_expr={:?}, rhs_expr={:?})",
            expr.hir_id, expr, op, lhs_expr, rhs_expr
        );

        match BinOpCategory::from(op.node) {
            BinOpCategory::Shortcircuit => {
                // && and || are a simple case.
                self.check_expr_coercible_to_type(lhs_expr, tcx.types.bool, None);
                let lhs_diverges = self.diverges.get();
                self.check_expr_coercible_to_type(rhs_expr, tcx.types.bool, None);

                // Depending on the LHS' value, the RHS can never execute.
                self.diverges.set(lhs_diverges);

                tcx.types.bool
            }
            _ => {
                // Otherwise, we always treat operators as if they are
                // overloaded. This is the way to be most flexible w/r/t
                // types that get inferred.
                let (lhs_ty, rhs_ty, return_ty) =
                    self.check_overloaded_binop(expr, lhs_expr, rhs_expr, Op::BinOp(op), expected);

                // Supply type inference hints if relevant. Probably these
                // hints should be enforced during select as part of the
                // `consider_unification_despite_ambiguity` routine, but this
                // more convenient for now.
                //
                // The basic idea is to help type inference by taking
                // advantage of things we know about how the impls for
                // scalar types are arranged. This is important in a
                // scenario like `1_u32 << 2`, because it lets us quickly
                // deduce that the result type should be `u32`, even
                // though we don't know yet what type 2 has and hence
                // can't pin this down to a specific impl.
                let category = BinOpCategory::from(op.node);
                if !lhs_ty.is_ty_var()
                    && !rhs_ty.is_ty_var()
                    && is_builtin_binop(lhs_ty, rhs_ty, category)
                {
                    let builtin_return_ty = self.enforce_builtin_binop_types(
                        lhs_expr.span,
                        lhs_ty,
                        rhs_expr.span,
                        rhs_ty,
                        category,
                    );
                    self.demand_eqtype(expr.span, builtin_return_ty, return_ty);
                    builtin_return_ty
                } else {
                    return_ty
                }
            }
        }
    }

    fn enforce_builtin_binop_types(
        &self,
        lhs_span: Span,
        lhs_ty: Ty<'tcx>,
        rhs_span: Span,
        rhs_ty: Ty<'tcx>,
        category: BinOpCategory,
    ) -> Ty<'tcx> {
        debug_assert!(is_builtin_binop(lhs_ty, rhs_ty, category));

        // Special-case a single layer of referencing, so that things like `5.0 + &6.0f32` work.
        // (See https://github.com/rust-lang/rust/issues/57447.)
        let (lhs_ty, rhs_ty) = (deref_ty_if_possible(lhs_ty), deref_ty_if_possible(rhs_ty));

        let tcx = self.tcx;
        match category {
            BinOpCategory::Shortcircuit => {
                self.demand_suptype(lhs_span, tcx.types.bool, lhs_ty);
                self.demand_suptype(rhs_span, tcx.types.bool, rhs_ty);
                tcx.types.bool
            }

            BinOpCategory::Shift => {
                // result type is same as LHS always
                lhs_ty
            }

            BinOpCategory::Math | BinOpCategory::Bitwise => {
                // both LHS and RHS and result will have the same type
                self.demand_suptype(rhs_span, lhs_ty, rhs_ty);
                lhs_ty
            }

            BinOpCategory::Comparison => {
                // both LHS and RHS and result will have the same type
                self.demand_suptype(rhs_span, lhs_ty, rhs_ty);
                tcx.types.bool
            }
        }
    }

    fn check_overloaded_binop(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        lhs_expr: &'tcx hir::Expr<'tcx>,
        rhs_expr: &'tcx hir::Expr<'tcx>,
        op: Op,
        expected: Expectation<'tcx>,
    ) -> (Ty<'tcx>, Ty<'tcx>, Ty<'tcx>) {
        debug!("check_overloaded_binop(expr.hir_id={}, op={:?})", expr.hir_id, op);

        let lhs_ty = match op {
            Op::BinOp(_) => {
                // Find a suitable supertype of the LHS expression's type, by coercing to
                // a type variable, to pass as the `Self` to the trait, avoiding invariant
                // trait matching creating lifetime constraints that are too strict.
                // e.g., adding `&'a T` and `&'b T`, given `&'x T: Add<&'x T>`, will result
                // in `&'a T <: &'x T` and `&'b T <: &'x T`, instead of `'a = 'b = 'x`.
                let lhs_ty = self.check_expr(lhs_expr);
                let fresh_var = self.next_ty_var(lhs_expr.span);
                self.demand_coerce(lhs_expr, lhs_ty, fresh_var, Some(rhs_expr), AllowTwoPhase::No)
            }
            Op::AssignOp(_) => {
                // rust-lang/rust#52126: We have to use strict
                // equivalence on the LHS of an assign-op like `+=`;
                // overwritten or mutably-borrowed places cannot be
                // coerced to a supertype.
                self.check_expr(lhs_expr)
            }
        };
        let lhs_ty = self.resolve_vars_with_obligations(lhs_ty);

        // N.B., as we have not yet type-checked the RHS, we don't have the
        // type at hand. Make a variable to represent it. The whole reason
        // for this indirection is so that, below, we can check the expr
        // using this variable as the expected type, which sometimes lets
        // us do better coercions than we would be able to do otherwise,
        // particularly for things like `String + &String`.
        let rhs_ty_var = self.next_ty_var(rhs_expr.span);
        let result = self.lookup_op_method(
            (lhs_expr, lhs_ty),
            Some((rhs_expr, rhs_ty_var)),
            lang_item_for_binop(self.tcx, op),
            op.span(),
            expected,
        );

        // see `NB` above
        let rhs_ty = self.check_expr_coercible_to_type_or_error(
            rhs_expr,
            rhs_ty_var,
            Some(lhs_expr),
            |err, ty| {
                if let Op::BinOp(binop) = op
                    && binop.node == hir::BinOpKind::Eq
                {
                    self.suggest_swapping_lhs_and_rhs(err, ty, lhs_ty, rhs_expr, lhs_expr);
                }
            },
        );
        let rhs_ty = self.resolve_vars_with_obligations(rhs_ty);

        let return_ty = match result {
            Ok(method) => {
                let by_ref_binop = !op.is_by_value();
                if matches!(op, Op::AssignOp(_)) || by_ref_binop {
                    if let ty::Ref(_, _, mutbl) = method.sig.inputs()[0].kind() {
                        let mutbl = AutoBorrowMutability::new(*mutbl, AllowTwoPhase::Yes);
                        let autoref = Adjustment {
                            kind: Adjust::Borrow(AutoBorrow::Ref(mutbl)),
                            target: method.sig.inputs()[0],
                        };
                        self.apply_adjustments(lhs_expr, vec![autoref]);
                    }
                }
                if by_ref_binop {
                    if let ty::Ref(_, _, mutbl) = method.sig.inputs()[1].kind() {
                        // Allow two-phase borrows for binops in initial deployment
                        // since they desugar to methods
                        let mutbl = AutoBorrowMutability::new(*mutbl, AllowTwoPhase::Yes);

                        let autoref = Adjustment {
                            kind: Adjust::Borrow(AutoBorrow::Ref(mutbl)),
                            target: method.sig.inputs()[1],
                        };
                        // HACK(eddyb) Bypass checks due to reborrows being in
                        // some cases applied on the RHS, on top of which we need
                        // to autoref, which is not allowed by apply_adjustments.
                        // self.apply_adjustments(rhs_expr, vec![autoref]);
                        self.typeck_results
                            .borrow_mut()
                            .adjustments_mut()
                            .entry(rhs_expr.hir_id)
                            .or_default()
                            .push(autoref);
                    }
                }
                self.write_method_call_and_enforce_effects(expr.hir_id, expr.span, method);

                method.sig.output()
            }
            // error types are considered "builtin"
            Err(_) if lhs_ty.references_error() || rhs_ty.references_error() => {
                Ty::new_misc_error(self.tcx)
            }
            Err(errors) => {
                let (_, trait_def_id) = lang_item_for_binop(self.tcx, op);
                let missing_trait = trait_def_id
                    .map(|def_id| with_no_trimmed_paths!(self.tcx.def_path_str(def_id)));
                let mut path = None;
                let lhs_ty_str = self.tcx.short_string(lhs_ty, &mut path);
                let rhs_ty_str = self.tcx.short_string(rhs_ty, &mut path);
                let (mut err, output_def_id) = match op {
                    Op::AssignOp(assign_op) => {
                        let s = assign_op.node.as_str();
                        let mut err = struct_span_code_err!(
                            self.dcx(),
                            expr.span,
                            E0368,
                            "binary assignment operation `{}` cannot be applied to type `{}`",
                            s,
                            lhs_ty_str,
                        );
                        err.span_label(
                            lhs_expr.span,
                            format!("cannot use `{}` on type `{}`", s, lhs_ty_str),
                        );
                        self.note_unmet_impls_on_type(&mut err, errors, false);
                        (err, None)
                    }
                    Op::BinOp(bin_op) => {
                        let message = match bin_op.node {
                            hir::BinOpKind::Add => {
                                format!("cannot add `{rhs_ty_str}` to `{lhs_ty_str}`")
                            }
                            hir::BinOpKind::Sub => {
                                format!("cannot subtract `{rhs_ty_str}` from `{lhs_ty_str}`")
                            }
                            hir::BinOpKind::Mul => {
                                format!("cannot multiply `{lhs_ty_str}` by `{rhs_ty_str}`")
                            }
                            hir::BinOpKind::Div => {
                                format!("cannot divide `{lhs_ty_str}` by `{rhs_ty_str}`")
                            }
                            hir::BinOpKind::Rem => {
                                format!(
                                    "cannot calculate the remainder of `{lhs_ty_str}` divided by \
                                     `{rhs_ty_str}`"
                                )
                            }
                            hir::BinOpKind::BitAnd => {
                                format!("no implementation for `{lhs_ty_str} & {rhs_ty_str}`")
                            }
                            hir::BinOpKind::BitXor => {
                                format!("no implementation for `{lhs_ty_str} ^ {rhs_ty_str}`")
                            }
                            hir::BinOpKind::BitOr => {
                                format!("no implementation for `{lhs_ty_str} | {rhs_ty_str}`")
                            }
                            hir::BinOpKind::Shl => {
                                format!("no implementation for `{lhs_ty_str} << {rhs_ty_str}`")
                            }
                            hir::BinOpKind::Shr => {
                                format!("no implementation for `{lhs_ty_str} >> {rhs_ty_str}`")
                            }
                            _ => format!(
                                "binary operation `{}` cannot be applied to type `{}`",
                                bin_op.node.as_str(),
                                lhs_ty_str
                            ),
                        };
                        let output_def_id = trait_def_id.and_then(|def_id| {
                            self.tcx
                                .associated_item_def_ids(def_id)
                                .iter()
                                .find(|item_def_id| {
                                    self.tcx.associated_item(*item_def_id).name() == sym::Output
                                })
                                .cloned()
                        });
                        let mut err =
                            struct_span_code_err!(self.dcx(), bin_op.span, E0369, "{message}");
                        if !lhs_expr.span.eq(&rhs_expr.span) {
                            err.span_label(lhs_expr.span, lhs_ty_str.clone());
                            err.span_label(rhs_expr.span, rhs_ty_str);
                        }
                        let suggest_derive = self.can_eq(self.param_env, lhs_ty, rhs_ty);
                        self.note_unmet_impls_on_type(&mut err, errors, suggest_derive);
                        (err, output_def_id)
                    }
                };
                *err.long_ty_path() = path;

                // Try to suggest a semicolon if it's `A \n *B` where `B` is a place expr
                let maybe_missing_semi = self.check_for_missing_semi(expr, &mut err);

                // We defer to the later error produced by `check_lhs_assignable`.
                // We only downgrade this if it's the LHS, though, and if this is a
                // valid assignment statement.
                if maybe_missing_semi
                    && let hir::Node::Expr(parent) = self.tcx.parent_hir_node(expr.hir_id)
                    && let hir::ExprKind::Assign(lhs, _, _) = parent.kind
                    && let hir::Node::Stmt(stmt) = self.tcx.parent_hir_node(parent.hir_id)
                    && let hir::StmtKind::Expr(_) | hir::StmtKind::Semi(_) = stmt.kind
                    && lhs.hir_id == expr.hir_id
                {
                    err.downgrade_to_delayed_bug();
                }

                let suggest_deref_binop = |err: &mut Diag<'_, _>, lhs_deref_ty: Ty<'tcx>| {
                    if self
                        .lookup_op_method(
                            (lhs_expr, lhs_deref_ty),
                            Some((rhs_expr, rhs_ty)),
                            lang_item_for_binop(self.tcx, op),
                            op.span(),
                            expected,
                        )
                        .is_ok()
                    {
                        let msg = format!(
                            "`{}` can be used on `{}` if you dereference the left-hand side",
                            op.as_str(),
                            self.tcx.short_string(lhs_deref_ty, err.long_ty_path()),
                        );
                        err.span_suggestion_verbose(
                            lhs_expr.span.shrink_to_lo(),
                            msg,
                            "*",
                            rustc_errors::Applicability::MachineApplicable,
                        );
                    }
                };

                let suggest_different_borrow =
                    |err: &mut Diag<'_, _>,
                     lhs_adjusted_ty,
                     lhs_new_mutbl: Option<ast::Mutability>,
                     rhs_adjusted_ty,
                     rhs_new_mutbl: Option<ast::Mutability>| {
                        if self
                            .lookup_op_method(
                                (lhs_expr, lhs_adjusted_ty),
                                Some((rhs_expr, rhs_adjusted_ty)),
                                lang_item_for_binop(self.tcx, op),
                                op.span(),
                                expected,
                            )
                            .is_ok()
                        {
                            let lhs = self.tcx.short_string(lhs_adjusted_ty, err.long_ty_path());
                            let rhs = self.tcx.short_string(rhs_adjusted_ty, err.long_ty_path());
                            let op = op.as_str();
                            err.note(format!("an implementation for `{lhs} {op} {rhs}` exists"));

                            if let Some(lhs_new_mutbl) = lhs_new_mutbl
                                && let Some(rhs_new_mutbl) = rhs_new_mutbl
                                && lhs_new_mutbl.is_not()
                                && rhs_new_mutbl.is_not()
                            {
                                err.multipart_suggestion_verbose(
                                    "consider reborrowing both sides",
                                    vec![
                                        (lhs_expr.span.shrink_to_lo(), "&*".to_string()),
                                        (rhs_expr.span.shrink_to_lo(), "&*".to_string()),
                                    ],
                                    rustc_errors::Applicability::MachineApplicable,
                                );
                            } else {
                                let mut suggest_new_borrow =
                                    |new_mutbl: ast::Mutability, sp: Span| {
                                        // Can reborrow (&mut -> &)
                                        if new_mutbl.is_not() {
                                            err.span_suggestion_verbose(
                                                sp.shrink_to_lo(),
                                                "consider reborrowing this side",
                                                "&*",
                                                rustc_errors::Applicability::MachineApplicable,
                                            );
                                        // Works on &mut but have &
                                        } else {
                                            err.span_help(
                                                sp,
                                                "consider making this expression a mutable borrow",
                                            );
                                        }
                                    };

                                if let Some(lhs_new_mutbl) = lhs_new_mutbl {
                                    suggest_new_borrow(lhs_new_mutbl, lhs_expr.span);
                                }
                                if let Some(rhs_new_mutbl) = rhs_new_mutbl {
                                    suggest_new_borrow(rhs_new_mutbl, rhs_expr.span);
                                }
                            }
                        }
                    };

                let is_compatible_after_call = |lhs_ty, rhs_ty| {
                    self.lookup_op_method(
                        (lhs_expr, lhs_ty),
                        Some((rhs_expr, rhs_ty)),
                        lang_item_for_binop(self.tcx, op),
                        op.span(),
                        expected,
                    )
                    .is_ok()
                        // Suggest calling even if, after calling, the types don't
                        // implement the operator, since it'll lead to better
                        // diagnostics later.
                        || self.can_eq(self.param_env, lhs_ty, rhs_ty)
                };

                // We should suggest `a + b` => `*a + b` if `a` is copy, and suggest
                // `a += b` => `*a += b` if a is a mut ref.
                if !op.span().can_be_used_for_suggestions() {
                    // Suppress suggestions when lhs and rhs are not in the same span as the error
                } else if let Op::AssignOp(_) = op
                    && let Some(lhs_deref_ty) = self.deref_once_mutably_for_diagnostic(lhs_ty)
                {
                    suggest_deref_binop(&mut err, lhs_deref_ty);
                } else if let Op::BinOp(_) = op
                    && let ty::Ref(region, lhs_deref_ty, mutbl) = lhs_ty.kind()
                {
                    if self.type_is_copy_modulo_regions(self.param_env, *lhs_deref_ty) {
                        suggest_deref_binop(&mut err, *lhs_deref_ty);
                    } else {
                        let lhs_inv_mutbl = mutbl.invert();
                        let lhs_inv_mutbl_ty =
                            Ty::new_ref(self.tcx, *region, *lhs_deref_ty, lhs_inv_mutbl);

                        suggest_different_borrow(
                            &mut err,
                            lhs_inv_mutbl_ty,
                            Some(lhs_inv_mutbl),
                            rhs_ty,
                            None,
                        );

                        if let ty::Ref(region, rhs_deref_ty, mutbl) = rhs_ty.kind() {
                            let rhs_inv_mutbl = mutbl.invert();
                            let rhs_inv_mutbl_ty =
                                Ty::new_ref(self.tcx, *region, *rhs_deref_ty, rhs_inv_mutbl);

                            suggest_different_borrow(
                                &mut err,
                                lhs_ty,
                                None,
                                rhs_inv_mutbl_ty,
                                Some(rhs_inv_mutbl),
                            );
                            suggest_different_borrow(
                                &mut err,
                                lhs_inv_mutbl_ty,
                                Some(lhs_inv_mutbl),
                                rhs_inv_mutbl_ty,
                                Some(rhs_inv_mutbl),
                            );
                        }
                    }
                } else if self.suggest_fn_call(&mut err, lhs_expr, lhs_ty, |lhs_ty| {
                    is_compatible_after_call(lhs_ty, rhs_ty)
                }) || self.suggest_fn_call(&mut err, rhs_expr, rhs_ty, |rhs_ty| {
                    is_compatible_after_call(lhs_ty, rhs_ty)
                }) || self.suggest_two_fn_call(
                    &mut err,
                    rhs_expr,
                    rhs_ty,
                    lhs_expr,
                    lhs_ty,
                    |lhs_ty, rhs_ty| is_compatible_after_call(lhs_ty, rhs_ty),
                ) {
                    // Cool
                }

                if let Some(missing_trait) = missing_trait {
                    if matches!(
                        op,
                        Op::BinOp(Spanned { node: hir::BinOpKind::Add, .. })
                            | Op::AssignOp(Spanned { node: hir::AssignOpKind::AddAssign, .. })
                    ) && self
                        .check_str_addition(lhs_expr, rhs_expr, lhs_ty, rhs_ty, &mut err, op)
                    {
                        // This has nothing here because it means we did string
                        // concatenation (e.g., "Hello " + "World!"). This means
                        // we don't want the note in the else clause to be emitted
                    } else if lhs_ty.has_non_region_param() {
                        // Look for a TraitPredicate in the Fulfillment errors,
                        // and use it to generate a suggestion.
                        //
                        // Note that lookup_op_method must be called again but
                        // with a specific rhs_ty instead of a placeholder so
                        // the resulting predicate generates a more specific
                        // suggestion for the user.
                        let errors = self
                            .lookup_op_method(
                                (lhs_expr, lhs_ty),
                                Some((rhs_expr, rhs_ty)),
                                lang_item_for_binop(self.tcx, op),
                                op.span(),
                                expected,
                            )
                            .unwrap_err();
                        if !errors.is_empty() {
                            for error in errors {
                                if let Some(trait_pred) =
                                    error.obligation.predicate.as_trait_clause()
                                {
                                    let output_associated_item = match error.obligation.cause.code()
                                    {
                                        ObligationCauseCode::BinOp {
                                            output_ty: Some(output_ty),
                                            ..
                                        } => {
                                            // Make sure that we're attaching `Output = ..` to the right trait predicate
                                            if let Some(output_def_id) = output_def_id
                                                && let Some(trait_def_id) = trait_def_id
                                                && self.tcx.parent(output_def_id) == trait_def_id
                                                && let Some(output_ty) = output_ty
                                                    .make_suggestable(self.tcx, false, None)
                                            {
                                                Some(("Output", output_ty))
                                            } else {
                                                None
                                            }
                                        }
                                        _ => None,
                                    };

                                    self.err_ctxt().suggest_restricting_param_bound(
                                        &mut err,
                                        trait_pred,
                                        output_associated_item,
                                        self.body_id,
                                    );
                                }
                            }
                        } else {
                            // When we know that a missing bound is responsible, we don't show
                            // this note as it is redundant.
                            err.note(format!(
                                "the trait `{missing_trait}` is not implemented for `{lhs_ty_str}`"
                            ));
                        }
                    }
                }

                // Suggest using `add`, `offset` or `offset_from` for pointer - {integer},
                // pointer + {integer} or pointer - pointer.
                if op.span().can_be_used_for_suggestions() {
                    match op {
                        Op::BinOp(Spanned { node: hir::BinOpKind::Add, .. })
                            if lhs_ty.is_raw_ptr() && rhs_ty.is_integral() =>
                        {
                            err.multipart_suggestion(
                                "consider using `wrapping_add` or `add` for pointer + {integer}",
                                vec![
                                    (
                                        lhs_expr.span.between(rhs_expr.span),
                                        ".wrapping_add(".to_owned(),
                                    ),
                                    (rhs_expr.span.shrink_to_hi(), ")".to_owned()),
                                ],
                                Applicability::MaybeIncorrect,
                            );
                        }
                        Op::BinOp(Spanned { node: hir::BinOpKind::Sub, .. }) => {
                            if lhs_ty.is_raw_ptr() && rhs_ty.is_integral() {
                                err.multipart_suggestion(
                                    "consider using `wrapping_sub` or `sub` for \
                                     pointer - {integer}",
                                    vec![
                                        (
                                            lhs_expr.span.between(rhs_expr.span),
                                            ".wrapping_sub(".to_owned(),
                                        ),
                                        (rhs_expr.span.shrink_to_hi(), ")".to_owned()),
                                    ],
                                    Applicability::MaybeIncorrect,
                                );
                            }

                            if lhs_ty.is_raw_ptr() && rhs_ty.is_raw_ptr() {
                                err.multipart_suggestion(
                                    "consider using `offset_from` for pointer - pointer if the \
                                     pointers point to the same allocation",
                                    vec![
                                        (lhs_expr.span.shrink_to_lo(), "unsafe { ".to_owned()),
                                        (
                                            lhs_expr.span.between(rhs_expr.span),
                                            ".offset_from(".to_owned(),
                                        ),
                                        (rhs_expr.span.shrink_to_hi(), ") }".to_owned()),
                                    ],
                                    Applicability::MaybeIncorrect,
                                );
                            }
                        }
                        _ => {}
                    }
                }

                let lhs_name_str = match lhs_expr.kind {
                    hir::ExprKind::Path(hir::QPath::Resolved(_, path)) => {
                        path.segments.last().map_or("_".to_string(), |s| s.ident.to_string())
                    }
                    _ => self
                        .tcx
                        .sess
                        .source_map()
                        .span_to_snippet(lhs_expr.span)
                        .unwrap_or_else(|_| "_".to_string()),
                };

                if op.span().can_be_used_for_suggestions() {
                    match op {
                        Op::AssignOp(Spanned { node: hir::AssignOpKind::AddAssign, .. })
                            if lhs_ty.is_raw_ptr() && rhs_ty.is_integral() =>
                        {
                            err.multipart_suggestion(
                                "consider using `add` or `wrapping_add` to do pointer arithmetic",
                                vec![
                                    (lhs_expr.span.shrink_to_lo(), format!("{} = ", lhs_name_str)),
                                    (
                                        lhs_expr.span.between(rhs_expr.span),
                                        ".wrapping_add(".to_owned(),
                                    ),
                                    (rhs_expr.span.shrink_to_hi(), ")".to_owned()),
                                ],
                                Applicability::MaybeIncorrect,
                            );
                        }
                        Op::AssignOp(Spanned { node: hir::AssignOpKind::SubAssign, .. }) => {
                            if lhs_ty.is_raw_ptr() && rhs_ty.is_integral() {
                                err.multipart_suggestion(
                                    "consider using `sub` or `wrapping_sub` to do pointer arithmetic",
                                    vec![
                                        (lhs_expr.span.shrink_to_lo(), format!("{} = ", lhs_name_str)),
                                        (
                                            lhs_expr.span.between(rhs_expr.span),
                                            ".wrapping_sub(".to_owned(),

                                        ),
                                        (rhs_expr.span.shrink_to_hi(), ")".to_owned()),
                                    ],
                                    Applicability::MaybeIncorrect,
                                );
                            }
                        }
                        _ => {}
                    }
                }

                let reported = err.emit();
                Ty::new_error(self.tcx, reported)
            }
        };

        (lhs_ty, rhs_ty, return_ty)
    }

    /// Provide actionable suggestions when trying to add two strings with incorrect types,
    /// like `&str + &str`, `String + String` and `&str + &String`.
    ///
    /// If this function returns `true` it means a note was printed, so we don't need
    /// to print the normal "implementation of `std::ops::Add` might be missing" note
    fn check_str_addition(
        &self,
        lhs_expr: &'tcx hir::Expr<'tcx>,
        rhs_expr: &'tcx hir::Expr<'tcx>,
        lhs_ty: Ty<'tcx>,
        rhs_ty: Ty<'tcx>,
        err: &mut Diag<'_>,
        op: Op,
    ) -> bool {
        let str_concat_note = "string concatenation requires an owned `String` on the left";
        let rm_borrow_msg = "remove the borrow to obtain an owned `String`";
        let to_owned_msg = "create an owned `String` from a string reference";

        let string_type = self.tcx.lang_items().string();
        let is_std_string =
            |ty: Ty<'tcx>| ty.ty_adt_def().is_some_and(|ty_def| Some(ty_def.did()) == string_type);

        match (lhs_ty.kind(), rhs_ty.kind()) {
            (&ty::Ref(_, l_ty, _), &ty::Ref(_, r_ty, _)) // &str or &String + &str, &String or &&str
                if (*l_ty.kind() == ty::Str || is_std_string(l_ty))
                    && (*r_ty.kind() == ty::Str
                        || is_std_string(r_ty)
                        || matches!(
                            r_ty.kind(), ty::Ref(_, inner_ty, _) if *inner_ty.kind() == ty::Str
                        )) =>
            {
                if let Op::BinOp(_) = op { // Do not supply this message if `&str += &str`
                    err.span_label(
                        op.span(),
                        "`+` cannot be used to concatenate two `&str` strings"
                    );
                    err.note(str_concat_note);
                    if let hir::ExprKind::AddrOf(_, _, lhs_inner_expr) = lhs_expr.kind {
                        err.span_suggestion_verbose(
                            lhs_expr.span.until(lhs_inner_expr.span),
                            rm_borrow_msg,
                            "",
                            Applicability::MachineApplicable
                        );
                    } else {
                        err.span_suggestion_verbose(
                            lhs_expr.span.shrink_to_hi(),
                            to_owned_msg,
                            ".to_owned()",
                            Applicability::MachineApplicable
                        );
                    }
                }
                true
            }
            (&ty::Ref(_, l_ty, _), &ty::Adt(..)) // Handle `&str` & `&String` + `String`
                if (*l_ty.kind() == ty::Str || is_std_string(l_ty)) && is_std_string(rhs_ty) =>
            {
                err.span_label(
                    op.span(),
                    "`+` cannot be used to concatenate a `&str` with a `String`",
                );
                match op {
                    Op::BinOp(_) => {
                        let sugg_msg;
                        let lhs_sugg = if let hir::ExprKind::AddrOf(_, _, lhs_inner_expr) = lhs_expr.kind {
                            sugg_msg = "remove the borrow on the left and add one on the right";
                            (lhs_expr.span.until(lhs_inner_expr.span), "".to_owned())
                        } else {
                            sugg_msg = "create an owned `String` on the left and add a borrow on the right";
                            (lhs_expr.span.shrink_to_hi(), ".to_owned()".to_owned())
                        };
                        let suggestions = vec![
                            lhs_sugg,
                            (rhs_expr.span.shrink_to_lo(), "&".to_owned()),
                        ];
                        err.multipart_suggestion_verbose(
                            sugg_msg,
                            suggestions,
                            Applicability::MachineApplicable,
                        );
                    }
                    Op::AssignOp(_) => {
                        err.note(str_concat_note);
                    }
                }
                true
            }
            _ => false,
        }
    }

    pub(crate) fn check_user_unop(
        &self,
        ex: &'tcx hir::Expr<'tcx>,
        operand_ty: Ty<'tcx>,
        op: hir::UnOp,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        assert!(op.is_by_value());
        match self.lookup_op_method(
            (ex, operand_ty),
            None,
            lang_item_for_unop(self.tcx, op),
            ex.span,
            expected,
        ) {
            Ok(method) => {
                self.write_method_call_and_enforce_effects(ex.hir_id, ex.span, method);
                method.sig.output()
            }
            Err(errors) => {
                let actual = self.resolve_vars_if_possible(operand_ty);
                let guar = actual.error_reported().err().unwrap_or_else(|| {
                    let mut file = None;
                    let ty_str = self.tcx.short_string(actual, &mut file);
                    let mut err = struct_span_code_err!(
                        self.dcx(),
                        ex.span,
                        E0600,
                        "cannot apply unary operator `{}` to type `{ty_str}`",
                        op.as_str(),
                    );
                    *err.long_ty_path() = file;
                    err.span_label(
                        ex.span,
                        format!("cannot apply unary operator `{}`", op.as_str()),
                    );

                    if operand_ty.has_non_region_param() {
                        let predicates = errors
                            .iter()
                            .filter_map(|error| error.obligation.predicate.as_trait_clause());
                        for pred in predicates {
                            self.err_ctxt().suggest_restricting_param_bound(
                                &mut err,
                                pred,
                                None,
                                self.body_id,
                            );
                        }
                    }

                    let sp = self.tcx.sess.source_map().start_point(ex.span).with_parent(None);
                    if let Some(sp) =
                        self.tcx.sess.psess.ambiguous_block_expr_parse.borrow().get(&sp)
                    {
                        // If the previous expression was a block expression, suggest parentheses
                        // (turning this into a binary subtraction operation instead.)
                        // for example, `{2} - 2` -> `({2}) - 2` (see src\test\ui\parser\expr-as-stmt.rs)
                        err.subdiagnostic(ExprParenthesesNeeded::surrounding(*sp));
                    } else {
                        match actual.kind() {
                            ty::Uint(_) if op == hir::UnOp::Neg => {
                                err.note("unsigned values cannot be negated");

                                if let hir::ExprKind::Unary(
                                    _,
                                    hir::Expr {
                                        kind:
                                            hir::ExprKind::Lit(Spanned {
                                                node: ast::LitKind::Int(Pu128(1), _),
                                                ..
                                            }),
                                        ..
                                    },
                                ) = ex.kind
                                {
                                    let span = if let hir::Node::Expr(parent) =
                                        self.tcx.parent_hir_node(ex.hir_id)
                                        && let hir::ExprKind::Cast(..) = parent.kind
                                    {
                                        // `-1 as usize` -> `usize::MAX`
                                        parent.span
                                    } else {
                                        ex.span
                                    };
                                    err.span_suggestion_verbose(
                                        span,
                                        format!(
                                            "you may have meant the maximum value of `{actual}`",
                                        ),
                                        format!("{actual}::MAX"),
                                        Applicability::MaybeIncorrect,
                                    );
                                }
                            }
                            ty::Str | ty::Never | ty::Char | ty::Tuple(_) | ty::Array(_, _) => {}
                            ty::Ref(_, lty, _) if *lty.kind() == ty::Str => {}
                            _ => {
                                self.note_unmet_impls_on_type(&mut err, errors, true);
                            }
                        }
                    }
                    err.emit()
                });
                Ty::new_error(self.tcx, guar)
            }
        }
    }

    fn lookup_op_method(
        &self,
        (lhs_expr, lhs_ty): (&'tcx hir::Expr<'tcx>, Ty<'tcx>),
        opt_rhs: Option<(&'tcx hir::Expr<'tcx>, Ty<'tcx>)>,
        (opname, trait_did): (Symbol, Option<hir::def_id::DefId>),
        span: Span,
        expected: Expectation<'tcx>,
    ) -> Result<MethodCallee<'tcx>, Vec<FulfillmentError<'tcx>>> {
        let Some(trait_did) = trait_did else {
            // Bail if the operator trait is not defined.
            return Err(vec![]);
        };

        debug!(
            "lookup_op_method(lhs_ty={:?}, opname={:?}, trait_did={:?})",
            lhs_ty, opname, trait_did
        );

        let (opt_rhs_expr, opt_rhs_ty) = opt_rhs.unzip();
        let cause = self.cause(
            span,
            ObligationCauseCode::BinOp {
                lhs_hir_id: lhs_expr.hir_id,
                rhs_hir_id: opt_rhs_expr.map(|expr| expr.hir_id),
                rhs_span: opt_rhs_expr.map(|expr| expr.span),
                rhs_is_lit: opt_rhs_expr
                    .is_some_and(|expr| matches!(expr.kind, hir::ExprKind::Lit(_))),
                output_ty: expected.only_has_type(self),
            },
        );

        let method =
            self.lookup_method_for_operator(cause.clone(), opname, trait_did, lhs_ty, opt_rhs_ty);
        match method {
            Some(ok) => {
                let method = self.register_infer_ok_obligations(ok);
                self.select_obligations_where_possible(|_| {});
                Ok(method)
            }
            None => {
                // This path may do some inference, so make sure we've really
                // doomed compilation so as to not accidentally stabilize new
                // inference or something here...
                self.dcx().span_delayed_bug(span, "this path really should be doomed...");
                // Guide inference for the RHS expression if it's provided --
                // this will allow us to better error reporting, at the expense
                // of making some error messages a bit more specific.
                if let Some((rhs_expr, rhs_ty)) = opt_rhs
                    && rhs_ty.is_ty_var()
                {
                    self.check_expr_coercible_to_type(rhs_expr, rhs_ty, None);
                }

                // Construct an obligation `self_ty : Trait<input_tys>`
                let args =
                    ty::GenericArgs::for_item(self.tcx, trait_did, |param, _| match param.kind {
                        ty::GenericParamDefKind::Lifetime
                        | ty::GenericParamDefKind::Const { .. } => {
                            unreachable!("did not expect operand trait to have lifetime/const args")
                        }
                        ty::GenericParamDefKind::Type { .. } => {
                            if param.index == 0 {
                                lhs_ty.into()
                            } else {
                                opt_rhs_ty.expect("expected RHS for binop").into()
                            }
                        }
                    });
                let obligation = Obligation::new(
                    self.tcx,
                    cause,
                    self.param_env,
                    ty::TraitRef::new_from_args(self.tcx, trait_did, args),
                );
                let ocx = ObligationCtxt::new_with_diagnostics(&self.infcx);
                ocx.register_obligation(obligation);
                Err(ocx.select_all_or_error())
            }
        }
    }
}

fn lang_item_for_binop(tcx: TyCtxt<'_>, op: Op) -> (Symbol, Option<hir::def_id::DefId>) {
    let lang = tcx.lang_items();
    match op {
        Op::AssignOp(op) => match op.node {
            hir::AssignOpKind::AddAssign => (sym::add_assign, lang.add_assign_trait()),
            hir::AssignOpKind::SubAssign => (sym::sub_assign, lang.sub_assign_trait()),
            hir::AssignOpKind::MulAssign => (sym::mul_assign, lang.mul_assign_trait()),
            hir::AssignOpKind::DivAssign => (sym::div_assign, lang.div_assign_trait()),
            hir::AssignOpKind::RemAssign => (sym::rem_assign, lang.rem_assign_trait()),
            hir::AssignOpKind::BitXorAssign => (sym::bitxor_assign, lang.bitxor_assign_trait()),
            hir::AssignOpKind::BitAndAssign => (sym::bitand_assign, lang.bitand_assign_trait()),
            hir::AssignOpKind::BitOrAssign => (sym::bitor_assign, lang.bitor_assign_trait()),
            hir::AssignOpKind::ShlAssign => (sym::shl_assign, lang.shl_assign_trait()),
            hir::AssignOpKind::ShrAssign => (sym::shr_assign, lang.shr_assign_trait()),
        },
        Op::BinOp(op) => match op.node {
            hir::BinOpKind::Add => (sym::add, lang.add_trait()),
            hir::BinOpKind::Sub => (sym::sub, lang.sub_trait()),
            hir::BinOpKind::Mul => (sym::mul, lang.mul_trait()),
            hir::BinOpKind::Div => (sym::div, lang.div_trait()),
            hir::BinOpKind::Rem => (sym::rem, lang.rem_trait()),
            hir::BinOpKind::BitXor => (sym::bitxor, lang.bitxor_trait()),
            hir::BinOpKind::BitAnd => (sym::bitand, lang.bitand_trait()),
            hir::BinOpKind::BitOr => (sym::bitor, lang.bitor_trait()),
            hir::BinOpKind::Shl => (sym::shl, lang.shl_trait()),
            hir::BinOpKind::Shr => (sym::shr, lang.shr_trait()),
            hir::BinOpKind::Lt => (sym::lt, lang.partial_ord_trait()),
            hir::BinOpKind::Le => (sym::le, lang.partial_ord_trait()),
            hir::BinOpKind::Ge => (sym::ge, lang.partial_ord_trait()),
            hir::BinOpKind::Gt => (sym::gt, lang.partial_ord_trait()),
            hir::BinOpKind::Eq => (sym::eq, lang.eq_trait()),
            hir::BinOpKind::Ne => (sym::ne, lang.eq_trait()),
            hir::BinOpKind::And | hir::BinOpKind::Or => {
                bug!("&& and || are not overloadable")
            }
        },
    }
}

fn lang_item_for_unop(tcx: TyCtxt<'_>, op: hir::UnOp) -> (Symbol, Option<hir::def_id::DefId>) {
    let lang = tcx.lang_items();
    match op {
        hir::UnOp::Not => (sym::not, lang.not_trait()),
        hir::UnOp::Neg => (sym::neg, lang.neg_trait()),
        hir::UnOp::Deref => bug!("Deref is not overloadable"),
    }
}

// Binary operator categories. These categories summarize the behavior
// with respect to the builtin operations supported.
#[derive(Clone, Copy)]
enum BinOpCategory {
    /// &&, || -- cannot be overridden
    Shortcircuit,

    /// <<, >> -- when shifting a single integer, rhs can be any
    /// integer type. For simd, types must match.
    Shift,

    /// +, -, etc -- takes equal types, produces same type as input,
    /// applicable to ints/floats/simd
    Math,

    /// &, |, ^ -- takes equal types, produces same type as input,
    /// applicable to ints/floats/simd/bool
    Bitwise,

    /// ==, !=, etc -- takes equal types, produces bools, except for simd,
    /// which produce the input type
    Comparison,
}

impl From<hir::BinOpKind> for BinOpCategory {
    fn from(op: hir::BinOpKind) -> BinOpCategory {
        use hir::BinOpKind::*;
        match op {
            Shl | Shr => BinOpCategory::Shift,
            Add | Sub | Mul | Div | Rem => BinOpCategory::Math,
            BitXor | BitAnd | BitOr => BinOpCategory::Bitwise,
            Eq | Ne | Lt | Le | Ge | Gt => BinOpCategory::Comparison,
            And | Or => BinOpCategory::Shortcircuit,
        }
    }
}

impl From<hir::AssignOpKind> for BinOpCategory {
    fn from(op: hir::AssignOpKind) -> BinOpCategory {
        use hir::AssignOpKind::*;
        match op {
            ShlAssign | ShrAssign => BinOpCategory::Shift,
            AddAssign | SubAssign | MulAssign | DivAssign | RemAssign => BinOpCategory::Math,
            BitXorAssign | BitAndAssign | BitOrAssign => BinOpCategory::Bitwise,
        }
    }
}

/// An assignment op (e.g. `a += b`), or a binary op (e.g. `a + b`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    BinOp(hir::BinOp),
    AssignOp(hir::AssignOp),
}

impl Op {
    fn span(&self) -> Span {
        match self {
            Op::BinOp(op) => op.span,
            Op::AssignOp(op) => op.span,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Op::BinOp(op) => op.node.as_str(),
            Op::AssignOp(op) => op.node.as_str(),
        }
    }

    fn is_by_value(&self) -> bool {
        match self {
            Op::BinOp(op) => op.node.is_by_value(),
            Op::AssignOp(op) => op.node.is_by_value(),
        }
    }
}

/// Dereferences a single level of immutable referencing.
fn deref_ty_if_possible(ty: Ty<'_>) -> Ty<'_> {
    match ty.kind() {
        ty::Ref(_, ty, hir::Mutability::Not) => *ty,
        _ => ty,
    }
}

/// Returns `true` if this is a built-in arithmetic operation (e.g., u32
/// + u32, i16x4 == i16x4) and false if these types would have to be
/// overloaded to be legal. There are two reasons that we distinguish
/// builtin operations from overloaded ones (vs trying to drive
/// everything uniformly through the trait system and intrinsics or
/// something like that):
///
/// 1. Builtin operations can trivially be evaluated in constants.
/// 2. For comparison operators applied to SIMD types the result is
///    not of type `bool`. For example, `i16x4 == i16x4` yields a
///    type like `i16x4`. This means that the overloaded trait
///    `PartialEq` is not applicable.
///
/// Reason #2 is the killer. I tried for a while to always use
/// overloaded logic and just check the types in constants/codegen after
/// the fact, and it worked fine, except for SIMD types. -nmatsakis
fn is_builtin_binop<'tcx>(lhs: Ty<'tcx>, rhs: Ty<'tcx>, category: BinOpCategory) -> bool {
    // Special-case a single layer of referencing, so that things like `5.0 + &6.0f32` work.
    // (See https://github.com/rust-lang/rust/issues/57447.)
    let (lhs, rhs) = (deref_ty_if_possible(lhs), deref_ty_if_possible(rhs));

    match category.into() {
        BinOpCategory::Shortcircuit => true,
        BinOpCategory::Shift => {
            lhs.references_error()
                || rhs.references_error()
                || lhs.is_integral() && rhs.is_integral()
        }
        BinOpCategory::Math => {
            lhs.references_error()
                || rhs.references_error()
                || lhs.is_integral() && rhs.is_integral()
                || lhs.is_floating_point() && rhs.is_floating_point()
        }
        BinOpCategory::Bitwise => {
            lhs.references_error()
                || rhs.references_error()
                || lhs.is_integral() && rhs.is_integral()
                || lhs.is_floating_point() && rhs.is_floating_point()
                || lhs.is_bool() && rhs.is_bool()
        }
        BinOpCategory::Comparison => {
            lhs.references_error() || rhs.references_error() || lhs.is_scalar() && rhs.is_scalar()
        }
    }
}
