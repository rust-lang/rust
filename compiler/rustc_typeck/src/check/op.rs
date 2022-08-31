//! Code related to processing overloaded binary and unary operators.

use super::method::MethodCallee;
use super::{has_expected_num_generic_args, FnCtxt};
use crate::check::Expectation;
use rustc_ast as ast;
use rustc_errors::{self, struct_span_err, Applicability, Diagnostic};
use rustc_hir as hir;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::traits::ObligationCauseCode;
use rustc_middle::ty::adjustment::{
    Adjust, Adjustment, AllowTwoPhase, AutoBorrow, AutoBorrowMutability,
};
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeFolder, TypeSuperFoldable, TypeSuperVisitable, TypeVisitable, TypeVisitor,
};
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::error_reporting::suggestions::InferCtxtExt as _;
use rustc_trait_selection::traits::{FulfillmentError, TraitEngine, TraitEngineExt};
use rustc_type_ir::sty::TyKind::*;

use std::ops::ControlFlow;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Checks a `a <op>= b`
    pub fn check_binop_assign(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        op: hir::BinOp,
        lhs: &'tcx hir::Expr<'tcx>,
        rhs: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let (lhs_ty, rhs_ty, return_ty) =
            self.check_overloaded_binop(expr, lhs, rhs, op, IsAssign::Yes, expected);

        let ty =
            if !lhs_ty.is_ty_var() && !rhs_ty.is_ty_var() && is_builtin_binop(lhs_ty, rhs_ty, op) {
                self.enforce_builtin_binop_types(lhs.span, lhs_ty, rhs.span, rhs_ty, op);
                self.tcx.mk_unit()
            } else {
                return_ty
            };

        self.check_lhs_assignable(lhs, "E0067", op.span, |err| {
            if let Some(lhs_deref_ty) = self.deref_once_mutably_for_diagnostic(lhs_ty) {
                if self
                    .lookup_op_method(
                        lhs_deref_ty,
                        Some(rhs_ty),
                        Some(rhs),
                        Op::Binary(op, IsAssign::Yes),
                        expected,
                    )
                    .is_ok()
                {
                    // Suppress this error, since we already emitted
                    // a deref suggestion in check_overloaded_binop
                    err.downgrade_to_delayed_bug();
                }
            }
        });

        ty
    }

    /// Checks a potentially overloaded binary operator.
    pub fn check_binop(
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

        match BinOpCategory::from(op) {
            BinOpCategory::Shortcircuit => {
                // && and || are a simple case.
                self.check_expr_coercable_to_type(lhs_expr, tcx.types.bool, None);
                let lhs_diverges = self.diverges.get();
                self.check_expr_coercable_to_type(rhs_expr, tcx.types.bool, None);

                // Depending on the LHS' value, the RHS can never execute.
                self.diverges.set(lhs_diverges);

                tcx.types.bool
            }
            _ => {
                // Otherwise, we always treat operators as if they are
                // overloaded. This is the way to be most flexible w/r/t
                // types that get inferred.
                let (lhs_ty, rhs_ty, return_ty) = self.check_overloaded_binop(
                    expr,
                    lhs_expr,
                    rhs_expr,
                    op,
                    IsAssign::No,
                    expected,
                );

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
                if !lhs_ty.is_ty_var()
                    && !rhs_ty.is_ty_var()
                    && is_builtin_binop(lhs_ty, rhs_ty, op)
                {
                    let builtin_return_ty = self.enforce_builtin_binop_types(
                        lhs_expr.span,
                        lhs_ty,
                        rhs_expr.span,
                        rhs_ty,
                        op,
                    );
                    self.demand_suptype(expr.span, builtin_return_ty, return_ty);
                }

                return_ty
            }
        }
    }

    fn enforce_builtin_binop_types(
        &self,
        lhs_span: Span,
        lhs_ty: Ty<'tcx>,
        rhs_span: Span,
        rhs_ty: Ty<'tcx>,
        op: hir::BinOp,
    ) -> Ty<'tcx> {
        debug_assert!(is_builtin_binop(lhs_ty, rhs_ty, op));

        // Special-case a single layer of referencing, so that things like `5.0 + &6.0f32` work.
        // (See https://github.com/rust-lang/rust/issues/57447.)
        let (lhs_ty, rhs_ty) = (deref_ty_if_possible(lhs_ty), deref_ty_if_possible(rhs_ty));

        let tcx = self.tcx;
        match BinOpCategory::from(op) {
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
        op: hir::BinOp,
        is_assign: IsAssign,
        expected: Expectation<'tcx>,
    ) -> (Ty<'tcx>, Ty<'tcx>, Ty<'tcx>) {
        debug!(
            "check_overloaded_binop(expr.hir_id={}, op={:?}, is_assign={:?})",
            expr.hir_id, op, is_assign
        );

        let lhs_ty = match is_assign {
            IsAssign::No => {
                // Find a suitable supertype of the LHS expression's type, by coercing to
                // a type variable, to pass as the `Self` to the trait, avoiding invariant
                // trait matching creating lifetime constraints that are too strict.
                // e.g., adding `&'a T` and `&'b T`, given `&'x T: Add<&'x T>`, will result
                // in `&'a T <: &'x T` and `&'b T <: &'x T`, instead of `'a = 'b = 'x`.
                let lhs_ty = self.check_expr(lhs_expr);
                let fresh_var = self.next_ty_var(TypeVariableOrigin {
                    kind: TypeVariableOriginKind::MiscVariable,
                    span: lhs_expr.span,
                });
                self.demand_coerce(lhs_expr, lhs_ty, fresh_var, Some(rhs_expr), AllowTwoPhase::No)
            }
            IsAssign::Yes => {
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
        let rhs_ty_var = self.next_ty_var(TypeVariableOrigin {
            kind: TypeVariableOriginKind::MiscVariable,
            span: rhs_expr.span,
        });

        let result = self.lookup_op_method(
            lhs_ty,
            Some(rhs_ty_var),
            Some(rhs_expr),
            Op::Binary(op, is_assign),
            expected,
        );

        // see `NB` above
        let rhs_ty = self.check_expr_coercable_to_type(rhs_expr, rhs_ty_var, Some(lhs_expr));
        let rhs_ty = self.resolve_vars_with_obligations(rhs_ty);

        let return_ty = match result {
            Ok(method) => {
                let by_ref_binop = !op.node.is_by_value();
                if is_assign == IsAssign::Yes || by_ref_binop {
                    if let ty::Ref(region, _, mutbl) = method.sig.inputs()[0].kind() {
                        let mutbl = match mutbl {
                            hir::Mutability::Not => AutoBorrowMutability::Not,
                            hir::Mutability::Mut => AutoBorrowMutability::Mut {
                                // Allow two-phase borrows for binops in initial deployment
                                // since they desugar to methods
                                allow_two_phase_borrow: AllowTwoPhase::Yes,
                            },
                        };
                        let autoref = Adjustment {
                            kind: Adjust::Borrow(AutoBorrow::Ref(*region, mutbl)),
                            target: method.sig.inputs()[0],
                        };
                        self.apply_adjustments(lhs_expr, vec![autoref]);
                    }
                }
                if by_ref_binop {
                    if let ty::Ref(region, _, mutbl) = method.sig.inputs()[1].kind() {
                        let mutbl = match mutbl {
                            hir::Mutability::Not => AutoBorrowMutability::Not,
                            hir::Mutability::Mut => AutoBorrowMutability::Mut {
                                // Allow two-phase borrows for binops in initial deployment
                                // since they desugar to methods
                                allow_two_phase_borrow: AllowTwoPhase::Yes,
                            },
                        };
                        let autoref = Adjustment {
                            kind: Adjust::Borrow(AutoBorrow::Ref(*region, mutbl)),
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
                self.write_method_call(expr.hir_id, method);

                method.sig.output()
            }
            // error types are considered "builtin"
            Err(_) if lhs_ty.references_error() || rhs_ty.references_error() => self.tcx.ty_error(),
            Err(errors) => {
                let source_map = self.tcx.sess.source_map();
                let (mut err, missing_trait, use_output) = match is_assign {
                    IsAssign::Yes => {
                        let mut err = struct_span_err!(
                            self.tcx.sess,
                            expr.span,
                            E0368,
                            "binary assignment operation `{}=` cannot be applied to type `{}`",
                            op.node.as_str(),
                            lhs_ty,
                        );
                        err.span_label(
                            lhs_expr.span,
                            format!("cannot use `{}=` on type `{}`", op.node.as_str(), lhs_ty),
                        );
                        let missing_trait = match op.node {
                            hir::BinOpKind::Add => Some("std::ops::AddAssign"),
                            hir::BinOpKind::Sub => Some("std::ops::SubAssign"),
                            hir::BinOpKind::Mul => Some("std::ops::MulAssign"),
                            hir::BinOpKind::Div => Some("std::ops::DivAssign"),
                            hir::BinOpKind::Rem => Some("std::ops::RemAssign"),
                            hir::BinOpKind::BitAnd => Some("std::ops::BitAndAssign"),
                            hir::BinOpKind::BitXor => Some("std::ops::BitXorAssign"),
                            hir::BinOpKind::BitOr => Some("std::ops::BitOrAssign"),
                            hir::BinOpKind::Shl => Some("std::ops::ShlAssign"),
                            hir::BinOpKind::Shr => Some("std::ops::ShrAssign"),
                            _ => None,
                        };
                        self.note_unmet_impls_on_type(&mut err, errors);
                        (err, missing_trait, false)
                    }
                    IsAssign::No => {
                        let (message, missing_trait, use_output) = match op.node {
                            hir::BinOpKind::Add => (
                                format!("cannot add `{rhs_ty}` to `{lhs_ty}`"),
                                Some("std::ops::Add"),
                                true,
                            ),
                            hir::BinOpKind::Sub => (
                                format!("cannot subtract `{rhs_ty}` from `{lhs_ty}`"),
                                Some("std::ops::Sub"),
                                true,
                            ),
                            hir::BinOpKind::Mul => (
                                format!("cannot multiply `{lhs_ty}` by `{rhs_ty}`"),
                                Some("std::ops::Mul"),
                                true,
                            ),
                            hir::BinOpKind::Div => (
                                format!("cannot divide `{lhs_ty}` by `{rhs_ty}`"),
                                Some("std::ops::Div"),
                                true,
                            ),
                            hir::BinOpKind::Rem => (
                                format!("cannot mod `{lhs_ty}` by `{rhs_ty}`"),
                                Some("std::ops::Rem"),
                                true,
                            ),
                            hir::BinOpKind::BitAnd => (
                                format!("no implementation for `{lhs_ty} & {rhs_ty}`"),
                                Some("std::ops::BitAnd"),
                                true,
                            ),
                            hir::BinOpKind::BitXor => (
                                format!("no implementation for `{lhs_ty} ^ {rhs_ty}`"),
                                Some("std::ops::BitXor"),
                                true,
                            ),
                            hir::BinOpKind::BitOr => (
                                format!("no implementation for `{lhs_ty} | {rhs_ty}`"),
                                Some("std::ops::BitOr"),
                                true,
                            ),
                            hir::BinOpKind::Shl => (
                                format!("no implementation for `{lhs_ty} << {rhs_ty}`"),
                                Some("std::ops::Shl"),
                                true,
                            ),
                            hir::BinOpKind::Shr => (
                                format!("no implementation for `{lhs_ty} >> {rhs_ty}`"),
                                Some("std::ops::Shr"),
                                true,
                            ),
                            hir::BinOpKind::Eq | hir::BinOpKind::Ne => (
                                format!(
                                    "binary operation `{}` cannot be applied to type `{}`",
                                    op.node.as_str(),
                                    lhs_ty
                                ),
                                Some("std::cmp::PartialEq"),
                                false,
                            ),
                            hir::BinOpKind::Lt
                            | hir::BinOpKind::Le
                            | hir::BinOpKind::Gt
                            | hir::BinOpKind::Ge => (
                                format!(
                                    "binary operation `{}` cannot be applied to type `{}`",
                                    op.node.as_str(),
                                    lhs_ty
                                ),
                                Some("std::cmp::PartialOrd"),
                                false,
                            ),
                            _ => (
                                format!(
                                    "binary operation `{}` cannot be applied to type `{}`",
                                    op.node.as_str(),
                                    lhs_ty
                                ),
                                None,
                                false,
                            ),
                        };
                        let mut err = struct_span_err!(self.tcx.sess, op.span, E0369, "{message}");
                        if !lhs_expr.span.eq(&rhs_expr.span) {
                            err.span_label(lhs_expr.span, lhs_ty.to_string());
                            err.span_label(rhs_expr.span, rhs_ty.to_string());
                        }
                        self.note_unmet_impls_on_type(&mut err, errors);
                        (err, missing_trait, use_output)
                    }
                };

                let mut suggest_deref_binop = |lhs_deref_ty: Ty<'tcx>| {
                    if self
                        .lookup_op_method(
                            lhs_deref_ty,
                            Some(rhs_ty),
                            Some(rhs_expr),
                            Op::Binary(op, is_assign),
                            expected,
                        )
                        .is_ok()
                    {
                        if let Ok(lstring) = source_map.span_to_snippet(lhs_expr.span) {
                            let msg = &format!(
                                "`{}{}` can be used on `{}`, you can dereference `{}`",
                                op.node.as_str(),
                                match is_assign {
                                    IsAssign::Yes => "=",
                                    IsAssign::No => "",
                                },
                                lhs_deref_ty.peel_refs(),
                                lstring,
                            );
                            err.span_suggestion_verbose(
                                lhs_expr.span.shrink_to_lo(),
                                msg,
                                "*",
                                rustc_errors::Applicability::MachineApplicable,
                            );
                        }
                    }
                };

                let is_compatible = |lhs_ty, rhs_ty| {
                    self.lookup_op_method(
                        lhs_ty,
                        Some(rhs_ty),
                        Some(rhs_expr),
                        Op::Binary(op, is_assign),
                        expected,
                    )
                    .is_ok()
                };

                // We should suggest `a + b` => `*a + b` if `a` is copy, and suggest
                // `a += b` => `*a += b` if a is a mut ref.
                if !op.span.can_be_used_for_suggestions() {
                    // Suppress suggestions when lhs and rhs are not in the same span as the error
                } else if is_assign == IsAssign::Yes
                    && let Some(lhs_deref_ty) = self.deref_once_mutably_for_diagnostic(lhs_ty)
                {
                    suggest_deref_binop(lhs_deref_ty);
                } else if is_assign == IsAssign::No
                    && let Ref(_, lhs_deref_ty, _) = lhs_ty.kind()
                {
                    if self.type_is_copy_modulo_regions(
                        self.param_env,
                        *lhs_deref_ty,
                        lhs_expr.span,
                    ) {
                        suggest_deref_binop(*lhs_deref_ty);
                    }
                } else if self.suggest_fn_call(&mut err, lhs_expr, lhs_ty, |lhs_ty| {
                    is_compatible(lhs_ty, rhs_ty)
                }) || self.suggest_fn_call(&mut err, rhs_expr, rhs_ty, |rhs_ty| {
                    is_compatible(lhs_ty, rhs_ty)
                }) || self.suggest_two_fn_call(
                    &mut err,
                    rhs_expr,
                    rhs_ty,
                    lhs_expr,
                    lhs_ty,
                    |lhs_ty, rhs_ty| is_compatible(lhs_ty, rhs_ty),
                ) {
                    // Cool
                }

                if let Some(missing_trait) = missing_trait {
                    let mut visitor = TypeParamVisitor(vec![]);
                    visitor.visit_ty(lhs_ty);

                    if op.node == hir::BinOpKind::Add
                        && self.check_str_addition(
                            lhs_expr, rhs_expr, lhs_ty, rhs_ty, &mut err, is_assign, op,
                        )
                    {
                        // This has nothing here because it means we did string
                        // concatenation (e.g., "Hello " + "World!"). This means
                        // we don't want the note in the else clause to be emitted
                    } else if let [ty] = &visitor.0[..] {
                        // Look for a TraitPredicate in the Fulfillment errors,
                        // and use it to generate a suggestion.
                        //
                        // Note that lookup_op_method must be called again but
                        // with a specific rhs_ty instead of a placeholder so
                        // the resulting predicate generates a more specific
                        // suggestion for the user.
                        let errors = self
                            .lookup_op_method(
                                lhs_ty,
                                Some(rhs_ty),
                                Some(rhs_expr),
                                Op::Binary(op, is_assign),
                                expected,
                            )
                            .unwrap_err();
                        if !errors.is_empty() {
                            for error in errors {
                                if let Some(trait_pred) =
                                    error.obligation.predicate.to_opt_poly_trait_pred()
                                {
                                    let proj_pred = match error.obligation.cause.code() {
                                        ObligationCauseCode::BinOp {
                                            output_pred: Some(output_pred),
                                            ..
                                        } if use_output => {
                                            output_pred.to_opt_poly_projection_pred()
                                        }
                                        _ => None,
                                    };

                                    self.suggest_restricting_param_bound(
                                        &mut err,
                                        trait_pred,
                                        proj_pred,
                                        self.body_id,
                                    );
                                }
                            }
                        } else if *ty != lhs_ty {
                            // When we know that a missing bound is responsible, we don't show
                            // this note as it is redundant.
                            err.note(&format!(
                                "the trait `{missing_trait}` is not implemented for `{lhs_ty}`"
                            ));
                        }
                    }
                }
                err.emit();
                self.tcx.ty_error()
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
        err: &mut Diagnostic,
        is_assign: IsAssign,
        op: hir::BinOp,
    ) -> bool {
        let str_concat_note = "string concatenation requires an owned `String` on the left";
        let rm_borrow_msg = "remove the borrow to obtain an owned `String`";
        let to_owned_msg = "create an owned `String` from a string reference";

        let is_std_string = |ty: Ty<'tcx>| {
            ty.ty_adt_def()
                .map_or(false, |ty_def| self.tcx.is_diagnostic_item(sym::String, ty_def.did()))
        };

        match (lhs_ty.kind(), rhs_ty.kind()) {
            (&Ref(_, l_ty, _), &Ref(_, r_ty, _)) // &str or &String + &str, &String or &&str
                if (*l_ty.kind() == Str || is_std_string(l_ty))
                    && (*r_ty.kind() == Str
                        || is_std_string(r_ty)
                        || matches!(
                            r_ty.kind(), Ref(_, inner_ty, _) if *inner_ty.kind() == Str
                        )) =>
            {
                if let IsAssign::No = is_assign { // Do not supply this message if `&str += &str`
                    err.span_label(op.span, "`+` cannot be used to concatenate two `&str` strings");
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
            (&Ref(_, l_ty, _), &Adt(..)) // Handle `&str` & `&String` + `String`
                if (*l_ty.kind() == Str || is_std_string(l_ty)) && is_std_string(rhs_ty) =>
            {
                err.span_label(
                    op.span,
                    "`+` cannot be used to concatenate a `&str` with a `String`",
                );
                match is_assign {
                    IsAssign::No => {
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
                    IsAssign::Yes => {
                        err.note(str_concat_note);
                    }
                }
                true
            }
            _ => false,
        }
    }

    pub fn check_user_unop(
        &self,
        ex: &'tcx hir::Expr<'tcx>,
        operand_ty: Ty<'tcx>,
        op: hir::UnOp,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        assert!(op.is_by_value());
        match self.lookup_op_method(operand_ty, None, None, Op::Unary(op, ex.span), expected) {
            Ok(method) => {
                self.write_method_call(ex.hir_id, method);
                method.sig.output()
            }
            Err(errors) => {
                let actual = self.resolve_vars_if_possible(operand_ty);
                if !actual.references_error() {
                    let mut err = struct_span_err!(
                        self.tcx.sess,
                        ex.span,
                        E0600,
                        "cannot apply unary operator `{}` to type `{}`",
                        op.as_str(),
                        actual
                    );
                    err.span_label(
                        ex.span,
                        format!("cannot apply unary operator `{}`", op.as_str()),
                    );

                    let mut visitor = TypeParamVisitor(vec![]);
                    visitor.visit_ty(operand_ty);
                    if let [_] = &visitor.0[..] && let ty::Param(_) = *operand_ty.kind() {
                        let predicates = errors
                            .iter()
                            .filter_map(|error| {
                                error.obligation.predicate.to_opt_poly_trait_pred()
                            });
                        for pred in predicates {
                            self.suggest_restricting_param_bound(
                                &mut err,
                                pred,
                                None,
                                self.body_id,
                            );
                        }
                    }

                    let sp = self.tcx.sess.source_map().start_point(ex.span);
                    if let Some(sp) =
                        self.tcx.sess.parse_sess.ambiguous_block_expr_parse.borrow().get(&sp)
                    {
                        // If the previous expression was a block expression, suggest parentheses
                        // (turning this into a binary subtraction operation instead.)
                        // for example, `{2} - 2` -> `({2}) - 2` (see src\test\ui\parser\expr-as-stmt.rs)
                        self.tcx.sess.parse_sess.expr_parentheses_needed(&mut err, *sp);
                    } else {
                        match actual.kind() {
                            Uint(_) if op == hir::UnOp::Neg => {
                                err.note("unsigned values cannot be negated");

                                if let hir::ExprKind::Unary(
                                    _,
                                    hir::Expr {
                                        kind:
                                            hir::ExprKind::Lit(Spanned {
                                                node: ast::LitKind::Int(1, _),
                                                ..
                                            }),
                                        ..
                                    },
                                ) = ex.kind
                                {
                                    err.span_suggestion(
                                        ex.span,
                                        &format!(
                                            "you may have meant the maximum value of `{actual}`",
                                        ),
                                        format!("{actual}::MAX"),
                                        Applicability::MaybeIncorrect,
                                    );
                                }
                            }
                            Str | Never | Char | Tuple(_) | Array(_, _) => {}
                            Ref(_, lty, _) if *lty.kind() == Str => {}
                            _ => {
                                self.note_unmet_impls_on_type(&mut err, errors);
                            }
                        }
                    }
                    err.emit();
                }
                self.tcx.ty_error()
            }
        }
    }

    fn lookup_op_method(
        &self,
        lhs_ty: Ty<'tcx>,
        other_ty: Option<Ty<'tcx>>,
        other_ty_expr: Option<&'tcx hir::Expr<'tcx>>,
        op: Op,
        expected: Expectation<'tcx>,
    ) -> Result<MethodCallee<'tcx>, Vec<FulfillmentError<'tcx>>> {
        let lang = self.tcx.lang_items();

        let span = match op {
            Op::Binary(op, _) => op.span,
            Op::Unary(_, span) => span,
        };
        let (opname, trait_did) = if let Op::Binary(op, IsAssign::Yes) = op {
            match op.node {
                hir::BinOpKind::Add => (sym::add_assign, lang.add_assign_trait()),
                hir::BinOpKind::Sub => (sym::sub_assign, lang.sub_assign_trait()),
                hir::BinOpKind::Mul => (sym::mul_assign, lang.mul_assign_trait()),
                hir::BinOpKind::Div => (sym::div_assign, lang.div_assign_trait()),
                hir::BinOpKind::Rem => (sym::rem_assign, lang.rem_assign_trait()),
                hir::BinOpKind::BitXor => (sym::bitxor_assign, lang.bitxor_assign_trait()),
                hir::BinOpKind::BitAnd => (sym::bitand_assign, lang.bitand_assign_trait()),
                hir::BinOpKind::BitOr => (sym::bitor_assign, lang.bitor_assign_trait()),
                hir::BinOpKind::Shl => (sym::shl_assign, lang.shl_assign_trait()),
                hir::BinOpKind::Shr => (sym::shr_assign, lang.shr_assign_trait()),
                hir::BinOpKind::Lt
                | hir::BinOpKind::Le
                | hir::BinOpKind::Ge
                | hir::BinOpKind::Gt
                | hir::BinOpKind::Eq
                | hir::BinOpKind::Ne
                | hir::BinOpKind::And
                | hir::BinOpKind::Or => {
                    span_bug!(span, "impossible assignment operation: {}=", op.node.as_str())
                }
            }
        } else if let Op::Binary(op, IsAssign::No) = op {
            match op.node {
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
                    span_bug!(span, "&& and || are not overloadable")
                }
            }
        } else if let Op::Unary(hir::UnOp::Not, _) = op {
            (sym::not, lang.not_trait())
        } else if let Op::Unary(hir::UnOp::Neg, _) = op {
            (sym::neg, lang.neg_trait())
        } else {
            bug!("lookup_op_method: op not supported: {:?}", op)
        };

        debug!(
            "lookup_op_method(lhs_ty={:?}, op={:?}, opname={:?}, trait_did={:?})",
            lhs_ty, op, opname, trait_did
        );

        // Catches cases like #83893, where a lang item is declared with the
        // wrong number of generic arguments. Should have yielded an error
        // elsewhere by now, but we have to catch it here so that we do not
        // index `other_tys` out of bounds (if the lang item has too many
        // generic arguments, `other_tys` is too short).
        if !has_expected_num_generic_args(
            self.tcx,
            trait_did,
            match op {
                // Binary ops have a generic right-hand side, unary ops don't
                Op::Binary(..) => 1,
                Op::Unary(..) => 0,
            },
        ) {
            return Err(vec![]);
        }

        let opname = Ident::with_dummy_span(opname);
        let method = trait_did.and_then(|trait_did| {
            self.lookup_op_method_in_trait(
                span,
                opname,
                trait_did,
                lhs_ty,
                other_ty,
                other_ty_expr,
                expected,
            )
        });

        match (method, trait_did) {
            (Some(ok), _) => {
                let method = self.register_infer_ok_obligations(ok);
                self.select_obligations_where_possible(false, |_| {});
                Ok(method)
            }
            (None, None) => Err(vec![]),
            (None, Some(trait_did)) => {
                let (obligation, _) = self.obligation_for_op_method(
                    span,
                    trait_did,
                    lhs_ty,
                    other_ty,
                    other_ty_expr,
                    expected,
                );
                let mut fulfill = <dyn TraitEngine<'_>>::new(self.tcx);
                fulfill.register_predicate_obligation(self, obligation);
                Err(fulfill.select_where_possible(&self.infcx))
            }
        }
    }
}

// Binary operator categories. These categories summarize the behavior
// with respect to the builtin operations supported.
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

impl BinOpCategory {
    fn from(op: hir::BinOp) -> BinOpCategory {
        match op.node {
            hir::BinOpKind::Shl | hir::BinOpKind::Shr => BinOpCategory::Shift,

            hir::BinOpKind::Add
            | hir::BinOpKind::Sub
            | hir::BinOpKind::Mul
            | hir::BinOpKind::Div
            | hir::BinOpKind::Rem => BinOpCategory::Math,

            hir::BinOpKind::BitXor | hir::BinOpKind::BitAnd | hir::BinOpKind::BitOr => {
                BinOpCategory::Bitwise
            }

            hir::BinOpKind::Eq
            | hir::BinOpKind::Ne
            | hir::BinOpKind::Lt
            | hir::BinOpKind::Le
            | hir::BinOpKind::Ge
            | hir::BinOpKind::Gt => BinOpCategory::Comparison,

            hir::BinOpKind::And | hir::BinOpKind::Or => BinOpCategory::Shortcircuit,
        }
    }
}

/// Whether the binary operation is an assignment (`a += b`), or not (`a + b`)
#[derive(Clone, Copy, Debug, PartialEq)]
enum IsAssign {
    No,
    Yes,
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Binary(hir::BinOp, IsAssign),
    Unary(hir::UnOp, Span),
}

/// Dereferences a single level of immutable referencing.
fn deref_ty_if_possible<'tcx>(ty: Ty<'tcx>) -> Ty<'tcx> {
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
fn is_builtin_binop<'tcx>(lhs: Ty<'tcx>, rhs: Ty<'tcx>, op: hir::BinOp) -> bool {
    // Special-case a single layer of referencing, so that things like `5.0 + &6.0f32` work.
    // (See https://github.com/rust-lang/rust/issues/57447.)
    let (lhs, rhs) = (deref_ty_if_possible(lhs), deref_ty_if_possible(rhs));

    match BinOpCategory::from(op) {
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

struct TypeParamVisitor<'tcx>(Vec<Ty<'tcx>>);

impl<'tcx> TypeVisitor<'tcx> for TypeParamVisitor<'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        if let ty::Param(_) = ty.kind() {
            self.0.push(ty);
        }
        ty.super_visit_with(self)
    }
}

struct TypeParamEraser<'a, 'tcx>(&'a FnCtxt<'a, 'tcx>, Span);

impl<'tcx> TypeFolder<'tcx> for TypeParamEraser<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.0.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match ty.kind() {
            ty::Param(_) => self.0.next_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::MiscVariable,
                span: self.1,
            }),
            _ => ty.super_fold_with(self),
        }
    }
}
