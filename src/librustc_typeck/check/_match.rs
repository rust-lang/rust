use crate::check::coercion::CoerceMany;
use crate::check::{Diverges, Expectation, FnCtxt, Needs};
use rustc::ty::Ty;
use rustc_hir as hir;
use rustc_hir::ExprKind;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_span::Span;
use rustc_trait_selection::traits::ObligationCauseCode;
use rustc_trait_selection::traits::{IfExpressionCause, MatchExpressionArmCause, ObligationCause};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn check_match(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        scrut: &'tcx hir::Expr<'tcx>,
        arms: &'tcx [hir::Arm<'tcx>],
        expected: Expectation<'tcx>,
        match_src: hir::MatchSource,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;

        use hir::MatchSource::*;
        let (source_if, if_no_else, force_scrutinee_bool) = match match_src {
            IfDesugar { contains_else_clause } => (true, !contains_else_clause, true),
            IfLetDesugar { contains_else_clause } => (true, !contains_else_clause, false),
            WhileDesugar => (false, false, true),
            _ => (false, false, false),
        };

        // Type check the descriminant and get its type.
        let scrut_ty = if force_scrutinee_bool {
            // Here we want to ensure:
            //
            // 1. That default match bindings are *not* accepted in the condition of an
            //    `if` expression. E.g. given `fn foo() -> &bool;` we reject `if foo() { .. }`.
            //
            // 2. By expecting `bool` for `expr` we get nice diagnostics for e.g. `if x = y { .. }`.
            //
            // FIXME(60707): Consider removing hack with principled solution.
            self.check_expr_has_type_or_error(scrut, self.tcx.types.bool, |_| {})
        } else {
            self.demand_scrutinee_type(arms, scrut)
        };

        // If there are no arms, that is a diverging match; a special case.
        if arms.is_empty() {
            self.diverges.set(self.diverges.get() | Diverges::always(expr.span));
            return tcx.types.never;
        }

        self.warn_arms_when_scrutinee_diverges(arms, match_src);

        // Otherwise, we have to union together the types that the arms produce and so forth.
        let scrut_diverges = self.diverges.replace(Diverges::Maybe);

        // #55810: Type check patterns first so we get types for all bindings.
        for arm in arms {
            self.check_pat_top(&arm.pat, scrut_ty, Some(scrut.span), true);
        }

        // Now typecheck the blocks.
        //
        // The result of the match is the common supertype of all the
        // arms. Start out the value as bottom, since it's the, well,
        // bottom the type lattice, and we'll be moving up the lattice as
        // we process each arm. (Note that any match with 0 arms is matching
        // on any empty type and is therefore unreachable; should the flow
        // of execution reach it, we will panic, so bottom is an appropriate
        // type in that case)
        let mut all_arms_diverge = Diverges::WarnedAlways;

        let expected = expected.adjust_for_branches(self);

        let mut coercion = {
            let coerce_first = match expected {
                // We don't coerce to `()` so that if the match expression is a
                // statement it's branches can have any consistent type. That allows
                // us to give better error messages (pointing to a usually better
                // arm for inconsistent arms or to the whole match when a `()` type
                // is required).
                Expectation::ExpectHasType(ety) if ety != self.tcx.mk_unit() => ety,
                _ => self.next_ty_var(TypeVariableOrigin {
                    kind: TypeVariableOriginKind::MiscVariable,
                    span: expr.span,
                }),
            };
            CoerceMany::with_coercion_sites(coerce_first, arms)
        };

        let mut other_arms = vec![]; // Used only for diagnostics.
        let mut prior_arm_ty = None;
        for (i, arm) in arms.iter().enumerate() {
            if let Some(g) = &arm.guard {
                self.diverges.set(Diverges::Maybe);
                match g {
                    hir::Guard::If(e) => {
                        self.check_expr_has_type_or_error(e, tcx.types.bool, |_| {})
                    }
                };
            }

            self.diverges.set(Diverges::Maybe);
            let arm_ty = if source_if
                && if_no_else
                && i != 0
                && self.if_fallback_coercion(expr.span, &arms[0].body, &mut coercion)
            {
                tcx.types.err
            } else {
                // Only call this if this is not an `if` expr with an expected type and no `else`
                // clause to avoid duplicated type errors. (#60254)
                self.check_expr_with_expectation(&arm.body, expected)
            };
            all_arms_diverge &= self.diverges.get();
            if source_if {
                let then_expr = &arms[0].body;
                match (i, if_no_else) {
                    (0, _) => coercion.coerce(self, &self.misc(expr.span), &arm.body, arm_ty),
                    (_, true) => {} // Handled above to avoid duplicated type errors (#60254).
                    (_, _) => {
                        let then_ty = prior_arm_ty.unwrap();
                        let cause = self.if_cause(expr.span, then_expr, &arm.body, then_ty, arm_ty);
                        coercion.coerce(self, &cause, &arm.body, arm_ty);
                    }
                }
            } else {
                let arm_span = if let hir::ExprKind::Block(blk, _) = &arm.body.kind {
                    // Point at the block expr instead of the entire block
                    blk.expr.as_ref().map(|e| e.span).unwrap_or(arm.body.span)
                } else {
                    arm.body.span
                };
                let (span, code) = match i {
                    // The reason for the first arm to fail is not that the match arms diverge,
                    // but rather that there's a prior obligation that doesn't hold.
                    0 => (arm_span, ObligationCauseCode::BlockTailExpression(arm.body.hir_id)),
                    _ => (
                        expr.span,
                        ObligationCauseCode::MatchExpressionArm(box MatchExpressionArmCause {
                            arm_span,
                            source: match_src,
                            prior_arms: other_arms.clone(),
                            last_ty: prior_arm_ty.unwrap(),
                            scrut_hir_id: scrut.hir_id,
                        }),
                    ),
                };
                let cause = self.cause(span, code);
                coercion.coerce(self, &cause, &arm.body, arm_ty);
                other_arms.push(arm_span);
                if other_arms.len() > 5 {
                    other_arms.remove(0);
                }
            }
            prior_arm_ty = Some(arm_ty);
        }

        // If all of the arms in the `match` diverge,
        // and we're dealing with an actual `match` block
        // (as opposed to a `match` desugared from something else'),
        // we can emit a better note. Rather than pointing
        // at a diverging expression in an arbitrary arm,
        // we can point at the entire `match` expression
        if let (Diverges::Always { .. }, hir::MatchSource::Normal) = (all_arms_diverge, match_src) {
            all_arms_diverge = Diverges::Always {
                span: expr.span,
                custom_note: Some(
                    "any code following this `match` expression is unreachable, as all arms diverge",
                ),
            };
        }

        // We won't diverge unless the scrutinee or all arms diverge.
        self.diverges.set(scrut_diverges | all_arms_diverge);

        coercion.complete(self)
    }

    /// When the previously checked expression (the scrutinee) diverges,
    /// warn the user about the match arms being unreachable.
    fn warn_arms_when_scrutinee_diverges(
        &self,
        arms: &'tcx [hir::Arm<'tcx>],
        source: hir::MatchSource,
    ) {
        use hir::MatchSource::*;
        let msg = match source {
            IfDesugar { .. } | IfLetDesugar { .. } => "block in `if` expression",
            WhileDesugar { .. } | WhileLetDesugar { .. } => "block in `while` expression",
            _ => "arm",
        };
        for arm in arms {
            self.warn_if_unreachable(arm.body.hir_id, arm.body.span, msg);
        }
    }

    /// Handle the fallback arm of a desugared if(-let) like a missing else.
    ///
    /// Returns `true` if there was an error forcing the coercion to the `()` type.
    fn if_fallback_coercion(
        &self,
        span: Span,
        then_expr: &'tcx hir::Expr<'tcx>,
        coercion: &mut CoerceMany<'tcx, '_, rustc_hir::Arm<'tcx>>,
    ) -> bool {
        // If this `if` expr is the parent's function return expr,
        // the cause of the type coercion is the return type, point at it. (#25228)
        let ret_reason = self.maybe_get_coercion_reason(then_expr.hir_id, span);
        let cause = self.cause(span, ObligationCauseCode::IfExpressionWithNoElse);
        let mut error = false;
        coercion.coerce_forced_unit(
            self,
            &cause,
            &mut |err| {
                if let Some((span, msg)) = &ret_reason {
                    err.span_label(*span, msg.as_str());
                } else if let ExprKind::Block(block, _) = &then_expr.kind {
                    if let Some(expr) = &block.expr {
                        err.span_label(expr.span, "found here".to_string());
                    }
                }
                err.note("`if` expressions without `else` evaluate to `()`");
                err.help("consider adding an `else` block that evaluates to the expected type");
                error = true;
            },
            ret_reason.is_none(),
        );
        error
    }

    fn maybe_get_coercion_reason(&self, hir_id: hir::HirId, span: Span) -> Option<(Span, String)> {
        use hir::Node::{Block, Item, Local};

        let hir = self.tcx.hir();
        let arm_id = hir.get_parent_node(hir_id);
        let match_id = hir.get_parent_node(arm_id);
        let containing_id = hir.get_parent_node(match_id);

        let node = hir.get(containing_id);
        if let Block(block) = node {
            // check that the body's parent is an fn
            let parent = hir.get(hir.get_parent_node(hir.get_parent_node(block.hir_id)));
            if let (Some(expr), Item(hir::Item { kind: hir::ItemKind::Fn(..), .. })) =
                (&block.expr, parent)
            {
                // check that the `if` expr without `else` is the fn body's expr
                if expr.span == span {
                    return self.get_fn_decl(hir_id).map(|(fn_decl, _)| {
                        (
                            fn_decl.output.span(),
                            format!("expected `{}` because of this return type", fn_decl.output),
                        )
                    });
                }
            }
        }
        if let Local(hir::Local { ty: Some(_), pat, .. }) = node {
            return Some((pat.span, "expected because of this assignment".to_string()));
        }
        None
    }

    fn if_cause(
        &self,
        span: Span,
        then_expr: &'tcx hir::Expr<'tcx>,
        else_expr: &'tcx hir::Expr<'tcx>,
        then_ty: Ty<'tcx>,
        else_ty: Ty<'tcx>,
    ) -> ObligationCause<'tcx> {
        let mut outer_sp = if self.tcx.sess.source_map().is_multiline(span) {
            // The `if`/`else` isn't in one line in the output, include some context to make it
            // clear it is an if/else expression:
            // ```
            // LL |      let x = if true {
            //    | _____________-
            // LL ||         10i32
            //    ||         ----- expected because of this
            // LL ||     } else {
            // LL ||         10u32
            //    ||         ^^^^^ expected `i32`, found `u32`
            // LL ||     };
            //    ||_____- `if` and `else` have incompatible types
            // ```
            Some(span)
        } else {
            // The entire expression is in one line, only point at the arms
            // ```
            // LL |     let x = if true { 10i32 } else { 10u32 };
            //    |                       -----          ^^^^^ expected `i32`, found `u32`
            //    |                       |
            //    |                       expected because of this
            // ```
            None
        };

        let mut remove_semicolon = None;
        let error_sp = if let ExprKind::Block(block, _) = &else_expr.kind {
            if let Some(expr) = &block.expr {
                expr.span
            } else if let Some(stmt) = block.stmts.last() {
                // possibly incorrect trailing `;` in the else arm
                remove_semicolon = self.could_remove_semicolon(block, then_ty);
                stmt.span
            } else {
                // empty block; point at its entirety
                // Avoid overlapping spans that aren't as readable:
                // ```
                // 2 |        let x = if true {
                //   |   _____________-
                // 3 |  |         3
                //   |  |         - expected because of this
                // 4 |  |     } else {
                //   |  |____________^
                // 5 | ||
                // 6 | ||     };
                //   | ||     ^
                //   | ||_____|
                //   | |______if and else have incompatible types
                //   |        expected integer, found `()`
                // ```
                // by not pointing at the entire expression:
                // ```
                // 2 |       let x = if true {
                //   |               ------- `if` and `else` have incompatible types
                // 3 |           3
                //   |           - expected because of this
                // 4 |       } else {
                //   |  ____________^
                // 5 | |
                // 6 | |     };
                //   | |_____^ expected integer, found `()`
                // ```
                if outer_sp.is_some() {
                    outer_sp = Some(self.tcx.sess.source_map().def_span(span));
                }
                else_expr.span
            }
        } else {
            // shouldn't happen unless the parser has done something weird
            else_expr.span
        };

        // Compute `Span` of `then` part of `if`-expression.
        let then_sp = if let ExprKind::Block(block, _) = &then_expr.kind {
            if let Some(expr) = &block.expr {
                expr.span
            } else if let Some(stmt) = block.stmts.last() {
                // possibly incorrect trailing `;` in the else arm
                remove_semicolon = remove_semicolon.or(self.could_remove_semicolon(block, else_ty));
                stmt.span
            } else {
                // empty block; point at its entirety
                outer_sp = None; // same as in `error_sp`; cleanup output
                then_expr.span
            }
        } else {
            // shouldn't happen unless the parser has done something weird
            then_expr.span
        };

        // Finally construct the cause:
        self.cause(
            error_sp,
            ObligationCauseCode::IfExpression(box IfExpressionCause {
                then: then_sp,
                outer: outer_sp,
                semicolon: remove_semicolon,
            }),
        )
    }

    fn demand_scrutinee_type(
        &self,
        arms: &'tcx [hir::Arm<'tcx>],
        scrut: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        // Not entirely obvious: if matches may create ref bindings, we want to
        // use the *precise* type of the scrutinee, *not* some supertype, as
        // the "scrutinee type" (issue #23116).
        //
        // arielb1 [writes here in this comment thread][c] that there
        // is certainly *some* potential danger, e.g., for an example
        // like:
        //
        // [c]: https://github.com/rust-lang/rust/pull/43399#discussion_r130223956
        //
        // ```
        // let Foo(x) = f()[0];
        // ```
        //
        // Then if the pattern matches by reference, we want to match
        // `f()[0]` as a lexpr, so we can't allow it to be
        // coerced. But if the pattern matches by value, `f()[0]` is
        // still syntactically a lexpr, but we *do* want to allow
        // coercions.
        //
        // However, *likely* we are ok with allowing coercions to
        // happen if there are no explicit ref mut patterns - all
        // implicit ref mut patterns must occur behind a reference, so
        // they will have the "correct" variance and lifetime.
        //
        // This does mean that the following pattern would be legal:
        //
        // ```
        // struct Foo(Bar);
        // struct Bar(u32);
        // impl Deref for Foo {
        //     type Target = Bar;
        //     fn deref(&self) -> &Bar { &self.0 }
        // }
        // impl DerefMut for Foo {
        //     fn deref_mut(&mut self) -> &mut Bar { &mut self.0 }
        // }
        // fn foo(x: &mut Foo) {
        //     {
        //         let Bar(z): &mut Bar = x;
        //         *z = 42;
        //     }
        //     assert_eq!(foo.0.0, 42);
        // }
        // ```
        //
        // FIXME(tschottdorf): don't call contains_explicit_ref_binding, which
        // is problematic as the HIR is being scraped, but ref bindings may be
        // implicit after #42640. We need to make sure that pat_adjustments
        // (once introduced) is populated by the time we get here.
        //
        // See #44848.
        let contains_ref_bindings = arms
            .iter()
            .filter_map(|a| a.pat.contains_explicit_ref_binding())
            .max_by_key(|m| match *m {
                hir::Mutability::Mut => 1,
                hir::Mutability::Not => 0,
            });

        if let Some(m) = contains_ref_bindings {
            self.check_expr_with_needs(scrut, Needs::maybe_mut_place(m))
        } else {
            // ...but otherwise we want to use any supertype of the
            // scrutinee. This is sort of a workaround, see note (*) in
            // `check_pat` for some details.
            let scrut_ty = self.next_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::TypeInference,
                span: scrut.span,
            });
            self.check_expr_has_type_or_error(scrut, scrut_ty, |_| {});
            scrut_ty
        }
    }
}
