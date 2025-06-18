use rustc_errors::{Applicability, Diag};
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{self as hir, ExprKind, PatKind};
use rustc_hir_pretty::ty_to_string;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;
use rustc_trait_selection::traits::{
    IfExpressionCause, MatchExpressionArmCause, ObligationCause, ObligationCauseCode,
};
use tracing::{debug, instrument};

use crate::coercion::{AsCoercionSite, CoerceMany};
use crate::{Diverges, Expectation, FnCtxt, GatherLocalsVisitor, Needs};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    #[instrument(skip(self), level = "debug", ret)]
    pub(crate) fn check_expr_match(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        scrut: &'tcx hir::Expr<'tcx>,
        arms: &'tcx [hir::Arm<'tcx>],
        orig_expected: Expectation<'tcx>,
        match_src: hir::MatchSource,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;

        let acrb = arms_contain_ref_bindings(arms);
        let scrutinee_ty = self.demand_scrutinee_type(scrut, acrb, arms.is_empty());
        debug!(?scrutinee_ty);

        // If there are no arms, that is a diverging match; a special case.
        if arms.is_empty() {
            self.diverges.set(self.diverges.get() | Diverges::always(expr.span));
            return tcx.types.never;
        }

        self.warn_arms_when_scrutinee_diverges(arms);

        // Otherwise, we have to union together the types that the arms produce and so forth.
        let scrut_diverges = self.diverges.replace(Diverges::Maybe);

        // #55810: Type check patterns first so we get types for all bindings.
        let scrut_span = scrut.span.find_ancestor_inside(expr.span).unwrap_or(scrut.span);
        for arm in arms {
            GatherLocalsVisitor::gather_from_arm(self, arm);

            self.check_pat_top(arm.pat, scrutinee_ty, Some(scrut_span), Some(scrut), None);
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

        let expected =
            orig_expected.try_structurally_resolve_and_adjust_for_branches(self, expr.span);
        debug!(?expected);

        let mut coercion = {
            let coerce_first = match expected {
                // We don't coerce to `()` so that if the match expression is a
                // statement it's branches can have any consistent type. That allows
                // us to give better error messages (pointing to a usually better
                // arm for inconsistent arms or to the whole match when a `()` type
                // is required).
                Expectation::ExpectHasType(ety) if ety != tcx.types.unit => ety,
                _ => self.next_ty_var(expr.span),
            };
            CoerceMany::with_coercion_sites(coerce_first, arms)
        };

        let mut prior_non_diverging_arms = vec![]; // Used only for diagnostics.
        let mut prior_arm = None;
        for arm in arms {
            self.diverges.set(Diverges::Maybe);

            if let Some(e) = &arm.guard {
                self.check_expr_has_type_or_error(e, tcx.types.bool, |_| {});

                // FIXME: If this is the first arm and the pattern is irrefutable,
                // e.g. `_` or `x`, and the guard diverges, then the whole match
                // may also be considered to diverge. We should warn on all subsequent
                // arms, too, just like we do for diverging scrutinees above.
            }

            // N.B. We don't reset diverges here b/c we want to warn in the arm
            // if the guard diverges, like: `x if { loop {} } => f()`, and we
            // also want to consider the arm to diverge itself.

            let arm_ty = self.check_expr_with_expectation(arm.body, expected);
            all_arms_diverge &= self.diverges.get();
            let tail_defines_return_position_impl_trait =
                self.return_position_impl_trait_from_match_expectation(orig_expected);

            let (arm_block_id, arm_span) = if let hir::ExprKind::Block(blk, _) = arm.body.kind {
                (Some(blk.hir_id), self.find_block_span(blk))
            } else {
                (None, arm.body.span)
            };

            let code = match prior_arm {
                // The reason for the first arm to fail is not that the match arms diverge,
                // but rather that there's a prior obligation that doesn't hold.
                None => ObligationCauseCode::BlockTailExpression(arm.body.hir_id, match_src),
                Some((prior_arm_block_id, prior_arm_ty, prior_arm_span)) => {
                    ObligationCauseCode::MatchExpressionArm(Box::new(MatchExpressionArmCause {
                        arm_block_id,
                        arm_span,
                        arm_ty,
                        prior_arm_block_id,
                        prior_arm_ty,
                        prior_arm_span,
                        scrut_span: scrut.span,
                        expr_span: expr.span,
                        source: match_src,
                        prior_non_diverging_arms: prior_non_diverging_arms.clone(),
                        tail_defines_return_position_impl_trait,
                    }))
                }
            };
            let cause = self.cause(arm_span, code);

            // This is the moral equivalent of `coercion.coerce(self, cause, arm.body, arm_ty)`.
            // We use it this way to be able to expand on the potential error and detect when a
            // `match` tail statement could be a tail expression instead. If so, we suggest
            // removing the stray semicolon.
            coercion.coerce_inner(
                self,
                &cause,
                Some(arm.body),
                arm_ty,
                |err| {
                    self.explain_never_type_coerced_to_unit(err, arm, arm_ty, prior_arm, expr);
                },
                false,
            );

            if !arm_ty.is_never() {
                // When a match arm has type `!`, then it doesn't influence the expected type for
                // the following arm. If all of the prior arms are `!`, then the influence comes
                // from elsewhere and we shouldn't point to any previous arm.
                prior_arm = Some((arm_block_id, arm_ty, arm_span));

                prior_non_diverging_arms.push(arm_span);
                if prior_non_diverging_arms.len() > 5 {
                    prior_non_diverging_arms.remove(0);
                }
            }
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

    fn explain_never_type_coerced_to_unit(
        &self,
        err: &mut Diag<'_>,
        arm: &hir::Arm<'tcx>,
        arm_ty: Ty<'tcx>,
        prior_arm: Option<(Option<hir::HirId>, Ty<'tcx>, Span)>,
        expr: &hir::Expr<'tcx>,
    ) {
        if let hir::ExprKind::Block(block, _) = arm.body.kind
            && let Some(expr) = block.expr
            && let arm_tail_ty = self.node_ty(expr.hir_id)
            && arm_tail_ty.is_never()
            && !arm_ty.is_never()
        {
            err.span_label(
                expr.span,
                format!(
                    "this expression is of type `!`, but it is coerced to `{arm_ty}` due to its \
                     surrounding expression",
                ),
            );
            self.suggest_mismatched_types_on_tail(
                err,
                expr,
                arm_ty,
                prior_arm.map_or(arm_tail_ty, |(_, ty, _)| ty),
                expr.hir_id,
            );
        }
        self.suggest_removing_semicolon_for_coerce(err, expr, arm_ty, prior_arm)
    }

    fn suggest_removing_semicolon_for_coerce(
        &self,
        diag: &mut Diag<'_>,
        expr: &hir::Expr<'tcx>,
        arm_ty: Ty<'tcx>,
        prior_arm: Option<(Option<hir::HirId>, Ty<'tcx>, Span)>,
    ) {
        // First, check that we're actually in the tail of a function.
        let Some(body) = self.tcx.hir_maybe_body_owned_by(self.body_id) else {
            return;
        };
        let hir::ExprKind::Block(block, _) = body.value.kind else {
            return;
        };
        let Some(hir::Stmt { kind: hir::StmtKind::Semi(last_expr), span: semi_span, .. }) =
            block.innermost_block().stmts.last()
        else {
            return;
        };
        if last_expr.hir_id != expr.hir_id {
            return;
        }

        // Next, make sure that we have no type expectation.
        let Some(ret) =
            self.tcx.hir_node_by_def_id(self.body_id).fn_decl().map(|decl| decl.output.span())
        else {
            return;
        };

        let can_coerce_to_return_ty = match self.ret_coercion.as_ref() {
            Some(ret_coercion) => {
                let ret_ty = ret_coercion.borrow().expected_ty();
                let ret_ty = self.infcx.shallow_resolve(ret_ty);
                self.may_coerce(arm_ty, ret_ty)
                    && prior_arm.is_none_or(|(_, ty, _)| self.may_coerce(ty, ret_ty))
                    // The match arms need to unify for the case of `impl Trait`.
                    && !matches!(ret_ty.kind(), ty::Alias(ty::Opaque, ..))
            }
            _ => false,
        };
        if !can_coerce_to_return_ty {
            return;
        }

        let semi = expr.span.shrink_to_hi().with_hi(semi_span.hi());
        let sugg = crate::errors::RemoveSemiForCoerce { expr: expr.span, ret, semi };
        diag.subdiagnostic(sugg);
    }

    /// When the previously checked expression (the scrutinee) diverges,
    /// warn the user about the match arms being unreachable.
    fn warn_arms_when_scrutinee_diverges(&self, arms: &'tcx [hir::Arm<'tcx>]) {
        for arm in arms {
            self.warn_if_unreachable(arm.body.hir_id, arm.body.span, "arm");
        }
    }

    /// Handle the fallback arm of a desugared if(-let) like a missing else.
    ///
    /// Returns `true` if there was an error forcing the coercion to the `()` type.
    pub(super) fn if_fallback_coercion<T>(
        &self,
        if_span: Span,
        cond_expr: &'tcx hir::Expr<'tcx>,
        then_expr: &'tcx hir::Expr<'tcx>,
        coercion: &mut CoerceMany<'tcx, '_, T>,
    ) -> bool
    where
        T: AsCoercionSite,
    {
        // If this `if` expr is the parent's function return expr,
        // the cause of the type coercion is the return type, point at it. (#25228)
        let hir_id = self.tcx.parent_hir_id(self.tcx.parent_hir_id(then_expr.hir_id));
        let ret_reason = self.maybe_get_coercion_reason(hir_id, if_span);
        let cause = self.cause(if_span, ObligationCauseCode::IfExpressionWithNoElse);
        let mut error = false;
        coercion.coerce_forced_unit(
            self,
            &cause,
            |err| self.explain_if_expr(err, ret_reason, if_span, cond_expr, then_expr, &mut error),
            false,
        );
        error
    }

    /// Explain why `if` expressions without `else` evaluate to `()` and detect likely irrefutable
    /// `if let PAT = EXPR {}` expressions that could be turned into `let PAT = EXPR;`.
    fn explain_if_expr(
        &self,
        err: &mut Diag<'_>,
        ret_reason: Option<(Span, String)>,
        if_span: Span,
        cond_expr: &'tcx hir::Expr<'tcx>,
        then_expr: &'tcx hir::Expr<'tcx>,
        error: &mut bool,
    ) {
        if let Some((if_span, msg)) = ret_reason {
            err.span_label(if_span, msg);
        } else if let ExprKind::Block(block, _) = then_expr.kind
            && let Some(expr) = block.expr
        {
            err.span_label(expr.span, "found here");
        }
        err.note("`if` expressions without `else` evaluate to `()`");
        err.help("consider adding an `else` block that evaluates to the expected type");
        *error = true;
        if let ExprKind::Let(hir::LetExpr { span, pat, init, .. }) = cond_expr.kind
            && let ExprKind::Block(block, _) = then_expr.kind
            // Refutability checks occur on the MIR, so we approximate it here by checking
            // if we have an enum with a single variant or a struct in the pattern.
            && let PatKind::TupleStruct(qpath, ..) | PatKind::Struct(qpath, ..) = pat.kind
            && let hir::QPath::Resolved(_, path) = qpath
        {
            match path.res {
                Res::Def(DefKind::Ctor(CtorOf::Struct, _), _) => {
                    // Structs are always irrefutable. Their fields might not be, but we
                    // don't check for that here, it's only an approximation.
                }
                Res::Def(DefKind::Ctor(CtorOf::Variant, _), def_id)
                    if self
                        .tcx
                        .adt_def(self.tcx.parent(self.tcx.parent(def_id)))
                        .variants()
                        .len()
                        == 1 =>
                {
                    // There's only a single variant in the `enum`, so we can suggest the
                    // irrefutable `let` instead of `if let`.
                }
                _ => return,
            }

            let mut sugg = vec![
                // Remove the `if`
                (if_span.until(*span), String::new()),
            ];
            match (block.stmts, block.expr) {
                ([first, ..], Some(expr)) => {
                    let padding = self
                        .tcx
                        .sess
                        .source_map()
                        .indentation_before(first.span)
                        .unwrap_or_else(|| String::new());
                    sugg.extend([
                        (init.span.between(first.span), format!(";\n{padding}")),
                        (expr.span.shrink_to_hi().with_hi(block.span.hi()), String::new()),
                    ]);
                }
                ([], Some(expr)) => {
                    let padding = self
                        .tcx
                        .sess
                        .source_map()
                        .indentation_before(expr.span)
                        .unwrap_or_else(|| String::new());
                    sugg.extend([
                        (init.span.between(expr.span), format!(";\n{padding}")),
                        (expr.span.shrink_to_hi().with_hi(block.span.hi()), String::new()),
                    ]);
                }
                // If there's no value in the body, then the `if` expression would already
                // be of type `()`, so checking for those cases is unnecessary.
                (_, None) => return,
            }
            err.multipart_suggestion(
                "consider using an irrefutable `let` binding instead",
                sugg,
                Applicability::MaybeIncorrect,
            );
        }
    }

    pub(crate) fn maybe_get_coercion_reason(
        &self,
        hir_id: hir::HirId,
        sp: Span,
    ) -> Option<(Span, String)> {
        let node = self.tcx.hir_node(hir_id);
        if let hir::Node::Block(block) = node {
            // check that the body's parent is an fn
            let parent = self.tcx.parent_hir_node(self.tcx.parent_hir_id(block.hir_id));
            if let (Some(expr), hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn { .. }, .. })) =
                (&block.expr, parent)
            {
                // check that the `if` expr without `else` is the fn body's expr
                if expr.span == sp {
                    return self.get_fn_decl(hir_id).map(|(_, fn_decl)| {
                        let (ty, span) = match fn_decl.output {
                            hir::FnRetTy::DefaultReturn(span) => ("()".to_string(), span),
                            hir::FnRetTy::Return(ty) => (ty_to_string(&self.tcx, ty), ty.span),
                        };
                        (span, format!("expected `{ty}` because of this return type"))
                    });
                }
            }
        }
        if let hir::Node::LetStmt(hir::LetStmt { ty: Some(_), pat, .. }) = node {
            return Some((pat.span, "expected because of this assignment".to_string()));
        }
        None
    }

    pub(crate) fn if_cause(
        &self,
        span: Span,
        cond_span: Span,
        then_expr: &'tcx hir::Expr<'tcx>,
        else_expr: &'tcx hir::Expr<'tcx>,
        then_ty: Ty<'tcx>,
        else_ty: Ty<'tcx>,
        tail_defines_return_position_impl_trait: Option<LocalDefId>,
    ) -> ObligationCause<'tcx> {
        let mut outer_span = if self.tcx.sess.source_map().is_multiline(span) {
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

        let (error_sp, else_id) = if let ExprKind::Block(block, _) = &else_expr.kind {
            let block = block.innermost_block();

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
            if block.expr.is_none()
                && block.stmts.is_empty()
                && let Some(outer_span) = &mut outer_span
                && let Some(cond_span) = cond_span.find_ancestor_inside(*outer_span)
            {
                *outer_span = outer_span.with_hi(cond_span.hi())
            }

            (self.find_block_span(block), block.hir_id)
        } else {
            (else_expr.span, else_expr.hir_id)
        };

        let then_id = if let ExprKind::Block(block, _) = &then_expr.kind {
            let block = block.innermost_block();
            // Exclude overlapping spans
            if block.expr.is_none() && block.stmts.is_empty() {
                outer_span = None;
            }
            block.hir_id
        } else {
            then_expr.hir_id
        };

        // Finally construct the cause:
        self.cause(
            error_sp,
            ObligationCauseCode::IfExpression(Box::new(IfExpressionCause {
                else_id,
                then_id,
                then_ty,
                else_ty,
                outer_span,
                tail_defines_return_position_impl_trait,
            })),
        )
    }

    pub(super) fn demand_scrutinee_type(
        &self,
        scrut: &'tcx hir::Expr<'tcx>,
        contains_ref_bindings: Option<hir::Mutability>,
        no_arms: bool,
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
        if let Some(m) = contains_ref_bindings {
            self.check_expr_with_needs(scrut, Needs::maybe_mut_place(m))
        } else if no_arms {
            self.check_expr(scrut)
        } else {
            // ...but otherwise we want to use any supertype of the
            // scrutinee. This is sort of a workaround, see note (*) in
            // `check_pat` for some details.
            let scrut_ty = self.next_ty_var(scrut.span);
            self.check_expr_has_type_or_error(scrut, scrut_ty, |_| {});
            scrut_ty
        }
    }

    // Does the expectation of the match define an RPIT?
    // (e.g. we're in the tail of a function body)
    //
    // Returns the `LocalDefId` of the RPIT, which is always identity-substituted.
    pub(crate) fn return_position_impl_trait_from_match_expectation(
        &self,
        expectation: Expectation<'tcx>,
    ) -> Option<LocalDefId> {
        let expected_ty = expectation.to_option(self)?;
        let (def_id, args) = match *expected_ty.kind() {
            // FIXME: Could also check that the RPIT is not defined
            ty::Alias(ty::Opaque, alias_ty) => (alias_ty.def_id.as_local()?, alias_ty.args),
            // FIXME(-Znext-solver=no): Remove this branch once `replace_opaque_types_with_infer` is gone.
            ty::Infer(ty::TyVar(_)) => self
                .inner
                .borrow_mut()
                .opaque_types()
                .iter_opaque_types()
                .find(|(_, v)| v.ty == expected_ty)
                .map(|(k, _)| (k.def_id, k.args))?,
            _ => return None,
        };
        let hir::OpaqueTyOrigin::FnReturn { parent: parent_def_id, .. } =
            self.tcx.local_opaque_ty_origin(def_id)
        else {
            return None;
        };
        if &args[0..self.tcx.generics_of(parent_def_id).count()]
            != ty::GenericArgs::identity_for_item(self.tcx, parent_def_id).as_slice()
        {
            return None;
        }
        Some(def_id)
    }
}

fn arms_contain_ref_bindings<'tcx>(arms: &'tcx [hir::Arm<'tcx>]) -> Option<hir::Mutability> {
    arms.iter().filter_map(|a| a.pat.contains_explicit_ref_binding()).max()
}
