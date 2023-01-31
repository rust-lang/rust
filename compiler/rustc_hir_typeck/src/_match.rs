use crate::coercion::{AsCoercionSite, CoerceMany};
use crate::{Diverges, Expectation, FnCtxt, Needs};
use rustc_errors::{Applicability, Diagnostic, MultiSpan};
use rustc_hir::{self as hir, ExprKind};
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::traits::Obligation;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_trait_selection::traits::{
    IfExpressionCause, MatchExpressionArmCause, ObligationCause, ObligationCauseCode,
};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    #[instrument(skip(self), level = "debug", ret)]
    pub fn check_match(
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
            self.check_pat_top(&arm.pat, scrutinee_ty, Some(scrut_span), true);
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

        let expected = orig_expected.adjust_for_branches(self);
        debug!(?expected);

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
        let mut prior_arm = None;
        for arm in arms {
            if let Some(g) = &arm.guard {
                self.diverges.set(Diverges::Maybe);
                match g {
                    hir::Guard::If(e) => {
                        self.check_expr_has_type_or_error(e, tcx.types.bool, |_| {});
                    }
                    hir::Guard::IfLet(l) => {
                        self.check_expr_let(l);
                    }
                };
            }

            self.diverges.set(Diverges::Maybe);

            let arm_ty = self.check_expr_with_expectation(&arm.body, expected);
            all_arms_diverge &= self.diverges.get();

            let opt_suggest_box_span = prior_arm.and_then(|(_, prior_arm_ty, _)| {
                self.opt_suggest_box_span(prior_arm_ty, arm_ty, orig_expected)
            });

            let (arm_block_id, arm_span) = if let hir::ExprKind::Block(blk, _) = arm.body.kind {
                (Some(blk.hir_id), self.find_block_span(blk))
            } else {
                (None, arm.body.span)
            };

            let (span, code) = match prior_arm {
                // The reason for the first arm to fail is not that the match arms diverge,
                // but rather that there's a prior obligation that doesn't hold.
                None => (arm_span, ObligationCauseCode::BlockTailExpression(arm.body.hir_id)),
                Some((prior_arm_block_id, prior_arm_ty, prior_arm_span)) => (
                    expr.span,
                    ObligationCauseCode::MatchExpressionArm(Box::new(MatchExpressionArmCause {
                        arm_block_id,
                        arm_span,
                        arm_ty,
                        prior_arm_block_id,
                        prior_arm_ty,
                        prior_arm_span,
                        scrut_span: scrut.span,
                        source: match_src,
                        prior_arms: other_arms.clone(),
                        scrut_hir_id: scrut.hir_id,
                        opt_suggest_box_span,
                    })),
                ),
            };
            let cause = self.cause(span, code);

            // This is the moral equivalent of `coercion.coerce(self, cause, arm.body, arm_ty)`.
            // We use it this way to be able to expand on the potential error and detect when a
            // `match` tail statement could be a tail expression instead. If so, we suggest
            // removing the stray semicolon.
            coercion.coerce_inner(
                self,
                &cause,
                Some(&arm.body),
                arm_ty,
                Some(&mut |err| {
                    self.suggest_removing_semicolon_for_coerce(
                        err,
                        expr,
                        orig_expected,
                        arm_ty,
                        prior_arm,
                    )
                }),
                false,
            );

            other_arms.push(arm_span);
            if other_arms.len() > 5 {
                other_arms.remove(0);
            }

            prior_arm = Some((arm_block_id, arm_ty, arm_span));
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

    fn suggest_removing_semicolon_for_coerce(
        &self,
        diag: &mut Diagnostic,
        expr: &hir::Expr<'tcx>,
        expectation: Expectation<'tcx>,
        arm_ty: Ty<'tcx>,
        prior_arm: Option<(Option<hir::HirId>, Ty<'tcx>, Span)>,
    ) {
        let hir = self.tcx.hir();

        // First, check that we're actually in the tail of a function.
        let Some(body_id) = hir.maybe_body_owned_by(self.body_id) else { return; };
        let body = hir.body(body_id);
        let hir::ExprKind::Block(block, _) = body.value.kind else { return; };
        let Some(hir::Stmt { kind: hir::StmtKind::Semi(last_expr), .. })
            = block.innermost_block().stmts.last() else {  return; };
        if last_expr.hir_id != expr.hir_id {
            return;
        }

        // Next, make sure that we have no type expectation.
        let Some(ret) = hir
            .find_by_def_id(self.body_id)
            .and_then(|owner| owner.fn_decl())
            .map(|decl| decl.output.span()) else { return; };
        let Expectation::IsLast(stmt) = expectation else {
            return;
        };

        let can_coerce_to_return_ty = match self.ret_coercion.as_ref() {
            Some(ret_coercion) => {
                let ret_ty = ret_coercion.borrow().expected_ty();
                let ret_ty = self.inh.infcx.shallow_resolve(ret_ty);
                self.can_coerce(arm_ty, ret_ty)
                    && prior_arm.map_or(true, |(_, ty, _)| self.can_coerce(ty, ret_ty))
                    // The match arms need to unify for the case of `impl Trait`.
                    && !matches!(ret_ty.kind(), ty::Alias(ty::Opaque, ..))
            }
            _ => false,
        };
        if !can_coerce_to_return_ty {
            return;
        }

        let semi_span = expr.span.shrink_to_hi().with_hi(stmt.hi());
        let mut ret_span: MultiSpan = semi_span.into();
        ret_span.push_span_label(
            expr.span,
            "this could be implicitly returned but it is a statement, not a tail expression",
        );
        ret_span.push_span_label(ret, "the `match` arms can conform to this return type");
        ret_span.push_span_label(
            semi_span,
            "the `match` is a statement because of this semicolon, consider removing it",
        );
        diag.span_note(ret_span, "you might have meant to return the `match` expression");
        diag.tool_only_span_suggestion(
            semi_span,
            "remove this semicolon",
            "",
            Applicability::MaybeIncorrect,
        );
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
        span: Span,
        then_expr: &'tcx hir::Expr<'tcx>,
        coercion: &mut CoerceMany<'tcx, '_, T>,
    ) -> bool
    where
        T: AsCoercionSite,
    {
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
                    err.span_label(*span, msg);
                } else if let ExprKind::Block(block, _) = &then_expr.kind
                    && let Some(expr) = &block.expr
                {
                    err.span_label(expr.span, "found here");
                }
                err.note("`if` expressions without `else` evaluate to `()`");
                err.help("consider adding an `else` block that evaluates to the expected type");
                error = true;
            },
            false,
        );
        error
    }

    fn maybe_get_coercion_reason(&self, hir_id: hir::HirId, sp: Span) -> Option<(Span, String)> {
        let node = {
            let rslt = self.tcx.hir().parent_id(self.tcx.hir().parent_id(hir_id));
            self.tcx.hir().get(rslt)
        };
        if let hir::Node::Block(block) = node {
            // check that the body's parent is an fn
            let parent = self.tcx.hir().get_parent(self.tcx.hir().parent_id(block.hir_id));
            if let (Some(expr), hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(..), .. })) =
                (&block.expr, parent)
            {
                // check that the `if` expr without `else` is the fn body's expr
                if expr.span == sp {
                    return self.get_fn_decl(hir_id).and_then(|(fn_decl, _)| {
                        let span = fn_decl.output.span();
                        let snippet = self.tcx.sess.source_map().span_to_snippet(span).ok()?;
                        Some((span, format!("expected `{snippet}` because of this return type")))
                    });
                }
            }
        }
        if let hir::Node::Local(hir::Local { ty: Some(_), pat, .. }) = node {
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
        opt_suggest_box_span: Option<Span>,
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
            if block.expr.is_none() && block.stmts.is_empty()
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
                opt_suggest_box_span,
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
            let scrut_ty = self.next_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::TypeInference,
                span: scrut.span,
            });
            self.check_expr_has_type_or_error(scrut, scrut_ty, |_| {});
            scrut_ty
        }
    }

    /// When we have a `match` as a tail expression in a `fn` with a returned `impl Trait`
    /// we check if the different arms would work with boxed trait objects instead and
    /// provide a structured suggestion in that case.
    pub(crate) fn opt_suggest_box_span(
        &self,
        first_ty: Ty<'tcx>,
        second_ty: Ty<'tcx>,
        orig_expected: Expectation<'tcx>,
    ) -> Option<Span> {
        // FIXME(compiler-errors): This really shouldn't need to be done during the
        // "good" path of typeck, but here we are.
        match orig_expected {
            Expectation::ExpectHasType(expected) => {
                let TypeVariableOrigin {
                    span,
                    kind: TypeVariableOriginKind::OpaqueTypeInference(rpit_def_id),
                    ..
                } = self.type_var_origin(expected)? else { return None; };

                let sig = self.body_fn_sig()?;

                let substs = sig.output().walk().find_map(|arg| {
                    if let ty::GenericArgKind::Type(ty) = arg.unpack()
                        && let ty::Alias(ty::Opaque, ty::AliasTy { def_id, substs, .. }) = *ty.kind()
                        && def_id == rpit_def_id
                    {
                        Some(substs)
                    } else {
                        None
                    }
                })?;

                if !self.can_coerce(first_ty, expected) || !self.can_coerce(second_ty, expected) {
                    return None;
                }

                for ty in [first_ty, second_ty] {
                    for (pred, _) in self
                        .tcx
                        .bound_explicit_item_bounds(rpit_def_id)
                        .subst_iter_copied(self.tcx, substs)
                    {
                        let pred = pred.kind().rebind(match pred.kind().skip_binder() {
                            ty::PredicateKind::Clause(ty::Clause::Trait(trait_pred)) => {
                                // FIXME(rpitit): This will need to be fixed when we move to associated types
                                assert!(matches!(
                                    *trait_pred.trait_ref.self_ty().kind(),
                                    ty::Alias(_, ty::AliasTy { def_id, substs, .. })
                                    if def_id == rpit_def_id && substs == substs
                                ));
                                ty::PredicateKind::Clause(ty::Clause::Trait(
                                    trait_pred.with_self_ty(self.tcx, ty),
                                ))
                            }
                            ty::PredicateKind::Clause(ty::Clause::Projection(mut proj_pred)) => {
                                assert!(matches!(
                                    *proj_pred.projection_ty.self_ty().kind(),
                                    ty::Alias(_, ty::AliasTy { def_id, substs, .. })
                                    if def_id == rpit_def_id && substs == substs
                                ));
                                proj_pred = proj_pred.with_self_ty(self.tcx, ty);
                                ty::PredicateKind::Clause(ty::Clause::Projection(proj_pred))
                            }
                            _ => continue,
                        });
                        if !self.predicate_must_hold_modulo_regions(&Obligation::new(
                            self.tcx,
                            ObligationCause::misc(span, self.body_id),
                            self.param_env,
                            pred,
                        )) {
                            return None;
                        }
                    }
                }

                Some(span)
            }
            _ => None,
        }
    }
}

fn arms_contain_ref_bindings<'tcx>(arms: &'tcx [hir::Arm<'tcx>]) -> Option<hir::Mutability> {
    arms.iter().filter_map(|a| a.pat.contains_explicit_ref_binding()).max()
}
