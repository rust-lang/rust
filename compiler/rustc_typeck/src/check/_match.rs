use crate::check::coercion::{AsCoercionSite, CoerceMany};
use crate::check::{Diverges, Expectation, FnCtxt, Needs};
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_hir::{self as hir, ExprKind};
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::traits::Obligation;
use rustc_middle::ty::{self, ToPredicate, Ty, TyS};
use rustc_span::{MultiSpan, Span};
use rustc_trait_selection::opaque_types::InferCtxtExt as _;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_trait_selection::traits::{
    IfExpressionCause, MatchExpressionArmCause, ObligationCause, ObligationCauseCode,
    StatementAsExpression,
};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    #[instrument(skip(self), level = "debug")]
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
        for arm in arms {
            self.check_pat_top(&arm.pat, scrutinee_ty, Some(scrut.span), true);
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
                        self.check_expr_has_type_or_error(e, tcx.types.bool, |_| {});
                    }
                    hir::Guard::IfLet(pat, e) => {
                        let scrutinee_ty = self.demand_scrutinee_type(
                            e,
                            pat.contains_explicit_ref_binding(),
                            false,
                        );
                        self.check_pat_top(&pat, scrutinee_ty, None, true);
                    }
                };
            }

            self.diverges.set(Diverges::Maybe);

            let arm_ty = self.check_expr_with_expectation(&arm.body, expected);
            all_arms_diverge &= self.diverges.get();

            let opt_suggest_box_span =
                self.opt_suggest_box_span(arm.body.span, arm_ty, orig_expected);

            let (arm_span, semi_span) =
                self.get_appropriate_arm_semicolon_removal_span(&arms, i, prior_arm_ty, arm_ty);
            let (span, code) = match i {
                // The reason for the first arm to fail is not that the match arms diverge,
                // but rather that there's a prior obligation that doesn't hold.
                0 => (arm_span, ObligationCauseCode::BlockTailExpression(arm.body.hir_id)),
                _ => (
                    expr.span,
                    ObligationCauseCode::MatchExpressionArm(Box::new(MatchExpressionArmCause {
                        arm_span,
                        scrut_span: scrut.span,
                        semi_span,
                        source: match_src,
                        prior_arms: other_arms.clone(),
                        last_ty: prior_arm_ty.unwrap(),
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
                Some(&mut |err: &mut DiagnosticBuilder<'_>| {
                    let can_coerce_to_return_ty = match self.ret_coercion.as_ref() {
                        Some(ret_coercion) if self.in_tail_expr => {
                            let ret_ty = ret_coercion.borrow().expected_ty();
                            let ret_ty = self.inh.infcx.shallow_resolve(ret_ty);
                            self.can_coerce(arm_ty, ret_ty)
                                && prior_arm_ty.map_or(true, |t| self.can_coerce(t, ret_ty))
                                // The match arms need to unify for the case of `impl Trait`.
                                && !matches!(ret_ty.kind(), ty::Opaque(..))
                        }
                        _ => false,
                    };
                    if let (Expectation::IsLast(stmt), Some(ret), true) =
                        (orig_expected, self.ret_type_span, can_coerce_to_return_ty)
                    {
                        let semi_span = expr.span.shrink_to_hi().with_hi(stmt.hi());
                        let mut ret_span: MultiSpan = semi_span.into();
                        ret_span.push_span_label(
                            expr.span,
                            "this could be implicitly returned but it is a statement, not a \
                                tail expression"
                                .to_owned(),
                        );
                        ret_span.push_span_label(
                            ret,
                            "the `match` arms can conform to this return type".to_owned(),
                        );
                        ret_span.push_span_label(
                            semi_span,
                            "the `match` is a statement because of this semicolon, consider \
                                removing it"
                                .to_owned(),
                        );
                        err.span_note(
                            ret_span,
                            "you might have meant to return the `match` expression",
                        );
                        err.tool_only_span_suggestion(
                            semi_span,
                            "remove this semicolon",
                            String::new(),
                            Applicability::MaybeIncorrect,
                        );
                    }
                }),
                false,
            );

            other_arms.push(arm_span);
            if other_arms.len() > 5 {
                other_arms.remove(0);
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

    fn get_appropriate_arm_semicolon_removal_span(
        &self,
        arms: &'tcx [hir::Arm<'tcx>],
        i: usize,
        prior_arm_ty: Option<Ty<'tcx>>,
        arm_ty: Ty<'tcx>,
    ) -> (Span, Option<(Span, StatementAsExpression)>) {
        let arm = &arms[i];
        let (arm_span, mut semi_span) = if let hir::ExprKind::Block(blk, _) = &arm.body.kind {
            self.find_block_span(blk, prior_arm_ty)
        } else {
            (arm.body.span, None)
        };
        if semi_span.is_none() && i > 0 {
            if let hir::ExprKind::Block(blk, _) = &arms[i - 1].body.kind {
                let (_, semi_span_prev) = self.find_block_span(blk, Some(arm_ty));
                semi_span = semi_span_prev;
            }
        }
        (arm_span, semi_span)
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

    fn maybe_get_coercion_reason(&self, hir_id: hir::HirId, sp: Span) -> Option<(Span, String)> {
        let node = {
            let rslt = self.tcx.hir().get_parent_node(self.tcx.hir().get_parent_node(hir_id));
            self.tcx.hir().get(rslt)
        };
        if let hir::Node::Block(block) = node {
            // check that the body's parent is an fn
            let parent = self
                .tcx
                .hir()
                .get(self.tcx.hir().get_parent_node(self.tcx.hir().get_parent_node(block.hir_id)));
            if let (Some(expr), hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(..), .. })) =
                (&block.expr, parent)
            {
                // check that the `if` expr without `else` is the fn body's expr
                if expr.span == sp {
                    return self.get_fn_decl(hir_id).and_then(|(fn_decl, _)| {
                        let span = fn_decl.output.span();
                        let snippet = self.tcx.sess.source_map().span_to_snippet(span).ok()?;
                        Some((span, format!("expected `{}` because of this return type", snippet)))
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
        then_expr: &'tcx hir::Expr<'tcx>,
        else_expr: &'tcx hir::Expr<'tcx>,
        then_ty: Ty<'tcx>,
        else_ty: Ty<'tcx>,
        opt_suggest_box_span: Option<Span>,
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
            let (error_sp, semi_sp) = self.find_block_span(block, Some(then_ty));
            remove_semicolon = semi_sp;
            if block.expr.is_none() && block.stmts.is_empty() {
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
                    outer_sp = Some(self.tcx.sess.source_map().guess_head_span(span));
                }
            }
            error_sp
        } else {
            // shouldn't happen unless the parser has done something weird
            else_expr.span
        };

        // Compute `Span` of `then` part of `if`-expression.
        let then_sp = if let ExprKind::Block(block, _) = &then_expr.kind {
            let (then_sp, semi_sp) = self.find_block_span(block, Some(else_ty));
            remove_semicolon = remove_semicolon.or(semi_sp);
            if block.expr.is_none() && block.stmts.is_empty() {
                outer_sp = None; // same as in `error_sp`; cleanup output
            }
            then_sp
        } else {
            // shouldn't happen unless the parser has done something weird
            then_expr.span
        };

        // Finally construct the cause:
        self.cause(
            error_sp,
            ObligationCauseCode::IfExpression(Box::new(IfExpressionCause {
                then: then_sp,
                else_sp: error_sp,
                outer: outer_sp,
                semicolon: remove_semicolon,
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

    fn find_block_span(
        &self,
        block: &'tcx hir::Block<'tcx>,
        expected_ty: Option<Ty<'tcx>>,
    ) -> (Span, Option<(Span, StatementAsExpression)>) {
        if let Some(expr) = &block.expr {
            (expr.span, None)
        } else if let Some(stmt) = block.stmts.last() {
            // possibly incorrect trailing `;` in the else arm
            (stmt.span, expected_ty.and_then(|ty| self.could_remove_semicolon(block, ty)))
        } else {
            // empty block; point at its entirety
            (block.span, None)
        }
    }

    // When we have a `match` as a tail expression in a `fn` with a returned `impl Trait`
    // we check if the different arms would work with boxed trait objects instead and
    // provide a structured suggestion in that case.
    pub(crate) fn opt_suggest_box_span(
        &self,
        span: Span,
        outer_ty: &'tcx TyS<'tcx>,
        orig_expected: Expectation<'tcx>,
    ) -> Option<Span> {
        match (orig_expected, self.ret_coercion_impl_trait.map(|ty| (self.body_id.owner, ty))) {
            (Expectation::ExpectHasType(expected), Some((_id, ty)))
                if self.in_tail_expr && self.can_coerce(outer_ty, expected) =>
            {
                let impl_trait_ret_ty =
                    self.infcx.instantiate_opaque_types(self.body_id, self.param_env, ty, span);
                assert!(
                    impl_trait_ret_ty.obligations.is_empty(),
                    "we should never get new obligations here"
                );
                let obligations = self.fulfillment_cx.borrow().pending_obligations();
                let mut suggest_box = !obligations.is_empty();
                for o in obligations {
                    match o.predicate.kind().skip_binder() {
                        ty::PredicateKind::Trait(t) => {
                            let pred =
                                ty::Binder::dummy(ty::PredicateKind::Trait(ty::TraitPredicate {
                                    trait_ref: ty::TraitRef {
                                        def_id: t.def_id(),
                                        substs: self.infcx.tcx.mk_substs_trait(outer_ty, &[]),
                                    },
                                    constness: t.constness,
                                }));
                            let obl = Obligation::new(
                                o.cause.clone(),
                                self.param_env,
                                pred.to_predicate(self.infcx.tcx),
                            );
                            suggest_box &= self.infcx.predicate_must_hold_modulo_regions(&obl);
                            if !suggest_box {
                                // We've encountered some obligation that didn't hold, so the
                                // return expression can't just be boxed. We don't need to
                                // evaluate the rest of the obligations.
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                // If all the obligations hold (or there are no obligations) the tail expression
                // we can suggest to return a boxed trait object instead of an opaque type.
                if suggest_box { self.ret_type_span } else { None }
            }
            _ => None,
        }
    }
}

fn arms_contain_ref_bindings(arms: &'tcx [hir::Arm<'tcx>]) -> Option<hir::Mutability> {
    arms.iter().filter_map(|a| a.pat.contains_explicit_ref_binding()).max_by_key(|m| match *m {
        hir::Mutability::Mut => 1,
        hir::Mutability::Not => 0,
    })
}
