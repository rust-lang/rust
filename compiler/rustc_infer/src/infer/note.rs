impl<'tcx> TypeErrCtxt<'_, 'tcx> {
    fn note_error_origin(
        &self,
        err: &mut Diagnostic,
        cause: &ObligationCause<'tcx>,
        exp_found: Option<ty::error::ExpectedFound<Ty<'tcx>>>,
        terr: TypeError<'tcx>,
    ) {
        match *cause.code() {
            ObligationCauseCode::Pattern { origin_expr: true, span: Some(span), root_ty } => {
                let ty = self.resolve_vars_if_possible(root_ty);
                if !matches!(ty.kind(), ty::Infer(ty::InferTy::TyVar(_) | ty::InferTy::FreshTy(_)))
                {
                    // don't show type `_`
                    if span.desugaring_kind() == Some(DesugaringKind::ForLoop)
                        && let ty::Adt(def, substs) = ty.kind()
                        && Some(def.did()) == self.tcx.get_diagnostic_item(sym::Option)
                    {
                        err.span_label(span, format!("this is an iterator with items of type `{}`", substs.type_at(0)));
                    } else {
                    err.span_label(span, format!("this expression has type `{}`", ty));
                }
                }
                if let Some(ty::error::ExpectedFound { found, .. }) = exp_found
                    && ty.is_box() && ty.boxed_ty() == found
                    && let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span)
                {
                    err.span_suggestion(
                        span,
                        "consider dereferencing the boxed value",
                        format!("*{}", snippet),
                        Applicability::MachineApplicable,
                    );
                }
            }
            ObligationCauseCode::Pattern { origin_expr: false, span: Some(span), .. } => {
                err.span_label(span, "expected due to this");
            }
            ObligationCauseCode::MatchExpressionArm(box MatchExpressionArmCause {
                arm_block_id,
                arm_span,
                arm_ty,
                prior_arm_block_id,
                prior_arm_span,
                prior_arm_ty,
                source,
                ref prior_arms,
                scrut_hir_id,
                opt_suggest_box_span,
                scrut_span,
                ..
            }) => match source {
                hir::MatchSource::TryDesugar => {
                    if let Some(ty::error::ExpectedFound { expected, .. }) = exp_found {
                        let scrut_expr = self.tcx.hir().expect_expr(scrut_hir_id);
                        let scrut_ty = if let hir::ExprKind::Call(_, args) = &scrut_expr.kind {
                            let arg_expr = args.first().expect("try desugaring call w/out arg");
                            self.typeck_results.as_ref().and_then(|typeck_results| {
                                typeck_results.expr_ty_opt(arg_expr)
                            })
                        } else {
                            bug!("try desugaring w/out call expr as scrutinee");
                        };

                        match scrut_ty {
                            Some(ty) if expected == ty => {
                                let source_map = self.tcx.sess.source_map();
                                err.span_suggestion(
                                    source_map.end_point(cause.span),
                                    "try removing this `?`",
                                    "",
                                    Applicability::MachineApplicable,
                                );
                            }
                            _ => {}
                        }
                    }
                }
                _ => {
                    // `prior_arm_ty` can be `!`, `expected` will have better info when present.
                    let t = self.resolve_vars_if_possible(match exp_found {
                        Some(ty::error::ExpectedFound { expected, .. }) => expected,
                        _ => prior_arm_ty,
                    });
                    let source_map = self.tcx.sess.source_map();
                    let mut any_multiline_arm = source_map.is_multiline(arm_span);
                    if prior_arms.len() <= 4 {
                        for sp in prior_arms {
                            any_multiline_arm |= source_map.is_multiline(*sp);
                            err.span_label(*sp, format!("this is found to be of type `{}`", t));
                        }
                    } else if let Some(sp) = prior_arms.last() {
                        any_multiline_arm |= source_map.is_multiline(*sp);
                        err.span_label(
                            *sp,
                            format!("this and all prior arms are found to be of type `{}`", t),
                        );
                    }
                    let outer_error_span = if any_multiline_arm {
                        // Cover just `match` and the scrutinee expression, not
                        // the entire match body, to reduce diagram noise.
                        cause.span.shrink_to_lo().to(scrut_span)
                    } else {
                        cause.span
                    };
                    let msg = "`match` arms have incompatible types";
                    err.span_label(outer_error_span, msg);
                    self.suggest_remove_semi_or_return_binding(
                        err,
                        prior_arm_block_id,
                        prior_arm_ty,
                        prior_arm_span,
                        arm_block_id,
                        arm_ty,
                        arm_span,
                    );
                    if let Some(ret_sp) = opt_suggest_box_span {
                        // Get return type span and point to it.
                        self.suggest_boxing_for_return_impl_trait(
                            err,
                            ret_sp,
                            prior_arms.iter().chain(std::iter::once(&arm_span)).map(|s| *s),
                        );
                    }
                }
            },
            ObligationCauseCode::IfExpression(box IfExpressionCause {
                then_id,
                else_id,
                then_ty,
                else_ty,
                outer_span,
                opt_suggest_box_span,
            }) => {
                let then_span = self.find_block_span_from_hir_id(then_id);
                let else_span = self.find_block_span_from_hir_id(else_id);
                err.span_label(then_span, "expected because of this");
                if let Some(sp) = outer_span {
                    err.span_label(sp, "`if` and `else` have incompatible types");
                }
                self.suggest_remove_semi_or_return_binding(
                    err,
                    Some(then_id),
                    then_ty,
                    then_span,
                    Some(else_id),
                    else_ty,
                    else_span,
                );
                if let Some(ret_sp) = opt_suggest_box_span {
                    self.suggest_boxing_for_return_impl_trait(
                        err,
                        ret_sp,
                        [then_span, else_span].into_iter(),
                    );
                }
            }
            ObligationCauseCode::LetElse => {
                err.help("try adding a diverging expression, such as `return` or `panic!(..)`");
                err.help("...or use `match` instead of `let...else`");
            }
            _ => {
                if let ObligationCauseCode::BindingObligation(_, span)
                | ObligationCauseCode::ExprBindingObligation(_, span, ..)
                = cause.code().peel_derives()
                    && let TypeError::RegionsPlaceholderMismatch = terr
                {
                    err.span_note( * span,
                    "the lifetime requirement is introduced here");
                }
            }
        }
    }
}

impl<'tcx> InferCtxt<'tcx> {
    /// Given a [`hir::Block`], get the span of its last expression or
    /// statement, peeling off any inner blocks.
    pub fn find_block_span(&self, block: &'tcx hir::Block<'tcx>) -> Span {
        let block = block.innermost_block();
        if let Some(expr) = &block.expr {
            expr.span
        } else if let Some(stmt) = block.stmts.last() {
            // possibly incorrect trailing `;` in the else arm
            stmt.span
        } else {
            // empty block; point at its entirety
            block.span
        }
    }

    /// Given a [`hir::HirId`] for a block, get the span of its last expression
    /// or statement, peeling off any inner blocks.
    pub fn find_block_span_from_hir_id(&self, hir_id: hir::HirId) -> Span {
        match self.tcx.hir().get(hir_id) {
            hir::Node::Block(blk) => self.find_block_span(blk),
            // The parser was in a weird state if either of these happen, but
            // it's better not to panic.
            hir::Node::Expr(e) => e.span,
            _ => rustc_span::DUMMY_SP,
        }
    }
}
