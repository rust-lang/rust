use std::sync::Arc;

use thin_vec::thin_vec;

use crate::{LoweringContext, ResolverAstLoweringExt};

impl<'hir, R: ResolverAstLoweringExt<'hir>> LoweringContext<'_, 'hir, R> {
    /// Lowered contracts are guarded with the `contract_checks` compiler flag,
    /// i.e. the flag turns into a boolean guard in the lowered HIR. The reason
    /// for not eliminating the contract code entirely when the `contract_checks`
    /// flag is disabled is so that contracts can be type checked, even when
    /// they are disabled, which avoids them becoming stale (i.e. out of sync
    /// with the codebase) over time.
    ///
    /// The optimiser should be able to eliminate all contract code guarded
    /// by `if false`, leaving the original body intact when runtime contract
    /// checks are disabled.
    pub(super) fn lower_contract(
        &mut self,
        body: impl FnOnce(&mut Self) -> rustc_hir::Expr<'hir>,
        contract: &rustc_ast::FnContract,
    ) -> rustc_hir::Expr<'hir> {
        // The order in which things are lowered is important! I.e to
        // refer to variables in contract_decls from postcond/precond,
        // we must lower it first!
        let contract_decls = self.lower_decls(contract);

        match (&contract.requires, &contract.ensures) {
            (Some(req), Some(ens)) => {
                // Lower the fn contract, which turns:
                //
                // { body }
                //
                // into:
                //
                // let __postcond = if contract_checks {
                //     let __ensures_builder = || {
                //         CONTRACT_DECLARATIONS;
                //         contract_check_requires(|| PRECOND);
                //         build_check_ensures(Some(|| { POSTCOND_DECLS; |ret_val| POSTCOND }))
                //     };
                //     contract_check_requires_and_build_ensures(__ensures_builder)
                // } else {
                //     None
                // };
                // {
                //     let ret = { body };
                //
                //     if contract_checks {
                //         contract_check_ensures(__postcond, ret)
                //     } else {
                //         ret
                //     }
                // }

                let lowered_req = self.lower_expr_mut(&req);
                let precond = self.lower_precond(lowered_req);
                let postcond_checker = self.lower_postcond_checker(ens);

                let contract_check = self.lower_contract_check_with_postcond(
                    contract_decls,
                    Some(precond),
                    postcond_checker,
                );

                let wrapped_body =
                    self.wrap_body_with_contract_check(body, contract_check, postcond_checker.span);
                self.expr_block(wrapped_body)
            }
            (None, Some(ens)) => {
                // Lower the fn contract, which turns:
                //
                // { body }
                //
                // into:
                //
                // let __postcond = if contract_checks {
                //     let __ensures_builder = || {
                //         CONTRACT_DECLARATIONS;
                //         build_check_ensures(Some(|| { POSTCOND_DECLS; |ret_val| POSTCOND }))
                //     };
                //     contract_check_requires_and_build_ensures(__ensures_builder)
                // } else {
                //     None
                // };
                // {
                //     let ret = { body };
                //
                //     if contract_checks {
                //         contract_check_ensures(__postcond, ret)
                //     } else {
                //         ret
                //     }
                // }
                let postcond_checker = self.lower_postcond_checker(ens);
                let contract_check =
                    self.lower_contract_check_with_postcond(contract_decls, None, postcond_checker);

                let wrapped_body =
                    self.wrap_body_with_contract_check(body, contract_check, postcond_checker.span);
                self.expr_block(wrapped_body)
            }
            (Some(req), None) => {
                // Lower the fn contract, which turns:
                //
                // { body }
                //
                // into:
                //
                // {
                //      if contracts_checks {
                //          contract_requires(|| { CONTRACT_DECLARATIONS; PRECOND });
                //      }
                //      body
                // }
                let lowered_req = self.lower_expr(&req);
                let precond = self.block_decls_with_precond(contract_decls, lowered_req);
                let precond_check = self.lower_contract_check_just_precond(precond);

                let body = self.arena.alloc(body(self));

                // Flatten the body into precond check, then body.
                let wrapped_body = self.block_all(
                    body.span,
                    self.arena.alloc_from_iter([precond_check].into_iter()),
                    Some(body),
                );
                self.expr_block(wrapped_body)
            }
            (None, None) => body(self),
        }
    }

    fn lower_decls(&mut self, contract: &rustc_ast::FnContract) -> &'hir [rustc_hir::Stmt<'hir>] {
        let (decls, decls_tail) = self.lower_stmts(&contract.declarations);

        if let Some(e) = decls_tail {
            // include the tail expression in the declaration statements
            let tail = self.stmt_expr(e.span, *e);
            self.arena.alloc_from_iter(decls.into_iter().map(|d| *d).chain([tail].into_iter()))
        } else {
            decls
        }
    }

    /// Lower the precondition check intrinsic.
    fn lower_precond(&mut self, req: rustc_hir::Expr<'hir>) -> rustc_hir::Stmt<'hir> {
        let req_span = self.mark_span_with_reason(
            rustc_span::DesugaringKind::Contract,
            req.span,
            Some(Arc::clone(&self.allow_contracts)),
        );
        let req_closure = self.expr_closure(req_span, req);
        let precond = self.expr_call_lang_item_fn_mut(
            req_span,
            rustc_hir::LangItem::ContractCheckRequires,
            &*arena_vec![self; req_closure],
        );
        self.stmt_expr(req.span, precond)
    }

    fn lower_postcond_checker(
        &mut self,
        ens: &Box<rustc_ast::Expr>,
    ) -> &'hir rustc_hir::Expr<'hir> {
        let ens_span = self.lower_span(ens.span);
        let ens_span = self.mark_span_with_reason(
            rustc_span::DesugaringKind::Contract,
            ens_span,
            Some(Arc::clone(&self.allow_contracts)),
        );
        let lowered_ens = self.lower_expr_mut(&ens);
        let ens_closure = self.expr_closure(ens_span, lowered_ens);
        self.expr_call_lang_item_fn(
            ens_span,
            rustc_hir::LangItem::ContractBuildCheckEnsures,
            &*arena_vec![self; ens_closure],
        )
    }

    fn block_decls_with_precond(
        &mut self,
        contract_decls: &'hir [rustc_hir::Stmt<'_>],
        lowered_req: &'hir rustc_hir::Expr<'_>,
    ) -> rustc_hir::Stmt<'hir> {
        let req_span = span_of_stmts(contract_decls, lowered_req.span);

        let precond_stmts = self.block_all(req_span, contract_decls, Some(lowered_req));
        let precond_stmts = self.expr_block(precond_stmts);
        self.lower_precond(precond_stmts)
    }

    fn lower_contract_check_just_precond(
        &mut self,
        precond: rustc_hir::Stmt<'hir>,
    ) -> rustc_hir::Stmt<'hir> {
        let span = precond.span;
        let then_block_stmts = self.block_all(span, &*arena_vec![self; precond], None);
        let then_block = self.arena.alloc(self.expr_block(&then_block_stmts));

        let precond_check = rustc_hir::ExprKind::If(
            self.arena.alloc(self.expr_bool_literal(span, self.tcx.sess.contract_checks())),
            then_block,
            None,
        );

        let precond_check = self.expr(span, precond_check);
        self.stmt_expr(span, precond_check)
    }

    fn lower_contract_check_with_postcond(
        &mut self,
        contract_decls: &'hir [rustc_hir::Stmt<'hir>],
        precond: Option<rustc_hir::Stmt<'hir>>,
        postcond_checker: &'hir rustc_hir::Expr<'hir>,
    ) -> &'hir rustc_hir::Expr<'hir> {
        let stmts = self
            .arena
            .alloc_from_iter(contract_decls.into_iter().map(|d| *d).chain(precond.into_iter()));

        let span = self.contract_check_with_postcond_span(stmts, postcond_checker);

        let then_block = self.contract_check_with_postcond_block(stmts, postcond_checker, span);
        let else_block = self.option_none_block(span);

        let contract_check = rustc_hir::ExprKind::If(
            self.arena.alloc(self.expr_bool_literal(span, self.tcx.sess.contract_checks())),
            then_block,
            Some(else_block),
        );
        self.arena.alloc(self.expr(span, contract_check))
    }

    fn contract_check_with_postcond_span(
        &mut self,
        stmts: &mut [rustc_hir::Stmt<'hir>],
        postcond_checker: &rustc_hir::Expr<'_>,
    ) -> rustc_span::Span {
        // For error diagnostics, span is set to decls + precondition, because
        // those will determine the well-typedness of the __ensures_builder
        // closure. postcond_checker is already type-checked as part of the
        // call to build_check_ensures.
        let span =
            span_of_stmts(stmts, stmts.last().map(|s| s.span).unwrap_or(postcond_checker.span));
        self.mark_span_with_reason(
            rustc_span::DesugaringKind::Contract,
            span,
            Some(Arc::clone(&self.allow_contracts)),
        )
    }

    fn contract_check_with_postcond_block(
        &mut self,
        stmts: &'hir mut [rustc_hir::Stmt<'hir>],
        postcond_checker: &'hir rustc_hir::Expr<'_>,
        span: rustc_span::Span,
    ) -> &'hir mut rustc_hir::Expr<'hir> {
        let (builder_decl, builder_ident_expr) =
            self.contract_check_with_postcond_builder(stmts, postcond_checker, span);

        let build_postcond_call = self.expr_call_lang_item_fn(
            span,
            rustc_hir::LangItem::ContractCheckRequiresAndBuildEnsures,
            &*arena_vec![self; *builder_ident_expr],
        );
        let block_stmts =
            self.block_all(span, arena_vec![self; builder_decl], Some(build_postcond_call));
        self.arena.alloc(self.expr_block(block_stmts))
    }

    fn contract_check_with_postcond_builder(
        &mut self,
        stmts: &'hir mut [rustc_hir::Stmt<'hir>],
        postcond_checker: &'hir rustc_hir::Expr<'_>,
        span: rustc_span::Span,
    ) -> (rustc_hir::Stmt<'hir>, &'hir rustc_hir::Expr<'hir>) {
        let block_closure =
            self.contract_check_with_postcond_builder_closure(stmts, postcond_checker, span);

        let (builder_ident, builder_hir_id, builder_decl) =
            self.bind_expression(block_closure, span, "__ensures_builder");
        let builder_ident_expr = self.expr_ident(span, builder_ident, builder_hir_id);

        (builder_decl, builder_ident_expr)
    }

    fn contract_check_with_postcond_builder_closure(
        &mut self,
        stmts: &'hir mut [rustc_hir::Stmt<'hir>],
        postcond_checker: &'hir rustc_hir::Expr<'_>,
        span: rustc_span::Span,
    ) -> &'hir mut rustc_hir::Expr<'hir> {
        let stmts = self.block_all(span, stmts, Some(postcond_checker));
        let stmts = self.expr_block(stmts);
        let closure = self.expr_closure(span, stmts);
        self.arena.alloc(closure)
    }

    fn option_none_block(&mut self, span: rustc_span::Span) -> &'hir mut rustc_hir::Expr<'hir> {
        let none_expr = self.arena.alloc(self.expr_enum_variant_lang_item(
            span,
            rustc_hir::lang_items::LangItem::OptionNone,
            Default::default(),
        ));
        let else_block = self.block_expr(none_expr);
        self.arena.alloc(self.expr_block(else_block))
    }

    fn wrap_body_with_contract_check(
        &mut self,
        body: impl FnOnce(&mut Self) -> rustc_hir::Expr<'hir>,
        contract_check: &'hir rustc_hir::Expr<'hir>,
        postcond_span: rustc_span::Span,
    ) -> &'hir rustc_hir::Block<'hir> {
        let (check_ident, check_hir_id, postcond_decl) =
            self.bind_expression(contract_check, postcond_span, "__ensures_checker");

        // Install contract_ensures so we will intercept `return` statements,
        // then lower the body.
        self.contract_ensures = Some((postcond_span, check_ident, check_hir_id));
        let body = self.arena.alloc(body(self));

        // Finally, inject an ensures check on the implicit return of the body.
        let body = self.inject_ensures_check(body, postcond_span, check_ident, check_hir_id);

        // Flatten the body into precond, then postcond, then wrapped body.
        let wrapped_body = self.block_all(
            body.span,
            self.arena.alloc_from_iter([postcond_decl].into_iter()),
            Some(body),
        );
        wrapped_body
    }

    fn bind_expression(
        &mut self,
        expr: &'hir rustc_hir::Expr<'hir>,
        span: rustc_span::Span,
        var_name: &str,
    ) -> (rustc_span::Ident, rustc_hir::HirId, rustc_hir::Stmt<'hir>) {
        let ident = rustc_span::Ident::from_str_and_span(var_name, span);
        let (pat, hir_id) =
            self.pat_ident_binding_mode_mut(span, ident, rustc_hir::BindingMode::NONE);

        let decl = self.stmt_let_pat(
            None,
            span,
            Some(expr),
            self.arena.alloc(pat),
            rustc_hir::LocalSource::Contract,
        );
        (ident, hir_id, decl)
    }

    /// Create an `ExprKind::Ret` that is optionally wrapped by a call to check
    /// a contract ensures clause, if it exists.
    pub(super) fn checked_return(
        &mut self,
        opt_expr: Option<&'hir rustc_hir::Expr<'hir>>,
    ) -> rustc_hir::ExprKind<'hir> {
        let checked_ret =
            if let Some((check_span, check_ident, check_hir_id)) = self.contract_ensures {
                let expr = opt_expr.unwrap_or_else(|| self.expr_unit(check_span));
                Some(self.inject_ensures_check(expr, check_span, check_ident, check_hir_id))
            } else {
                opt_expr
            };
        rustc_hir::ExprKind::Ret(checked_ret)
    }

    /// Wraps an expression with a call to the ensures check before it gets returned.
    pub(super) fn inject_ensures_check(
        &mut self,
        expr: &'hir rustc_hir::Expr<'hir>,
        span: rustc_span::Span,
        cond_ident: rustc_span::Ident,
        cond_hir_id: rustc_hir::HirId,
    ) -> &'hir rustc_hir::Expr<'hir> {
        // {
        //     let ret = { body };
        //
        //     if contract_checks {
        //         contract_check_ensures(__postcond, ret)
        //     } else {
        //         ret
        //     }
        // }
        let (ret_ident, ret_hir_id, ret_stmt) = self.bind_expression(expr, span, "__ret");
        let ret = self.expr_ident(span, ret_ident, ret_hir_id);

        let cond_fn = self.expr_ident(span, cond_ident, cond_hir_id);
        let contract_check = self.expr_call_lang_item_fn_mut(
            span,
            rustc_hir::LangItem::ContractCheckEnsures,
            arena_vec![self; *cond_fn, *ret],
        );
        let contract_check = self.arena.alloc(contract_check);
        let call_expr = self.block_expr_block(contract_check);

        // same ident can't be used in 2 places, so we create a new one for the
        // else branch
        let ret = self.expr_ident(span, ret_ident, ret_hir_id);
        let ret_block = self.block_expr_block(ret);

        let contracts_enabled: rustc_hir::Expr<'_> =
            self.expr_bool_literal(span, self.tcx.sess.contract_checks());
        let contract_check = self.arena.alloc(self.expr(
            span,
            rustc_hir::ExprKind::If(
                self.arena.alloc(contracts_enabled),
                call_expr,
                Some(ret_block),
            ),
        ));

        let attrs: rustc_ast::AttrVec = thin_vec![self.unreachable_code_attr(span)];
        self.lower_attrs(contract_check.hir_id, &attrs, span, rustc_hir::Target::Expression);

        let ret_block = self.block_all(span, arena_vec![self; ret_stmt], Some(contract_check));
        self.arena.alloc(self.expr_block(self.arena.alloc(ret_block)))
    }
}

fn span_of_stmts<'hir>(
    stmts: &'hir [rustc_hir::Stmt<'_>],
    default_span: rustc_span::Span,
) -> rustc_span::Span {
    match stmts {
        [] => default_span,
        [first, ..] => first.span.to(default_span),
    }
}
