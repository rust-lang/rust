use crate::LoweringContext;

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    pub(super) fn lower_contract(
        &mut self,
        body: impl FnOnce(&mut Self) -> rustc_hir::Expr<'hir>,
        contract: &rustc_ast::FnContract,
    ) -> rustc_hir::Expr<'hir> {
        // The order in which things are lowered is important! I.e to
        // refer to variables in contract_decls from postcond/precond,
        // we must lower it first!
        let contract_decls = self.lower_stmts(&contract.declarations).0;

        match (&contract.requires, &contract.ensures) {
            (Some(req), Some(ens)) => {
                // Lower the fn contract, which turns:
                //
                // { body }
                //
                // into:
                //
                // {
                //      let __postcond = if contracts_checks() {
                //          CONTRACT_DECLARATIONS;
                //          contract_check_requires(PRECOND);
                //          Some(|ret_val| POSTCOND)
                //      } else {
                //          None
                //      };
                //      contract_check_ensures(__postcond, { body })
                // }

                let precond = self.lower_precond(req);
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
                // {
                //      let __postcond = if contracts_check() {
                //          CONTRACT_DECLARATIONS;
                //          Some(|ret_val| POSTCOND)
                //      } else {
                //          None
                //      };
                //      contract_check_ensures(__postcond, { body })
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
                //      if contracts_check() {
                //          CONTRACT_DECLARATIONS;
                //          contract_requires(PRECOND);
                //      }
                //      body
                // }
                let precond = self.lower_precond(req);
                let precond_check = self.lower_contract_check_just_precond(contract_decls, precond);

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

    /// Lower the precondition check intrinsic.
    fn lower_precond(&mut self, req: &Box<rustc_ast::Expr>) -> rustc_hir::Stmt<'hir> {
        let lowered_req = self.lower_expr_mut(&req);
        let req_span = self.mark_span_with_reason(
            rustc_span::DesugaringKind::Contract,
            lowered_req.span,
            None,
        );
        let precond = self.expr_call_lang_item_fn_mut(
            req_span,
            rustc_hir::LangItem::ContractCheckRequires,
            &*arena_vec![self; lowered_req],
        );
        self.stmt_expr(req.span, precond)
    }

    fn lower_postcond_checker(
        &mut self,
        ens: &Box<rustc_ast::Expr>,
    ) -> &'hir rustc_hir::Expr<'hir> {
        let ens_span = self.lower_span(ens.span);
        let ens_span =
            self.mark_span_with_reason(rustc_span::DesugaringKind::Contract, ens_span, None);
        let lowered_ens = self.lower_expr_mut(&ens);
        self.expr_call_lang_item_fn(
            ens_span,
            rustc_hir::LangItem::ContractBuildCheckEnsures,
            &*arena_vec![self; lowered_ens],
        )
    }

    fn lower_contract_check_just_precond(
        &mut self,
        contract_decls: &'hir [rustc_hir::Stmt<'hir>],
        precond: rustc_hir::Stmt<'hir>,
    ) -> rustc_hir::Stmt<'hir> {
        let stmts = self
            .arena
            .alloc_from_iter(contract_decls.into_iter().map(|d| *d).chain([precond].into_iter()));

        let then_block_stmts = self.block_all(precond.span, stmts, None);
        let then_block = self.arena.alloc(self.expr_block(&then_block_stmts));

        let precond_check = rustc_hir::ExprKind::If(
            self.expr_call_lang_item_fn(
                precond.span,
                rustc_hir::LangItem::ContractChecks,
                Default::default(),
            ),
            then_block,
            None,
        );

        let precond_check = self.expr(precond.span, precond_check);
        self.stmt_expr(precond.span, precond_check)
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
        let span = match precond {
            Some(precond) => precond.span,
            None => postcond_checker.span,
        };

        let postcond_checker = self.arena.alloc(self.expr_enum_variant_lang_item(
            postcond_checker.span,
            rustc_hir::lang_items::LangItem::OptionSome,
            &*arena_vec![self; *postcond_checker],
        ));
        let then_block_stmts = self.block_all(span, stmts, Some(postcond_checker));
        let then_block = self.arena.alloc(self.expr_block(&then_block_stmts));

        let none_expr = self.arena.alloc(self.expr_enum_variant_lang_item(
            postcond_checker.span,
            rustc_hir::lang_items::LangItem::OptionNone,
            Default::default(),
        ));
        let else_block = self.block_expr(none_expr);
        let else_block = self.arena.alloc(self.expr_block(else_block));

        let contract_check = rustc_hir::ExprKind::If(
            self.expr_call_lang_item_fn(
                span,
                rustc_hir::LangItem::ContractChecks,
                Default::default(),
            ),
            then_block,
            Some(else_block),
        );
        self.arena.alloc(self.expr(span, contract_check))
    }

    fn wrap_body_with_contract_check(
        &mut self,
        body: impl FnOnce(&mut Self) -> rustc_hir::Expr<'hir>,
        contract_check: &'hir rustc_hir::Expr<'hir>,
        postcond_span: rustc_span::Span,
    ) -> &'hir rustc_hir::Block<'hir> {
        let check_ident: rustc_span::Ident =
            rustc_span::Ident::from_str_and_span("__ensures_checker", postcond_span);
        let (check_hir_id, postcond_decl) = {
            // Set up the postcondition `let` statement.
            let (checker_pat, check_hir_id) = self.pat_ident_binding_mode_mut(
                postcond_span,
                check_ident,
                rustc_hir::BindingMode::NONE,
            );
            (
                check_hir_id,
                self.stmt_let_pat(
                    None,
                    postcond_span,
                    Some(contract_check),
                    self.arena.alloc(checker_pat),
                    rustc_hir::LocalSource::Contract,
                ),
            )
        };

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
        let cond_fn = self.expr_ident(span, cond_ident, cond_hir_id);
        let call_expr = self.expr_call_lang_item_fn_mut(
            span,
            rustc_hir::LangItem::ContractCheckEnsures,
            arena_vec![self; *cond_fn, *expr],
        );
        self.arena.alloc(call_expr)
    }
}
