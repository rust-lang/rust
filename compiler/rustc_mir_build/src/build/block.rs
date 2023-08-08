use crate::build::ForGuard::OutsideGuard;
use crate::build::{BlockAnd, BlockAndExtension, BlockFrame, Builder};
use rustc_middle::middle::region::Scope;
use rustc_middle::thir::*;
use rustc_middle::{mir::*, ty};
use rustc_span::Span;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    pub(crate) fn ast_block(
        &mut self,
        destination: Place<'tcx>,
        block: BasicBlock,
        ast_block: BlockId,
        source_info: SourceInfo,
    ) -> BlockAnd<()> {
        let Block {
            region_scope,
            opt_destruction_scope,
            span,
            ref stmts,
            expr,
            targeted_by_break,
            safety_mode,
        } = self.thir[ast_block];
        let expr = expr.map(|expr| &self.thir[expr]);
        self.in_opt_scope(opt_destruction_scope.map(|de| (de, source_info)), move |this| {
            this.in_scope((region_scope, source_info), LintLevel::Inherited, move |this| {
                if targeted_by_break {
                    this.in_breakable_scope(None, destination, span, |this| {
                        Some(this.ast_block_stmts(
                            destination,
                            block,
                            span,
                            &stmts,
                            expr,
                            safety_mode,
                            region_scope,
                        ))
                    })
                } else {
                    this.ast_block_stmts(
                        destination,
                        block,
                        span,
                        &stmts,
                        expr,
                        safety_mode,
                        region_scope,
                    )
                }
            })
        })
    }

    fn ast_block_stmts(
        &mut self,
        destination: Place<'tcx>,
        mut block: BasicBlock,
        span: Span,
        stmts: &[StmtId],
        expr: Option<&Expr<'tcx>>,
        safety_mode: BlockSafety,
        region_scope: Scope,
    ) -> BlockAnd<()> {
        let this = self;

        // This convoluted structure is to avoid using recursion as we walk down a list
        // of statements. Basically, the structure we get back is something like:
        //
        //    let x = <init> in {
        //       expr1;
        //       let y = <init> in {
        //           expr2;
        //           expr3;
        //           ...
        //       }
        //    }
        //
        // The let bindings are valid till the end of block so all we have to do is to pop all
        // the let-scopes at the end.
        //
        // First we build all the statements in the block.
        let mut let_scope_stack = Vec::with_capacity(8);
        let outer_source_scope = this.source_scope;
        let outer_in_scope_unsafe = this.in_scope_unsafe;
        // This scope information is kept for breaking out of the parent remainder scope in case
        // one let-else pattern matching fails.
        // By doing so, we can be sure that even temporaries that receive extended lifetime
        // assignments are dropped, too.
        let mut last_remainder_scope = region_scope;
        this.update_source_scope_for_safety_mode(span, safety_mode);

        let source_info = this.source_info(span);
        for stmt in stmts {
            let Stmt { ref kind, opt_destruction_scope } = this.thir[*stmt];
            match kind {
                StmtKind::Expr { scope, expr } => {
                    this.block_context.push(BlockFrame::Statement { ignores_expr_result: true });
                    unpack!(
                        block = this.in_opt_scope(
                            opt_destruction_scope.map(|de| (de, source_info)),
                            |this| {
                                let si = (*scope, source_info);
                                this.in_scope(si, LintLevel::Inherited, |this| {
                                    this.stmt_expr(block, &this.thir[*expr], Some(*scope))
                                })
                            }
                        )
                    );
                }
                StmtKind::Let {
                    remainder_scope,
                    init_scope,
                    pattern,
                    initializer: Some(initializer),
                    lint_level,
                    else_block: Some(else_block),
                    span: _,
                } => {
                    // When lowering the statement `let <pat> = <expr> else { <else> };`,
                    // the `<else>` block is nested in the parent scope enclosing this statement.
                    // That scope is usually either the enclosing block scope,
                    // or the remainder scope of the last statement.
                    // This is to make sure that temporaries instantiated in `<expr>` are dropped
                    // as well.
                    // In addition, even though bindings in `<pat>` only come into scope if
                    // the pattern matching passes, in the MIR building the storages for them
                    // are declared as live any way.
                    // This is similar to `let x;` statements without an initializer expression,
                    // where the value of `x` in this example may or may be assigned,
                    // because the storage for their values may not be live after all due to
                    // failure in pattern matching.
                    // For this reason, we declare those storages as live but we do not schedule
                    // any drop yet- they are scheduled later after the pattern matching.
                    // The generated MIR will have `StorageDead` whenever the control flow breaks out
                    // of the parent scope, regardless of the result of the pattern matching.
                    // However, the drops are inserted in MIR only when the control flow breaks out of
                    // the scope of the remainder scope associated with this `let .. else` statement.
                    // Pictorial explanation of the scope structure:
                    // ┌─────────────────────────────────┐
                    // │  Scope of the enclosing block,  │
                    // │  or the last remainder scope    │
                    // │  ┌───────────────────────────┐  │
                    // │  │  Scope for <else> block   │  │
                    // │  └───────────────────────────┘  │
                    // │  ┌───────────────────────────┐  │
                    // │  │  Remainder scope of       │  │
                    // │  │  this let-else statement  │  │
                    // │  │  ┌─────────────────────┐  │  │
                    // │  │  │ <expr> scope        │  │  │
                    // │  │  └─────────────────────┘  │  │
                    // │  │  extended temporaries in  │  │
                    // │  │  <expr> lives in this     │  │
                    // │  │  scope                    │  │
                    // │  │  ┌─────────────────────┐  │  │
                    // │  │  │ Scopes for the rest │  │  │
                    // │  │  └─────────────────────┘  │  │
                    // │  └───────────────────────────┘  │
                    // └─────────────────────────────────┘
                    // Generated control flow:
                    //          │ let Some(x) = y() else { return; }
                    //          │
                    // ┌────────▼───────┐
                    // │ evaluate y()   │
                    // └────────┬───────┘
                    //          │              ┌────────────────┐
                    // ┌────────▼───────┐      │Drop temporaries│
                    // │Test the pattern├──────►in y()          │
                    // └────────┬───────┘      │because breaking│
                    //          │              │out of <expr>   │
                    // ┌────────▼───────┐      │scope           │
                    // │Move value into │      └───────┬────────┘
                    // │binding x       │              │
                    // └────────┬───────┘      ┌───────▼────────┐
                    //          │              │Drop extended   │
                    // ┌────────▼───────┐      │temporaries in  │
                    // │Drop temporaries│      │<expr> because  │
                    // │in y()          │      │breaking out of │
                    // │because breaking│      │remainder scope │
                    // │out of <expr>   │      └───────┬────────┘
                    // │scope           │              │
                    // └────────┬───────┘      ┌───────▼────────┐
                    //          │              │Enter <else>    ├────────►
                    // ┌────────▼───────┐      │block           │ return;
                    // │Continue...     │      └────────────────┘
                    // └────────────────┘

                    let ignores_expr_result = matches!(pattern.kind, PatKind::Wild);
                    this.block_context.push(BlockFrame::Statement { ignores_expr_result });

                    // Lower the `else` block first because its parent scope is actually
                    // enclosing the rest of the `let .. else ..` parts.
                    let else_block_span = this.thir[*else_block].span;
                    // This place is not really used because this destination place
                    // should never be used to take values at the end of the failure
                    // block.
                    let dummy_place = this.temp(this.tcx.types.never, else_block_span);
                    let failure_entry = this.cfg.start_new_block();
                    let failure_block;
                    unpack!(
                        failure_block = this.ast_block(
                            dummy_place,
                            failure_entry,
                            *else_block,
                            this.source_info(else_block_span),
                        )
                    );
                    this.cfg.terminate(
                        failure_block,
                        this.source_info(else_block_span),
                        TerminatorKind::Unreachable,
                    );

                    // Declare the bindings, which may create a source scope.
                    let remainder_span = remainder_scope.span(this.tcx, this.region_scope_tree);
                    this.push_scope((*remainder_scope, source_info));
                    let_scope_stack.push(remainder_scope);

                    let visibility_scope =
                        Some(this.new_source_scope(remainder_span, LintLevel::Inherited, None));

                    let init = &this.thir[*initializer];
                    let initializer_span = init.span;
                    let failure = unpack!(
                        block = this.in_opt_scope(
                            opt_destruction_scope.map(|de| (de, source_info)),
                            |this| {
                                let scope = (*init_scope, source_info);
                                this.in_scope(scope, *lint_level, |this| {
                                    this.declare_bindings(
                                        visibility_scope,
                                        remainder_span,
                                        pattern,
                                        None,
                                        Some((Some(&destination), initializer_span)),
                                    );
                                    this.visit_primary_bindings(
                                        pattern,
                                        UserTypeProjections::none(),
                                        &mut |this, _, _, _, node, span, _, _| {
                                            this.storage_live_binding(
                                                block,
                                                node,
                                                span,
                                                OutsideGuard,
                                                true,
                                            );
                                        },
                                    );
                                    this.ast_let_else(
                                        block,
                                        init,
                                        initializer_span,
                                        *else_block,
                                        &last_remainder_scope,
                                        pattern,
                                    )
                                })
                            }
                        )
                    );
                    this.cfg.goto(failure, source_info, failure_entry);

                    if let Some(source_scope) = visibility_scope {
                        this.source_scope = source_scope;
                    }
                    last_remainder_scope = *remainder_scope;
                }
                StmtKind::Let { init_scope, initializer: None, else_block: Some(_), .. } => {
                    span_bug!(
                        init_scope.span(this.tcx, this.region_scope_tree),
                        "initializer is missing, but else block is present in this let binding",
                    )
                }
                StmtKind::Let {
                    remainder_scope,
                    init_scope,
                    ref pattern,
                    initializer,
                    lint_level,
                    else_block: None,
                    span: _,
                } => {
                    let ignores_expr_result = matches!(pattern.kind, PatKind::Wild);
                    this.block_context.push(BlockFrame::Statement { ignores_expr_result });

                    // Enter the remainder scope, i.e., the bindings' destruction scope.
                    this.push_scope((*remainder_scope, source_info));
                    let_scope_stack.push(remainder_scope);

                    // Declare the bindings, which may create a source scope.
                    let remainder_span = remainder_scope.span(this.tcx, this.region_scope_tree);

                    let visibility_scope =
                        Some(this.new_source_scope(remainder_span, LintLevel::Inherited, None));

                    // Evaluate the initializer, if present.
                    if let Some(init) = initializer {
                        let init = &this.thir[*init];
                        let initializer_span = init.span;

                        unpack!(
                            block = this.in_opt_scope(
                                opt_destruction_scope.map(|de| (de, source_info)),
                                |this| {
                                    let scope = (*init_scope, source_info);
                                    this.in_scope(scope, *lint_level, |this| {
                                        this.declare_bindings(
                                            visibility_scope,
                                            remainder_span,
                                            pattern,
                                            None,
                                            Some((None, initializer_span)),
                                        );
                                        this.expr_into_pattern(block, &pattern, init)
                                        // irrefutable pattern
                                    })
                                },
                            )
                        )
                    } else {
                        let scope = (*init_scope, source_info);
                        unpack!(this.in_scope(scope, *lint_level, |this| {
                            this.declare_bindings(
                                visibility_scope,
                                remainder_span,
                                pattern,
                                None,
                                None,
                            );
                            block.unit()
                        }));

                        debug!("ast_block_stmts: pattern={:?}", pattern);
                        this.visit_primary_bindings(
                            pattern,
                            UserTypeProjections::none(),
                            &mut |this, _, _, _, node, span, _, _| {
                                this.storage_live_binding(block, node, span, OutsideGuard, true);
                                this.schedule_drop_for_binding(node, span, OutsideGuard);
                            },
                        )
                    }

                    // Enter the visibility scope, after evaluating the initializer.
                    if let Some(source_scope) = visibility_scope {
                        this.source_scope = source_scope;
                    }
                    last_remainder_scope = *remainder_scope;
                }
            }

            let popped = this.block_context.pop();
            assert!(popped.is_some_and(|bf| bf.is_statement()));
        }

        // Then, the block may have an optional trailing expression which is a “return” value
        // of the block, which is stored into `destination`.
        let tcx = this.tcx;
        let destination_ty = destination.ty(&this.local_decls, tcx).ty;
        if let Some(expr) = expr {
            let tail_result_is_ignored =
                destination_ty.is_unit() || this.block_context.currently_ignores_tail_results();
            this.block_context
                .push(BlockFrame::TailExpr { tail_result_is_ignored, span: expr.span });

            unpack!(block = this.expr_into_dest(destination, block, expr));
            let popped = this.block_context.pop();

            assert!(popped.is_some_and(|bf| bf.is_tail_expr()));
        } else {
            // If a block has no trailing expression, then it is given an implicit return type.
            // This return type is usually `()`, unless the block is diverging, in which case the
            // return type is `!`. For the unit type, we need to actually return the unit, but in
            // the case of `!`, no return value is required, as the block will never return.
            // Opaque types of empty bodies also need this unit assignment, in order to infer that their
            // type is actually unit. Otherwise there will be no defining use found in the MIR.
            if destination_ty.is_unit()
                || matches!(destination_ty.kind(), ty::Alias(ty::Opaque, ..))
            {
                // We only want to assign an implicit `()` as the return value of the block if the
                // block does not diverge. (Otherwise, we may try to assign a unit to a `!`-type.)
                this.cfg.push_assign_unit(block, source_info, destination, this.tcx);
            }
        }
        // Finally, we pop all the let scopes before exiting out from the scope of block
        // itself.
        for scope in let_scope_stack.into_iter().rev() {
            unpack!(block = this.pop_scope((*scope, source_info), block));
        }
        // Restore the original source scope.
        this.source_scope = outer_source_scope;
        this.in_scope_unsafe = outer_in_scope_unsafe;
        block.unit()
    }

    /// If we are entering an unsafe block, create a new source scope
    fn update_source_scope_for_safety_mode(&mut self, span: Span, safety_mode: BlockSafety) {
        debug!("update_source_scope_for({:?}, {:?})", span, safety_mode);
        let new_unsafety = match safety_mode {
            BlockSafety::Safe => return,
            BlockSafety::BuiltinUnsafe => Safety::BuiltinUnsafe,
            BlockSafety::ExplicitUnsafe(hir_id) => {
                self.in_scope_unsafe = Safety::ExplicitUnsafe(hir_id);
                Safety::ExplicitUnsafe(hir_id)
            }
        };

        self.source_scope = self.new_source_scope(span, LintLevel::Inherited, Some(new_unsafety));
    }
}
