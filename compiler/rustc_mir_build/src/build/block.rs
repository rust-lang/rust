use crate::build::matches::ArmHasGuard;
use crate::build::ForGuard::OutsideGuard;
use crate::build::{BlockAnd, BlockAndExtension, BlockFrame, Builder};
use crate::thir::*;
use rustc_middle::mir::*;
use rustc_session::lint::builtin::UNSAFE_OP_IN_UNSAFE_FN;
use rustc_session::lint::Level;
use rustc_span::Span;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    crate fn ast_block(
        &mut self,
        destination: Place<'tcx>,
        block: BasicBlock,
        ast_block: &Block,
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
        } = *ast_block;
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
                        ))
                    })
                } else {
                    this.ast_block_stmts(destination, block, span, &stmts, expr, safety_mode)
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
        let outer_push_unsafe_count = this.push_unsafe_count;
        let outer_unpushed_unsafe = this.unpushed_unsafe;
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
                    ref pattern,
                    initializer,
                    lint_level,
                } => {
                    let ignores_expr_result = matches!(*pattern.kind, PatKind::Wild);
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
                                            ArmHasGuard(false),
                                            Some((None, initializer_span)),
                                        );
                                        this.expr_into_pattern(block, pattern.clone(), init)
                                    })
                                }
                            )
                        );
                    } else {
                        let scope = (*init_scope, source_info);
                        unpack!(this.in_scope(scope, *lint_level, |this| {
                            this.declare_bindings(
                                visibility_scope,
                                remainder_span,
                                pattern,
                                ArmHasGuard(false),
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
                }
            }

            let popped = this.block_context.pop();
            assert!(popped.map_or(false, |bf| bf.is_statement()));
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

            assert!(popped.map_or(false, |bf| bf.is_tail_expr()));
        } else {
            // If a block has no trailing expression, then it is given an implicit return type.
            // This return type is usually `()`, unless the block is diverging, in which case the
            // return type is `!`. For the unit type, we need to actually return the unit, but in
            // the case of `!`, no return value is required, as the block will never return.
            if destination_ty.is_unit() {
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
        this.push_unsafe_count = outer_push_unsafe_count;
        this.unpushed_unsafe = outer_unpushed_unsafe;
        block.unit()
    }

    /// If we are changing the safety mode, create a new source scope
    fn update_source_scope_for_safety_mode(&mut self, span: Span, safety_mode: BlockSafety) {
        debug!("update_source_scope_for({:?}, {:?})", span, safety_mode);
        let new_unsafety = match safety_mode {
            BlockSafety::Safe => None,
            BlockSafety::ExplicitUnsafe(hir_id) => {
                assert_eq!(self.push_unsafe_count, 0);
                match self.unpushed_unsafe {
                    Safety::Safe => {}
                    // no longer treat `unsafe fn`s as `unsafe` contexts (see RFC #2585)
                    Safety::FnUnsafe
                        if self.tcx.lint_level_at_node(UNSAFE_OP_IN_UNSAFE_FN, hir_id).0
                            != Level::Allow => {}
                    _ => return,
                }
                self.unpushed_unsafe = Safety::ExplicitUnsafe(hir_id);
                Some(Safety::ExplicitUnsafe(hir_id))
            }
            BlockSafety::PushUnsafe => {
                self.push_unsafe_count += 1;
                Some(Safety::BuiltinUnsafe)
            }
            BlockSafety::PopUnsafe => {
                self.push_unsafe_count = self
                    .push_unsafe_count
                    .checked_sub(1)
                    .unwrap_or_else(|| span_bug!(span, "unsafe count underflow"));
                if self.push_unsafe_count == 0 { Some(self.unpushed_unsafe) } else { None }
            }
        };

        if let Some(unsafety) = new_unsafety {
            self.source_scope = self.new_source_scope(span, LintLevel::Inherited, Some(unsafety));
        }
    }
}
