// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use build::{BlockAnd, BlockAndExtension, Builder};
use hair::*;
use rustc::mir::*;
use rustc::hir;
use syntax_pos::Span;

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    pub fn ast_block(&mut self,
                     destination: &Place<'tcx>,
                     block: BasicBlock,
                     ast_block: &'tcx hir::Block,
                     source_info: SourceInfo)
                     -> BlockAnd<()> {
        let Block {
            region_scope,
            opt_destruction_scope,
            span,
            stmts,
            expr,
            targeted_by_break,
            safety_mode
        } =
            self.hir.mirror(ast_block);
        self.in_opt_scope(opt_destruction_scope.map(|de|(de, source_info)), block, move |this| {
            this.in_scope((region_scope, source_info), LintLevel::Inherited, block, move |this| {
                if targeted_by_break {
                    // This is a `break`-able block (currently only `catch { ... }`)
                    let exit_block = this.cfg.start_new_block();
                    let block_exit = this.in_breakable_scope(
                        None, exit_block, destination.clone(), |this| {
                            this.ast_block_stmts(destination, block, span, stmts, expr,
                                                 safety_mode)
                        });
                    this.cfg.terminate(unpack!(block_exit), source_info,
                                       TerminatorKind::Goto { target: exit_block });
                    exit_block.unit()
                } else {
                    this.ast_block_stmts(destination, block, span, stmts, expr,
                                         safety_mode)
                }
            })
        })
    }

    fn ast_block_stmts(&mut self,
                       destination: &Place<'tcx>,
                       mut block: BasicBlock,
                       span: Span,
                       stmts: Vec<StmtRef<'tcx>>,
                       expr: Option<ExprRef<'tcx>>,
                       safety_mode: BlockSafety)
                       -> BlockAnd<()> {
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
        let outer_visibility_scope = this.visibility_scope;
        let outer_push_unsafe_count = this.push_unsafe_count;
        let outer_unpushed_unsafe = this.unpushed_unsafe;
        this.update_visibility_scope_for_safety_mode(span, safety_mode);

        let source_info = this.source_info(span);
        for stmt in stmts {
            let Stmt { kind, opt_destruction_scope } = this.hir.mirror(stmt);
            match kind {
                StmtKind::Expr { scope, expr } => {
                    unpack!(block = this.in_opt_scope(
                        opt_destruction_scope.map(|de|(de, source_info)), block, |this| {
                            let si = (scope, source_info);
                            this.in_scope(si, LintLevel::Inherited, block, |this| {
                                let expr = this.hir.mirror(expr);
                                this.stmt_expr(block, expr)
                            })
                        }));
                }
                StmtKind::Let {
                    remainder_scope,
                    init_scope,
                    pattern,
                    initializer,
                    lint_level
                } => {
                    // Enter the remainder scope, i.e. the bindings' destruction scope.
                    this.push_scope((remainder_scope, source_info));
                    let_scope_stack.push(remainder_scope);

                    // Declare the bindings, which may create a visibility scope.
                    let remainder_span = remainder_scope.span(this.hir.tcx(),
                                                              &this.hir.region_scope_tree);
                    let scope = this.declare_bindings(None, remainder_span, lint_level, &pattern);

                    // Evaluate the initializer, if present.
                    if let Some(init) = initializer {
                        unpack!(block = this.in_opt_scope(
                            opt_destruction_scope.map(|de|(de, source_info)), block, move |this| {
                                let scope = (init_scope, source_info);
                                this.in_scope(scope, lint_level, block, move |this| {
                                    // FIXME #30046                             ^~~~
                                    this.expr_into_pattern(block, pattern, init)
                                })
                            }));
                    } else {
                        this.visit_bindings(&pattern, &mut |this, _, _, node, span, _| {
                            this.storage_live_binding(block, node, span);
                            this.schedule_drop_for_binding(node, span);
                        })
                    }

                    // Enter the visibility scope, after evaluating the initializer.
                    if let Some(visibility_scope) = scope {
                        this.visibility_scope = visibility_scope;
                    }
                }
            }
        }
        // Then, the block may have an optional trailing expression which is a “return” value
        // of the block.
        if let Some(expr) = expr {
            unpack!(block = this.into(destination, block, expr));
        } else {
            this.cfg.push_assign_unit(block, source_info, destination);
        }
        // Finally, we pop all the let scopes before exiting out from the scope of block
        // itself.
        for scope in let_scope_stack.into_iter().rev() {
            unpack!(block = this.pop_scope((scope, source_info), block));
        }
        // Restore the original visibility scope.
        this.visibility_scope = outer_visibility_scope;
        this.push_unsafe_count = outer_push_unsafe_count;
        this.unpushed_unsafe = outer_unpushed_unsafe;
        block.unit()
    }

    /// If we are changing the safety mode, create a new visibility scope
    fn update_visibility_scope_for_safety_mode(&mut self,
                                               span: Span,
                                               safety_mode: BlockSafety)
    {
        debug!("update_visibility_scope_for({:?}, {:?})", span, safety_mode);
        let new_unsafety = match safety_mode {
            BlockSafety::Safe => None,
            BlockSafety::ExplicitUnsafe(node_id) => {
                assert_eq!(self.push_unsafe_count, 0);
                match self.unpushed_unsafe {
                    Safety::Safe => {}
                    _ => return
                }
                self.unpushed_unsafe = Safety::ExplicitUnsafe(node_id);
                Some(Safety::ExplicitUnsafe(node_id))
            }
            BlockSafety::PushUnsafe => {
                self.push_unsafe_count += 1;
                Some(Safety::BuiltinUnsafe)
            }
            BlockSafety::PopUnsafe => {
                self.push_unsafe_count =
                    self.push_unsafe_count.checked_sub(1).unwrap_or_else(|| {
                        span_bug!(span, "unsafe count underflow")
                    });
                if self.push_unsafe_count == 0 {
                    Some(self.unpushed_unsafe)
                } else {
                    None
                }
            }
        };

        if let Some(unsafety) = new_unsafety {
            self.visibility_scope = self.new_visibility_scope(
                span, LintLevel::Inherited, Some(unsafety));
        }
    }
}
